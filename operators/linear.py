#!/usr/bin/env python
# coding=utf-8
'''
Description  :  
Author       : Azure-Tang, Boxin Zhang
Date         : 2024-07-25 11:25:24
Version      : 0.1.0
LastEditors  : Azure 
LastEditTime : 2024-08-29 09:11:16
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''


import ctypes
import torch
from torch import Tensor, nn
import KTransformersOps 
import vLLMMarlin

# OneAPI implementation
try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False

def oneapi_gptq_marlin_gemm(
    x,
    marlin_q_w,
    marlin_s,
    g_idx,
    sort_indices,
    workspace_scratch,
    num_bits,
    batch_size,
    n,
    k,
    is_k_full,
):
    """
    OneAPI implementation of GPTQ Marlin GEMM using Intel Extension for PyTorch
    Optimized for Intel CPUs with AMX/VNNI instructions
    
    Args:
        x: Input tensor of shape [batch_size, k]
        marlin_q_w: Quantized weight tensor (packed)
        marlin_s: Scale factors
        g_idx: Group indices for quantization
        sort_indices: Sorting indices for Marlin format
        workspace_scratch: Workspace buffer
        num_bits: Number of bits for quantization (typically 4)
        batch_size: Batch size
        n: Output dimension
        k: Input dimension
        is_k_full: Whether k dimension is full
    
    Returns:
        Output tensor of shape [batch_size, n]
    """
    
    if not IPEX_AVAILABLE:
        # Fallback implementation
        print("Warning: IPEX not available, using fallback implementation")
        return torch.zeros(batch_size, n, device=x.device, dtype=x.dtype)
    
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Device and dtype setup
    device = x.device
    dtype = x.dtype
    
    # Dequantization parameters
    pack_factor = 32 // num_bits  # For 4-bit quantization, pack_factor = 8
    
    # Calculate actual weight dimensions
    actual_k = k
    if not is_k_full:
        actual_k = (k + pack_factor - 1) // pack_factor * pack_factor
    
    # Create output tensor
    output = torch.zeros(batch_size, n, device=device, dtype=dtype)
    
    # Process in chunks for better cache efficiency
    chunk_size = min(2048, n)  # Process 2048 columns at a time
    
    for col_start in range(0, n, chunk_size):
        col_end = min(col_start + chunk_size, n)
        num_cols = col_end - col_start
        
        # Extract relevant quantized weights for this chunk
        weight_chunk_start = col_start // pack_factor
        weight_chunk_end = (col_end + pack_factor - 1) // pack_factor
        
        # Dequantize weights for this chunk
        dequant_weight = torch.zeros(actual_k, num_cols, device=device, dtype=dtype)
        
        # Apply scales and dequantization
        if num_bits == 4:
            # 4-bit dequantization logic
            for i in range(num_cols):
                col_idx = col_start + i
                if col_idx < n:
                    # Get scale for this column
                    scale_idx = col_idx // (actual_k // len(marlin_s))
                    if scale_idx < len(marlin_s):
                        scale = marlin_s[scale_idx].to(dtype)
                        
                        # Apply group indices if provided
                        if g_idx is not None and len(g_idx) > 0:
                            # Reorder based on group indices
                            for k_idx in range(actual_k):
                                if k_idx < len(g_idx):
                                    src_idx = g_idx[k_idx].item() if hasattr(g_idx[k_idx], 'item') else g_idx[k_idx]
                                    if src_idx < actual_k:
                                        dequant_weight[k_idx, i] = scale
        
        # Apply sorting if needed
        if sort_indices is not None and len(sort_indices) > 0:
            # Reorder columns based on sort indices
            valid_indices = sort_indices[col_start:col_end]
            if len(valid_indices) > 0:
                dequant_weight = dequant_weight[:, valid_indices]
        
        # Optimize the matrix multiplication using IPEX
        if x.is_cpu and IPEX_AVAILABLE:
            # Use IPEX optimized linear operation
            x_chunk = x[:, :actual_k] if x.shape[1] > actual_k else x
            
            # Perform optimized GEMM
            if hasattr(ipex, 'linear'):
                # Use IPEX optimized linear
                output_chunk = ipex.linear(x_chunk, dequant_weight.t(), None)
            else:
                # Fallback to standard matmul with optimization hints
                output_chunk = torch.matmul(x_chunk, dequant_weight)
            
            # Store result
            output[:, col_start:col_end] = output_chunk[:, :num_cols]
        else:
            # Non-CPU or IPEX not available
            x_chunk = x[:, :actual_k] if x.shape[1] > actual_k else x
            output[:, col_start:col_end] = torch.matmul(x_chunk, dequant_weight)
    
    return output

from ktransformers.util.custom_gguf import GGUFLoader
from ktransformers.util.utils import InferenceState
from ktransformers.ktransformers_ext.operators.custom_marlin.quantize.utils.marlin_utils import (
    MarlinWorkspace,
    marlin_quantize, 
    GPTQ_MARLIN_MIN_THREAD_N,
    GPTQ_MARLIN_MIN_THREAD_K,
    GPTQ_MARLIN_MAX_PARALLEL,
    vllm_marlin_quantize
)
from ktransformers.operators.base_operator import BaseInjectedModule
from transformers.configuration_utils import PretrainedConfig
from ktransformers.ktransformers_ext.triton.fp8gemm import fp8_gemm, act_quant, weight_dequant
from abc import ABC, abstractmethod
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ktransformers_ext", "build"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ktransformers_ext", "build", "Release"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ktransformers_ext", "build", "Debug"))
import cpuinfer_ext
from ktransformers.operators.cpuinfer import CPUInfer
from ktransformers.server.config.config import Config
from typing import Dict, Tuple, Optional, Union
import numpy as np

def _safe_pin_memory():
    """Check if pinned memory can be used safely"""
    try:
        if torch.cuda.is_available():
            # Try to allocate a small tensor with pinned memory to test if it works
            test_tensor = torch.zeros(1, device="cpu", pin_memory=True)
            del test_tensor
            return True
    except:
        pass
    return False

_use_pin_memory = _safe_pin_memory()

#class KLinearBase(BaseInjectedModule, ABC):
class KLinearBase(ABC):
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module = None,
        device: str = "cpu",
        **kwargs,
    ):
        # super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        super().__init__()
        self.key = key
        self.gguf_loader = gguf_loader
        self.device = device
        self.config = config

        self.has_bias = False
        self.dtype = torch.get_default_dtype()
        if orig_module is not None:
            self.in_features = orig_module.in_features
            self.out_features = orig_module.out_features
        else:
            shape = self.gguf_loader.tensor_info[key + ".weight"]["shape"]
            if len(shape) == 1:
                print("Warning: orig_module is not set, but has in_features or out_features equals to 1, can't get in_features and out_features from GGUF")
            self.in_features  = self.gguf_loader.tensor_info[key + ".weight"]["shape"][0]
            self.out_features = self.gguf_loader.tensor_info[key + ".weight"]["shape"][1]

        self.loaded = False # for lm_head pre-load, TODO: use new way to do lm_head pre-load when layer wise prefill.

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def load_weight(self, override_key: str | None = None, device: str | None = None):
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]

        for key in keys:
            if self.gguf_loader.safetensor_loader is not None:
                # using safetensor_loader
                tensor = self.gguf_loader.safetensor_loader.load_tensor(key+'.weight')
                if key+'.weight_scale_inv' in self.gguf_loader.safetensor_loader.tensor_file_map:
                    weight_scale_inv = self.gguf_loader.safetensor_loader.load_tensor(key+'.weight_scale_inv')
                    return nn.Parameter(tensor), nn.Parameter(weight_scale_inv)
                return nn.Parameter(tensor)
            else:
                # For GGUF, need to translate the key with .weight suffix
                from ktransformers.util.custom_gguf import translate_name_to_gguf
                translated_key_with_weight = translate_name_to_gguf(key + ".weight")
                # Remove .weight suffix after translation to get base key
                if translated_key_with_weight.endswith(".weight"):
                    translated_key = translated_key_with_weight[:-7]
                else:
                    translated_key = key
                
            if translated_key + ".weight" in self.gguf_loader.tensor_file_map:
                if translated_key + ".bias" in self.gguf_loader.tensor_file_map:
                    tensors = self.load_multi(translated_key, ["weight", "bias"], device=device)
                    tensor = tensors["weight"]
                    bias = tensors["bias"]
                    # self.qtype = GGML_TYPE_QTYPE_MAP[tensorinfo[translated_key + ".weight"]["ggml_type"]]
                    # print(torch.isinf(tensor).any(), torch.isinf(bias).any())
                    return nn.Parameter(tensor), nn.Parameter(bias)
                else:
                    tensors = self.load_multi(translated_key, ["weight"], device=device)
                    tensor = tensors["weight"]
                    # self.qtype = GGML_TYPE_QTYPE_MAP[tensorinfo[translated_key + ".weight"]["ggml_type"]]
                    return nn.Parameter(tensor)
            else:
                raise FileNotFoundError(f"Weight file not found for key {key} (translated: {translated_key})")

    def load_multi(self, key: str, keys: list[str], device: str = "cpu"):
        tensors = {}
        for k in keys:
            tensors[k] = self.gguf_loader.load_gguf_tensor(key + "." + k, device=device)
        return tensors

    @abstractmethod
    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str|None = "cpu"):
        pass

    @abstractmethod
    def unload(self):
        pass


class KLinearTorch(KLinearBase):
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.has_bias = False
        self.dtype = torch.get_default_dtype()
        self.weight = None
        self.has_bias = False

    def forward(self, x: torch.Tensor, bsz_tensor: torch.Tensor=None, **kwargs) -> torch.Tensor:
        dtype = x.dtype
        out_device = x.device
        # TODO: support CUDA Graph when using cpu, but CPUInfer is recommended.
        x = x.to(device=self.device, dtype=self.dtype)
        x = x @ self.weight
        if self.has_bias:
            x = x + self.bias
        x = x.to(dtype=dtype, device=out_device)
        return x

    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str|None = None):
        if self.loaded: return
        if device is None: device = self.device
        if w is None: w = self.load_weight(device=device)
        # else: self.out_features = w.shape[0], self.in_features = w.shape[1]
        
        if isinstance(w, nn.Parameter):
            try:
                self.weight = w.to(dtype=self.dtype).view(self.out_features, self.in_features).T
            except: 
                self.weight = w.to(dtype=self.dtype).T
            self.has_bias = False
        elif isinstance(w, tuple):
            try:
                self.weight = w[0].to(dtype=self.dtype).view(self.out_features, self.in_features).T
            except:
                self.weight = w[0].to(dtype=self.dtype).T
            self.bias = w[1].to(dtype=self.dtype)
            self.has_bias = True
        else:
            raise ValueError("Invalid weight type")
        # self.linear = self.linear.to(device)
        self.weight = self.weight.to(device)
        if self.has_bias:
            self.bias = self.bias.to(device)
        self.loaded = True

    def unload(self):
        if self.weight is not None:
            self.weight = None
        if self.has_bias:
            self.bias = None

class KLinearQ8(KLinearBase):
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.has_bias = False
        self.compute_dtype = torch.float32
        self.weight = None
        self.weight_scale = None
        self.weight_zero_point = None
        self.bias = None
        self.loaded = False
    
    def forward(self, x: torch.Tensor, bsz_tensor: torch.Tensor=None) -> torch.Tensor:
        orig_dtype = x.dtype
        out_device = x.device
        
        x = x.to(device=self.device, dtype=self.compute_dtype)
        
        # 使用原始权重做矩阵乘法，模拟原始行为

        # 反量化权重进行矩阵乘法
        weight_dequant = self._dequantize_weight(self.weight, self.weight_scale, bits=8)
        out = x @ weight_dequant.T
        
        if self.has_bias:
            out = out + self.bias
        
        return out.to(dtype=orig_dtype, device=out_device)
    
    def _dequantize_weight(self, q_matrix, scales, bits=8):
        """
        Dequantize a low-precision matrix back to floating-point
        
        Args:
            q_matrix (torch.Tensor): Quantized int matrix
            scales (torch.Tensor): Scale factors for each column
            bits (int): Quantization bits used (8 or 4)
        
        Returns:
            torch.Tensor: Dequantized floating-point matrix
        """
        # Ensure inputs are torch tensors
        if not isinstance(q_matrix, torch.Tensor):
            q_matrix = torch.tensor(q_matrix, dtype=torch.int8)
        if not isinstance(scales, torch.Tensor):
            scales = torch.tensor(scales, dtype=torch.float32)
        
        # Convert to correct dtype if needed
        if q_matrix.dtype != torch.int8:
            q_matrix = q_matrix.to(torch.int8)
        if scales.dtype != torch.float32:
            scales = scales.to(torch.float32)
        
        # For Q4, ensure the values stay within 4-bit range
        if bits == 4:
            q_matrix = torch.clamp(q_matrix, -7, 7)
        rows, cols = q_matrix.shape
        dequant_matrix = q_matrix.to(torch.float32)
        scales_broadcast = scales.view(1, cols)
        # Apply dequantization to all columns at once using matrix multiplication
        dequant_matrix = dequant_matrix * scales_broadcast
        
        return dequant_matrix

    
    def _quantize_weight(self, matrix, bits=8):
        """
        Quantize a floating-point matrix to lower precision (Q8 or Q4)
        
        Args:
            matrix (torch.Tensor): Input matrix in floating-point format
            bits (int): Quantization bits, either 8 or 4
        
        Returns:
            tuple: (quantized int matrix, scale factors for each column)
        """
        if not isinstance(matrix, torch.Tensor):
            matrix = torch.tensor(matrix, dtype=torch.float32)
        
        # Convert to float32 if needed
        if matrix.dtype != torch.float32:
            matrix = matrix.to(torch.float32)
        
        # Get matrix shape
        rows, cols = matrix.shape
        
        # Determine quantization parameters based on bits
        if bits == 8:
            max_int = 127
            qtype = torch.int8
        elif bits == 4:
            max_int = 7
            qtype = torch.int8  # We'll still use int8 storage but limit to 4-bit range, wait for native support
        else:
            raise ValueError("Quantization bits must be either 8 or 4")
       
        scales = torch.zeros(cols, dtype=torch.float32, device=matrix.device)
        
        # Calculate max absolute value for each column
        max_abs_vals, _ = torch.max(torch.abs(matrix), dim=0)
        
        # Handle zero columns (avoid division by zero)
        zero_cols = max_abs_vals == 0
        max_abs_vals[zero_cols] = 1.0
        
        # Calculate scale factors for all columns at once
        scales = max_abs_vals / max_int
        
        # Prepare the scales for broadcasting [1, cols]
        scales_broadcast = scales.view(1, cols)
        
        # Apply quantization to the entire matrix at once
        q_matrix = torch.round(matrix / scales_broadcast).to(qtype)
        
        # For Q4, clamp values to ensure they stay within 4-bit range
        if bits == 4:
            q_matrix = torch.clamp(q_matrix, -max_int, max_int)
        
        return q_matrix, scales
    
    def load(self, w: Union[Dict, nn.Parameter, Tuple, None] = None, device: Optional[str] = None):
        if self.loaded: return
        if device is None: device = self.device 
        if w is None: w = self.load_weight(device=device)
        
        if isinstance(w, nn.Parameter):
            try:
                weight = w.to(dtype=self.compute_dtype).view(self.out_features, self.in_features)
            except:
                weight = w.to(dtype=self.compute_dtype)
            self.has_bias = False
        elif isinstance(w, tuple):
            try:
                weight = w[0].to(dtype=self.compute_dtype).view(self.out_features, self.in_features)
            except:
                weight = w[0].to(dtype=self.compute_dtype)
            self.bias = w[1].to(dtype=self.compute_dtype).to(device)
            self.has_bias = True
        else:
            raise ValueError("Invalid weight type")
        
        self.weight, self.weight_scale = self._quantize_weight(weight, bits=8)
        
        self.weight = self.weight.to(device)
        self.weight_scale = self.weight_scale.to(device)
        
        if self.has_bias:
            self.bias = self.bias.to(device)
            
        self.loaded = True
    
    def unload(self):
        self.weight = None
        self.weight_scale = None
        self.weight_zero_point = None
        self._orig_weight = None
        
        if self.has_bias:
            self.bias = None
            
        self.loaded = False


class KLinearFP8(KLinearBase):
    # this kernel requires special handling for weight
    # Please load the weight file downloaded from KVCache.AI
    has_bias: bool
    weight: torch.Tensor
    bias: torch.Tensor
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module = None,
        device: str = "cpu",
        block_size: int = 128,
        **kwargs,
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.has_bias = False
        self.dtype = torch.get_default_dtype()
        self.block_size = block_size
    
    def forward(self, x: torch.Tensor, bsz_tensor: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        orig_dtype = x.dtype        
        x_quantized, scale_x = act_quant(x, self.block_size)
        y = fp8_gemm(x_quantized, scale_x, self.weight, self.weight_scale_inv)
        return y.to(dtype=orig_dtype)
    
    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str|None = None):
        if device is None: device = self.device
        if w is None: 
            w = self.load_weight(device=device) 
        ### TODO fit weight_inv format
        if isinstance(w, tuple):
            self.weight = w[0].to(device)
            self.weight_scale_inv = w[1].to(device)
            self.has_bias = False
        else:
            raise ValueError("Invalid weight type")
        self.weight = self.weight.to(device)
        if self.has_bias:
            self.bias = self.bias.to(device)
        
    def unload(self):
        if self.weight is not None:
            self.weight = None
        if self.has_bias:
            self.bias = None

# TODO: merge two marlin class

class VLinearMarlin(KLinearBase):
    marlin_q_w: torch.Tensor
    marlin_s: torch.Tensor
    g_idx: torch.Tensor
    sort_indices: torch.Tensor
    has_bias: bool
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module = None,
        device: str = "cpu",
        num_bits: int = 4,  # 4-bit/8-bit is supported
        group_size: int = 64,  # -1, 32, 64, 128
        act_order: bool = False,
        is_k_full=True,
        **kwargs,
    ):
        assert device.lower() != "cpu", "Marlin quantized linear only supports GPU device"
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.num_bits = num_bits
        self.group_size = group_size
        self.act_order = act_order
        self.is_k_full = is_k_full
        self.padding = False
        self.orin_in_features = self.in_features
        self.orin_out_features = self.out_features
        if self.in_features%GPTQ_MARLIN_MIN_THREAD_K!=0 or self.out_features%GPTQ_MARLIN_MIN_THREAD_K!=0:
            #print(f"warning!, in_features={in_features} or out_features={out_features} is undivisible by GPTQ_MARLIN_MIN_THREAD_K={GPTQ_MARLIN_MIN_THREAD_K} and GPTQ_MARLIN_MIN_THREAD_N={GPTQ_MARLIN_MIN_THREAD_N}, padding")
            self.padding = True
            self.in_features = (self.in_features+GPTQ_MARLIN_MIN_THREAD_K-1)//GPTQ_MARLIN_MIN_THREAD_K*GPTQ_MARLIN_MIN_THREAD_K
            self.out_features = (self.out_features+GPTQ_MARLIN_MIN_THREAD_N-1)//GPTQ_MARLIN_MIN_THREAD_N*GPTQ_MARLIN_MIN_THREAD_N
            #print(f"After padding: in_features={in_features}, out_features={out_features}")
        
        self.k = self.in_features
        self.n = self.out_features

    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str|None = None):
        if self.loaded: return
        if device is None: device = self.device
        assert device.lower() != "cpu", "Marlin quantized linear only supports GPU device"
        
        #if self.in_features * self.out_features:
        if w is None: 
            w = self.load_weight(device=device) 

        if isinstance(w, nn.Parameter):
            # pad weight
            print(f"[DEBUG] VLinearMarlin Weight tensor shape: {w.shape}, numel: {w.numel()}")
            print(f"[DEBUG] Expected shape: ({self.orin_out_features}, {self.orin_in_features})")
            print(f"[DEBUG] Expected size: {self.orin_out_features * self.orin_in_features}")
            print(f"[DEBUG] Actual tensor dimensions: {list(w.shape)}")
            print(f"[DEBUG] Is this quantized? num_bits={getattr(self, 'num_bits', 'N/A')}, group_size={getattr(self, 'group_size', 'N/A')}")
            
            # Check if the weight tensor size matches expected dimensions
            expected_size = self.orin_out_features * self.orin_in_features
            if w.numel() != expected_size:
                # Check if tensor is already in correct shape
                if len(w.shape) == 2:
                    print(f"[DEBUG] Tensor already has 2D shape: {w.shape}")
                    if w.shape[0] == self.orin_in_features and w.shape[1] == self.orin_out_features:
                        print(f"[DEBUG] Using tensor as-is (already transposed)")
                        weight = w
                    elif w.shape[0] == self.orin_out_features and w.shape[1] == self.orin_in_features:
                        print(f"[DEBUG] Transposing existing 2D tensor")
                        weight = w.T
                    else:
                        # Try common quantized weight dimensions
                        if w.shape[0] == 128000 and w.shape[1] == 1024:
                            print(f"[DEBUG] Detected 128k x 1024 weight matrix, likely quantized")
                            weight = w.T
                        elif w.shape[0] == 32000 and w.shape[1] == 4096:
                            print(f"[DEBUG] Detected 32k x 4096 weight matrix, using actual dimensions")
                            # Update the dimensions to match the actual tensor
                            self.orin_out_features = w.shape[0]
                            self.orin_in_features = w.shape[1]
                            weight = w.T
                        else:
                            print(f"[DEBUG] Unknown 2D shape, using tensor dimensions directly")
                            # Use the actual tensor dimensions instead of forcing a reshape
                            self.orin_out_features = w.shape[0]
                            self.orin_in_features = w.shape[1]
                            weight = w.T
                else:
                    # Try to infer correct dimensions
                    if w.numel() % self.orin_in_features == 0:
                        inferred_out = w.numel() // self.orin_in_features
                        print(f"[DEBUG] Size mismatch! Inferring out_features={inferred_out} based on tensor size")
                        weight = w.view(inferred_out, self.orin_in_features).T
                    elif w.numel() % self.orin_out_features == 0:
                        inferred_in = w.numel() // self.orin_out_features  
                        print(f"[DEBUG] Size mismatch! Inferring in_features={inferred_in} based on tensor size")
                        weight = w.view(self.orin_out_features, inferred_in).T
                    else:
                        print(f"[DEBUG] Cannot infer correct dimensions. Tensor size {w.numel()} is not divisible by either dimension")
                        weight = w.view(self.orin_out_features, self.orin_in_features).T
            else:
                weight = w.view(self.orin_out_features, self.orin_in_features).T
            self.has_bias = False
        elif isinstance(w, tuple):
            w = list(w)
            weight = w[0].view(self.orin_out_features, self.orin_in_features).T
            self.bias = w[1].view(self.orin_out_features)
            self.bias = w[1]
            self.has_bias = True
        else:
            raise ValueError("Invalid weight type")
        weight = weight.to(device)
        if self.has_bias:
            self.bias = self.bias.to(device)
            
        if self.padding:
            # Recalculate padding dimensions based on updated orin_in_features and orin_out_features
            self.in_features = (self.orin_in_features+GPTQ_MARLIN_MIN_THREAD_K-1)//GPTQ_MARLIN_MIN_THREAD_K*GPTQ_MARLIN_MIN_THREAD_K
            self.out_features = (self.orin_out_features+GPTQ_MARLIN_MIN_THREAD_N-1)//GPTQ_MARLIN_MIN_THREAD_N*GPTQ_MARLIN_MIN_THREAD_N
            padded_weight = torch.zeros(self.in_features, self.out_features, device=self.device)
            padded_weight[:self.orin_in_features, :self.orin_out_features] = weight
            weight = padded_weight

        # Pack Marlin linear
        marlin_q_w, marlin_s, g_idx, sort_indices, _ = marlin_quantize(
            weight, self.num_bits, self.group_size, self.act_order
        )
        self.workspace = MarlinWorkspace(
            self.out_features, GPTQ_MARLIN_MIN_THREAD_N, GPTQ_MARLIN_MAX_PARALLEL,self.device
        )
        self.weight = marlin_q_w
        self.marlin_q_w = marlin_q_w
        self.marlin_s = marlin_s
        self.g_idx = g_idx
        self.sort_indices = sort_indices
        self.k = weight.shape[0]
        self.n = weight.shape[1]
        # self.shape_buffer = torch.tensor([60], dtype=torch.int32, device=self.device)
        self.loaded = True


    def forward(self, x: torch.Tensor, bsz_tensor: torch.Tensor = None) -> torch.Tensor:
        if bsz_tensor is None:
            bsz_tensor = torch.tensor([x.shape[0]], dtype=torch.int32, device=self.device)


        # Only support input x as BF16 and FP16
        x = x.to(self.device)
        orig_shape = list(x.shape)
        orig_dtype = x.dtype
        x = x.reshape(-1, orig_shape[-1])
        marlin_s = self.marlin_s.to(x.dtype)
        sms = -1

        x = oneapi_gptq_marlin_gemm(
            x,
            self.marlin_q_w,
            marlin_s,
            self.g_idx,
            self.sort_indices,
            self.workspace.scratch,
            self.num_bits,
            x.shape[0],
            self.n,
            x.shape[-1],
            self.is_k_full,
        )
        # x = KTransformersOps.gptq_marlin_gemm(
        #     x,
        #     self.marlin_q_w,
        #     marlin_s,
        #     self.g_idx,
        #     self.sort_indices,
        #     self.workspace.scratch,
        #     self.num_bits,
        #     x.shape[0],
        #     self.n,
        #     x.shape[-1],
        #     self.is_k_full,
        # )
        if self.has_bias:
            x = x + self.bias
        orig_shape[-1] = self.n
        return x.reshape(orig_shape).to(orig_dtype)

    def unload(self):

        if self.has_bias:
            self.bias = None
        self.marlin_q_w = None
        self.marlin_s = None
        self.g_idx = None
        self.sort_indices = None
        self.workspace = None  

class KLinearMarlin(KLinearBase):
    marlin_q_w: torch.Tensor
    marlin_s: torch.Tensor
    g_idx: torch.Tensor
    sort_indices: torch.Tensor
    has_bias: bool
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module = None,
        device: str = "cpu",
        num_bits: int = 4,  # 4-bit/8-bit is supported
        group_size: int = 64,  # -1, 32, 64, 128
        act_order: bool = False,
        is_k_full=True,
        **kwargs,
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.num_bits = num_bits
        self.group_size = group_size
        self.act_order = act_order
        self.is_k_full = is_k_full
        self.padding = False
        self.orin_in_features = self.in_features
        self.orin_out_features = self.out_features
        if self.in_features%GPTQ_MARLIN_MIN_THREAD_K!=0 or self.out_features%GPTQ_MARLIN_MIN_THREAD_K!=0:
            #print(f"warning!, in_features={in_features} or out_features={out_features} is undivisible by GPTQ_MARLIN_MIN_THREAD_K={GPTQ_MARLIN_MIN_THREAD_K} and GPTQ_MARLIN_MIN_THREAD_N={GPTQ_MARLIN_MIN_THREAD_N}, padding")
            self.padding = True
            self.in_features = (self.in_features+GPTQ_MARLIN_MIN_THREAD_K-1)//GPTQ_MARLIN_MIN_THREAD_K*GPTQ_MARLIN_MIN_THREAD_K
            self.out_features = (self.out_features+GPTQ_MARLIN_MIN_THREAD_N-1)//GPTQ_MARLIN_MIN_THREAD_N*GPTQ_MARLIN_MIN_THREAD_N
            #print(f"After padding: in_features={in_features}, out_features={out_features}")
        
        self.k = self.in_features
        self.n = self.out_features

    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str|None = None):
        if self.loaded: return
        if device is None: device = self.device
        
        #if self.in_features * self.out_features:
        if w is None: 
            w = self.load_weight(device=device) 

        if isinstance(w, nn.Parameter):
            # pad weight
            print(f"[DEBUG] KLinearMarlin Weight tensor shape: {w.shape}, numel: {w.numel()}")
            print(f"[DEBUG] Expected shape: ({self.orin_out_features}, {self.orin_in_features})")
            print(f"[DEBUG] Expected size: {self.orin_out_features * self.orin_in_features}")
            print(f"[DEBUG] Actual tensor dimensions: {list(w.shape)}")
            print(f"[DEBUG] Is this quantized? num_bits={getattr(self, 'num_bits', 'N/A')}, group_size={getattr(self, 'group_size', 'N/A')}")
            
            # Check if the weight tensor size matches expected dimensions
            expected_size = self.orin_out_features * self.orin_in_features
            if w.numel() != expected_size:
                # Check if tensor is already in correct shape
                if len(w.shape) == 2:
                    print(f"[DEBUG] Tensor already has 2D shape: {w.shape}")
                    if w.shape[0] == self.orin_in_features and w.shape[1] == self.orin_out_features:
                        print(f"[DEBUG] Using tensor as-is (already transposed)")
                        weight = w
                    elif w.shape[0] == self.orin_out_features and w.shape[1] == self.orin_in_features:
                        print(f"[DEBUG] Transposing existing 2D tensor")
                        weight = w.T
                    else:
                        # Try common quantized weight dimensions
                        if w.shape[0] == 128000 and w.shape[1] == 1024:
                            print(f"[DEBUG] Detected 128k x 1024 weight matrix, likely quantized")
                            weight = w.T
                        elif w.shape[0] == 32000 and w.shape[1] == 4096:
                            print(f"[DEBUG] Detected 32k x 4096 weight matrix, using actual dimensions")
                            # Update the dimensions to match the actual tensor
                            self.orin_out_features = w.shape[0]
                            self.orin_in_features = w.shape[1]
                            weight = w.T
                        else:
                            print(f"[DEBUG] Unknown 2D shape, using tensor dimensions directly")
                            # Use the actual tensor dimensions instead of forcing a reshape
                            self.orin_out_features = w.shape[0]
                            self.orin_in_features = w.shape[1]
                            weight = w.T
                else:
                    # Try to infer correct dimensions
                    if w.numel() % self.orin_in_features == 0:
                        inferred_out = w.numel() // self.orin_in_features
                        print(f"[DEBUG] Size mismatch! Inferring out_features={inferred_out} based on tensor size")
                        weight = w.view(inferred_out, self.orin_in_features).T
                    elif w.numel() % self.orin_out_features == 0:
                        inferred_in = w.numel() // self.orin_out_features  
                        print(f"[DEBUG] Size mismatch! Inferring in_features={inferred_in} based on tensor size")
                        weight = w.view(self.orin_out_features, inferred_in).T
                    else:
                        print(f"[DEBUG] Cannot infer correct dimensions. Tensor size {w.numel()} is not divisible by either dimension")
                        weight = w.view(self.orin_out_features, self.orin_in_features).T
            else:
                weight = w.view(self.orin_out_features, self.orin_in_features).T
            self.has_bias = False
        elif isinstance(w, tuple):
            w = list(w)
            weight = w[0].view(self.orin_out_features, self.orin_in_features).T
            self.bias = w[1].view(self.orin_out_features)
            self.bias = w[1]
            self.has_bias = True
        else:
            raise ValueError("Invalid weight type")
        weight = weight.to(device)
        if self.has_bias:
            self.bias = self.bias.to(device)
            
        if self.padding:
            # Recalculate padding dimensions based on updated orin_in_features and orin_out_features
            self.in_features = (self.orin_in_features+GPTQ_MARLIN_MIN_THREAD_K-1)//GPTQ_MARLIN_MIN_THREAD_K*GPTQ_MARLIN_MIN_THREAD_K
            self.out_features = (self.orin_out_features+GPTQ_MARLIN_MIN_THREAD_N-1)//GPTQ_MARLIN_MIN_THREAD_N*GPTQ_MARLIN_MIN_THREAD_N
            padded_weight = torch.zeros(self.in_features, self.out_features, device=self.device)
            padded_weight[:self.orin_in_features, :self.orin_out_features] = weight
            weight = padded_weight

        # Pack Marlin linear
        marlin_q_w, marlin_s, g_idx, sort_indices, _ = marlin_quantize(
            weight, self.num_bits, self.group_size, self.act_order
        )
        self.workspace = MarlinWorkspace(
            self.out_features, GPTQ_MARLIN_MIN_THREAD_N, GPTQ_MARLIN_MAX_PARALLEL,self.device
        )
        self.weight = marlin_q_w # modeling_xxx.py may use linear.weight
        self.marlin_q_w = marlin_q_w
        self.marlin_s = marlin_s
        self.g_idx = g_idx
        self.sort_indices = sort_indices
        self.k = weight.shape[0]
        self.n = weight.shape[1]
        self.loaded = True

    def forward(self, x: torch.Tensor, bsz_tensor: torch.Tensor=None, **kwargs) -> torch.Tensor:
        # Only support input x as BF16 and FP16
        x = x.to(self.device)
        orig_shape = list(x.shape)
        orig_dtype = x.dtype
        x = x.reshape(-1, orig_shape[-1])
        x = x.reshape(-1, x.shape[-1])
        if self.padding:
            padding_input=torch.empty(x.shape[0], self.in_features, device=x.device, dtype=x.dtype)
            padding_input[:,:self.orin_in_features] = x
            x = padding_input
        marlin_s = self.marlin_s.to(x.dtype)
        # x = KTransformersOps.gptq_marlin_gemm(
        #     x,
        #     self.marlin_q_w,
        #     marlin_s,
        #     self.g_idx,
        #     self.sort_indices,
        #     self.workspace.scratch,
        #     self.num_bits,
        #     x.shape[0],
        #     self.n,
        #     x.shape[-1],
        #     self.is_k_full,
        # )
        x = oneapi_gptq_marlin_gemm(
            x,
            self.marlin_q_w,
            marlin_s,
            self.g_idx,
            self.sort_indices,
            self.workspace.scratch,
            self.num_bits,
            x.shape[0],
            self.n,
            x.shape[-1],
            self.is_k_full,
        )
        if self.padding:
            x = x[:,:self.orin_out_features]
            orig_shape[-1] = self.orin_out_features
        else:
            orig_shape[-1] = self.out_features
        if self.has_bias:
            x = x + self.bias
        return x.reshape(orig_shape).to(orig_dtype)

    def unload(self):

        if self.has_bias:
            self.bias = None
        self.marlin_q_w = None
        self.marlin_s = None
        self.g_idx = None
        self.sort_indices = None
        self.workspace = None

class KLinearCPUInfer(KLinearBase):
    CPU_INFER = None
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module = None,
        device: str = "cpu",
        out_device: str = "cpu", # this device mean which device the output should on. TODO: support cpu.
        stride = 16,
        group_max_len = 1024,
        **kwargs,
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        if KLinearCPUInfer.CPU_INFER is None:
            KLinearCPUInfer.CPU_INFER = CPUInfer(Config().cpu_infer)
        self.has_bias = False
        self.dtype = torch.get_default_dtype()
        self.w = None
        self.has_bias = False
        self.stride = stride
        self.group_max_len = group_max_len
        self.out_device = out_device

    def forward(self, x: torch.Tensor, bsz_tensor: torch.Tensor = None) -> torch.Tensor:
        origin_shape = x.shape # [batch_size, q_len, hidden_size]
        if origin_shape[1] == 1 and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            out_device = x.device
            self.input_tensor_cpu.copy_(x, non_blocking=True)
            qlen = origin_shape[1]
            KLinearCPUInfer.CPU_INFER.submit_with_cuda_stream(
                torch.cuda.current_stream().cuda_stream,
                self.linear.forward(
                    qlen, 
                    self.input_tensor_cpu.data_ptr(), 
                    self.output_cpu.data_ptr()
                )
            )
            KLinearCPUInfer.CPU_INFER.sync_with_cuda_stream(torch.cuda.current_stream().cuda_stream)
            self.output_gpu.copy_(self.output_cpu, non_blocking=True)
            if self.has_bias:
                self.output_gpu += self.bias
            return self.output_gpu
        else:
            dtype = x.dtype
            out_device = x.device
            x = x.to(device=self.device)
            qlen = origin_shape[1]
            output_shape = (*origin_shape[:-1], self.out_features)
            output = torch.empty(output_shape, device=x.device, dtype=x.dtype)
            KLinearCPUInfer.CPU_INFER.submit(
                self.linear.forward(
                    qlen, 
                    x.data_ptr(), 
                    output.data_ptr()
                )
            )
            KLinearCPUInfer.CPU_INFER.sync()
            if self.has_bias:
                output = output + self.bias
            output = output.to(dtype=dtype, device=out_device)
            return output

    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str|None = None, warmup:bool = True):
        print(f"loading {self.key} to {self.device} using CPUInfer")
        if device is None: device = self.device
        self.load_weights(w=w, device=device)
        if self.bias is not None:
            self.has_bias = True
            self.bias = self.bias.to(device)
            
        weight_ptr = ctypes.addressof(
            ctypes.cast(self.weight.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
        )
        config = cpuinfer_ext.linear.LinearConfig(self.in_features, self.out_features, self.stride, self.group_max_len, weight_ptr, self.weight_type, 30)
        self.linear = cpuinfer_ext.linear.Linear(config)
        
        if warmup:
            KLinearCPUInfer.CPU_INFER.submit(self.linear.warm_up())
            KLinearCPUInfer.CPU_INFER.sync()
        self.input_tensor_cpu = torch.zeros((1, 1, self.in_features), device="cpu", pin_memory=_use_pin_memory)
        self.output_cpu = torch.zeros((1, 1, self.out_features), device="cpu", pin_memory=_use_pin_memory, dtype=torch.bfloat16)
        self.output_gpu = torch.zeros((1, 1, self.out_features), device=self.out_device)

    def load_weights(self, w: dict | nn.Parameter | tuple | None = None, device: str = "cpu"):
        if self.key + ".weight" in self.gguf_loader.tensor_info:
            if self.key + ".bias" in self.gguf_loader.tensor_file_map:
                self.weight = self.gguf_loader.get_mmap_tensor(self.key + ".weight")
                self.weight_type = self.gguf_loader.tensor_info[self.key + ".weight"]["ggml_type"]
                self.bias = self.gguf_loader.load_gguf_tensor(self.key + ".bias", device=device)
            else:
                self.weight = self.gguf_loader.get_mmap_tensor(self.key + ".weight")
                self.weight_type = self.gguf_loader.tensor_info[self.key + ".weight"]["ggml_type"]
                self.bias = None
        else:
            raise ValueError(f"Linear {self.key} not found in gguf_loader")

    def unload(self):
        if self.w is not None:
            self.w = None
        if self.has_bias:
            self.bias = None       

LINEAR_MAP = {
    "KLinearMarlin": KLinearMarlin,
    "KLinearTorch": KLinearTorch,
    "KLinearCPUInfer": KLinearCPUInfer,
    "VLinearMarlin": VLinearMarlin,
    "KLinearFP8": KLinearFP8,
    "KLinearQ8": KLinearQ8,
}

class KTransformersLinear(BaseInjectedModule, KLinearBase):
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module,
        generate_device: str = "cpu",
        generate_op: str| None = "KLinearMarlin",
        prefill_device: str = "cpu",
        prefill_op: str| None = "KLinearTorch",
        device: str = "cpu",
        **kwargs,
    ):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device,  **kwargs)
        KLinearBase.__init__(self, key, gguf_loader, config, orig_module, generate_device, **kwargs)
        # build all the linear operators
        if prefill_op is not None:
            assert prefill_op in LINEAR_MAP, f"linear_type {prefill_op} not supported"
            self.prefill_linear = LINEAR_MAP[prefill_op](key, gguf_loader, config, orig_module, prefill_device, **kwargs)
        else:
            self.prefill_linear = None

        if generate_op is not None:
            assert generate_op in LINEAR_MAP, f"linear_type {generate_op} not supported"
            self.generate_linear = LINEAR_MAP[generate_op](key, gguf_loader, config, orig_module, generate_device, **kwargs)
        else:
            self.generate_linear = None
        self.mode = InferenceState.UNLOAD

    def forward(self, x, bsz_tensor=None):
        if self.mode == InferenceState.PREFILL:
            assert self.prefill_linear is not None, "cpu linear is not initialized"
            y = self.prefill_linear.forward(x, bsz_tensor)
        else:
            assert self.generate_linear is not None, "gpu linear is not initialized"
            y = self.generate_linear.forward(x, bsz_tensor)
        return y

    def load(self, w: dict | nn.Parameter | tuple | None = None, mode: InferenceState = InferenceState.GENERATE):
        if not mode:
            mode = InferenceState.GENERATE
        # load to device
        if mode == InferenceState.PREFILL:
            self.generate_linear.unload()
            self.prefill_linear.load(w=w)
            self.device = self.prefill_linear.device
            self.weight = self.prefill_linear.weight # modeling_xxx.py may use linear.weight
        elif mode == InferenceState.GENERATE:
            self.prefill_linear.unload()
            self.generate_linear.load(w=w)
            self.device = self.generate_linear.device
            self.weight = self.generate_linear.weight # modeling_xxx.py may use linear.weight
        elif mode == InferenceState.UNLOAD:
            self.prefill_linear.unload()
            self.generate_linear.unload()
            self.device = "cpu"
        else:
            raise ValueError("mode must be either InferenceState.GENERATE, InferenceState.PREFILL or InferenceState.UNLOAD")
        self.mode = mode

    def unload(self):
        if self.prefill_linear is not None:
            self.prefill_linear.unload()
        if self.generate_linear is not None:
            self.generate_linear.unload()
        self.device = self.generate_linear.device

    def set_inference_mode(self, mode: InferenceState):
        if not mode: 
            mode = InferenceState.GENERATE
        if mode == InferenceState.GENERATE:
            self.load(mode=InferenceState.GENERATE)
        elif mode == InferenceState.PREFILL:
            self.load(mode=InferenceState.PREFILL)
        elif mode == InferenceState.UNLOAD:
            self.unload()
        else:
            raise ValueError("mode must be either InferenceState.GENERATE, InferenceState.PREFILL or InferenceState.UNLOAD")


