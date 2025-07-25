"""
CXL-aware memory manager with zero-copy operations and duplex memory allocation
Optimized for CXL memory systems with NUMA awareness
"""

import torch
import torch.nn as nn
import numpy as np
import os
import psutil
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import mmap
import ctypes
from concurrent.futures import ThreadPoolExecutor

# Try to import numa library for CXL/NUMA awareness
try:
    import numa
    NUMA_AVAILABLE = True
except ImportError:
    NUMA_AVAILABLE = False

class MemoryTier(Enum):
    """Memory tier classification for CXL systems"""
    LOCAL_DRAM = 0      # Local CPU DRAM
    CXL_NEAR = 1        # CXL near memory (low latency)
    CXL_FAR = 2         # CXL far memory (high capacity)
    PERSISTENT = 3      # Persistent memory

@dataclass
class CXLMemoryRegion:
    """Represents a CXL memory region"""
    node_id: int
    tier: MemoryTier
    size: int
    available: int
    latency_ns: int  # Access latency in nanoseconds
    bandwidth_gbps: float

class CXLMemoryAllocator:
    """CXL-aware memory allocator with zero-copy support"""
    
    def __init__(self):
        self.memory_regions = self._detect_cxl_topology()
        self.allocated_tensors: Dict[str, Tuple[torch.Tensor, MemoryTier]] = {}
        self.memory_maps: Dict[str, mmap.mmap] = {}
        
    def _detect_cxl_topology(self) -> List[CXLMemoryRegion]:
        """Detect CXL memory topology"""
        regions = []
        
        if NUMA_AVAILABLE:
            # Get NUMA node information
            num_nodes = numa.get_max_node() + 1
            for node in range(num_nodes):
                # Check if this is a CXL node (typically nodes > 0)
                is_cxl = node > 0
                
                # Get memory info for this node
                node_size = numa.node_size(node)
                
                # Estimate tier based on node distance
                distance = numa.distance(0, node)
                if distance <= 10:
                    tier = MemoryTier.LOCAL_DRAM
                    latency = 100  # ~100ns for local DRAM
                    bandwidth = 100.0
                elif distance <= 20:
                    tier = MemoryTier.CXL_NEAR
                    latency = 200  # ~200ns for CXL near
                    bandwidth = 50.0
                else:
                    tier = MemoryTier.CXL_FAR
                    latency = 500  # ~500ns for CXL far
                    bandwidth = 25.0
                
                regions.append(CXLMemoryRegion(
                    node_id=node,
                    tier=tier,
                    size=node_size[0] if node_size else 0,
                    available=node_size[1] if node_size else 0,
                    latency_ns=latency,
                    bandwidth_gbps=bandwidth
                ))
        else:
            # Fallback: assume single DRAM region
            mem = psutil.virtual_memory()
            regions.append(CXLMemoryRegion(
                node_id=0,
                tier=MemoryTier.LOCAL_DRAM,
                size=mem.total,
                available=mem.available,
                latency_ns=100,
                bandwidth_gbps=100.0
            ))
        
        return regions
    
    def allocate_tensor_cxl(self, shape: Tuple, dtype: torch.dtype, 
                           tier: MemoryTier = MemoryTier.CXL_NEAR,
                           zero_copy: bool = True) -> torch.Tensor:
        """Allocate tensor in specific CXL memory tier with zero-copy support"""
        
        # Calculate required size
        numel = np.prod(shape)
        dtype_size = torch.finfo(dtype).bits // 8 if dtype.is_floating_point else torch.iinfo(dtype).bits // 8
        size_bytes = numel * dtype_size
        
        # Find suitable memory region
        suitable_region = None
        for region in self.memory_regions:
            if region.tier == tier and region.available >= size_bytes:
                suitable_region = region
                break
        
        if not suitable_region:
            # Fallback to any available region
            for region in sorted(self.memory_regions, key=lambda r: r.latency_ns):
                if region.available >= size_bytes:
                    suitable_region = region
                    break
        
        if not suitable_region:
            raise MemoryError(f"Cannot allocate {size_bytes} bytes in CXL memory")
        
        if zero_copy and NUMA_AVAILABLE:
            # Allocate on specific NUMA node with zero-copy
            numa.set_preferred(suitable_region.node_id)
            
            # Use mmap for zero-copy allocation
            buffer = mmap.mmap(-1, size_bytes)
            
            # Create tensor from buffer without copying
            storage = torch.FloatStorage.from_buffer(buffer, 'native')
            tensor = torch.tensor([], dtype=dtype).set_(storage, 0, shape, tuple([1] * len(shape)))
            
            # Pin memory to prevent swapping
            if hasattr(torch.cuda, 'cudart'):
                torch.cuda.cudart().cudaHostRegister(
                    tensor.data_ptr(), size_bytes, 0
                )
        else:
            # Regular allocation
            tensor = torch.empty(shape, dtype=dtype)
        
        return tensor

class ZeroCopyTensorView:
    """Provides zero-copy views of tensors for in-place operations"""
    
    @staticmethod
    def create_strided_view(tensor: torch.Tensor, offset: int, shape: Tuple, 
                           stride: Optional[Tuple] = None) -> torch.Tensor:
        """Create a zero-copy strided view of a tensor"""
        if stride is None:
            stride = tuple([1] * len(shape))
        
        # Create view without copying data
        return tensor.as_strided(shape, stride, offset)
    
    @staticmethod
    def create_transposed_view(tensor: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
        """Create zero-copy transposed view"""
        return tensor.transpose(dim0, dim1)
    
    @staticmethod
    def create_reshaped_view(tensor: torch.Tensor, shape: Tuple) -> torch.Tensor:
        """Create zero-copy reshaped view"""
        return tensor.view(shape)

class CXLAwareKCustomAttention(nn.Module):
    """CXL-aware custom attention with zero-copy operations"""
    
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.allocator = CXLMemoryAllocator()
        
        # Attention parameters
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_seq_len = kwargs.get('max_seq_len', 32768)
        self.block_size = kwargs.get('block_size', 128)
        self.local_windows_len = kwargs.get('local_windows_len', 1024)
        self.topk = kwargs.get('topk', 16)
        
        # Pre-allocate CXL memory for KV cache (zero-copy)
        self.cache_k = self.allocator.allocate_tensor_cxl(
            (self.max_seq_len, self.num_heads, self.head_dim),
            dtype=torch.float16,
            tier=MemoryTier.CXL_NEAR,
            zero_copy=True
        )
        self.cache_v = self.allocator.allocate_tensor_cxl(
            (self.max_seq_len, self.num_heads, self.head_dim),
            dtype=torch.float16,
            tier=MemoryTier.CXL_NEAR,
            zero_copy=True
        )
        
        # Pre-allocate workspace in local DRAM for computation
        self.workspace = self.allocator.allocate_tensor_cxl(
            (self.block_size, self.num_heads, self.head_dim),
            dtype=torch.float16,
            tier=MemoryTier.LOCAL_DRAM,
            zero_copy=True
        )
        
        # Initialize sparse pattern buffer in CXL far memory
        self.sparse_pattern = self.allocator.allocate_tensor_cxl(
            (self.max_seq_len // self.block_size, self.max_seq_len // self.block_size),
            dtype=torch.bool,
            tier=MemoryTier.CXL_FAR,
            zero_copy=True
        )
    
    def forward(self, query, key, value, position_ids, attention_mask=None, **kwargs):
        """Forward pass with zero-copy operations"""
        batch_size, seq_len = query.shape[:2]
        
        # Zero-copy reshape operations
        q = ZeroCopyTensorView.create_reshaped_view(
            query, (batch_size, seq_len, self.num_heads, self.head_dim)
        )
        k = ZeroCopyTensorView.create_reshaped_view(
            key, (batch_size, seq_len, self.num_heads, self.head_dim)
        )
        v = ZeroCopyTensorView.create_reshaped_view(
            value, (batch_size, seq_len, self.num_heads, self.head_dim)
        )
        
        # In-place transpose for attention computation
        q = ZeroCopyTensorView.create_transposed_view(q, 1, 2)
        k = ZeroCopyTensorView.create_transposed_view(k, 1, 2)
        v = ZeroCopyTensorView.create_transposed_view(v, 1, 2)
        
        # Update KV cache in-place (zero-copy)
        cache_position = position_ids[0]
        self.cache_k[cache_position:cache_position+seq_len].copy_(k[0], non_blocking=True)
        self.cache_v[cache_position:cache_position+seq_len].copy_(v[0], non_blocking=True)
        
        # Compute attention scores with block-sparse pattern
        attn_output = self._compute_sparse_attention_inplace(q, cache_position, seq_len)
        
        # Zero-copy reshape back
        attn_output = ZeroCopyTensorView.create_transposed_view(attn_output, 1, 2)
        attn_output = ZeroCopyTensorView.create_reshaped_view(
            attn_output, (batch_size, seq_len, self.hidden_size)
        )
        
        return attn_output, None, None
    
    def _compute_sparse_attention_inplace(self, q, cache_pos, seq_len):
        """Compute sparse attention with in-place operations"""
        batch_size = q.shape[0]
        
        # Pre-allocate output buffer
        output = torch.zeros_like(q)
        
        # Process in blocks for memory efficiency
        for i in range(0, seq_len, self.block_size):
            block_end = min(i + self.block_size, seq_len)
            block_size = block_end - i
            
            # Get query block (zero-copy view)
            q_block = ZeroCopyTensorView.create_strided_view(
                q, i * self.num_heads * self.head_dim,
                (batch_size, self.num_heads, block_size, self.head_dim)
            )
            
            # Determine which KV blocks to attend to (sparse pattern)
            attend_blocks = self._get_sparse_attend_blocks(cache_pos + i, block_size)
            
            # Compute attention for selected blocks only
            for kv_block_idx in attend_blocks:
                kv_start = kv_block_idx * self.block_size
                kv_end = min(kv_start + self.block_size, cache_pos + seq_len)
                
                # Zero-copy views of KV cache
                k_block = ZeroCopyTensorView.create_strided_view(
                    self.cache_k, kv_start * self.num_heads * self.head_dim,
                    (1, self.num_heads, kv_end - kv_start, self.head_dim)
                )
                v_block = ZeroCopyTensorView.create_strided_view(
                    self.cache_v, kv_start * self.num_heads * self.head_dim,
                    (1, self.num_heads, kv_end - kv_start, self.head_dim)
                )
                
                # In-place attention computation
                scores = torch.matmul(q_block, k_block.transpose(-2, -1))
                scores.mul_(self.head_dim ** -0.5)  # In-place scaling
                
                # In-place softmax
                scores_max = scores.max(dim=-1, keepdim=True)[0]
                scores.sub_(scores_max)  # In-place subtract
                scores.exp_()  # In-place exp
                scores_sum = scores.sum(dim=-1, keepdim=True)
                scores.div_(scores_sum)  # In-place normalize
                
                # Accumulate to output (in-place)
                output[:, :, i:block_end].add_(
                    torch.matmul(scores, v_block.squeeze(0))
                )
        
        return output
    
    def _get_sparse_attend_blocks(self, position, block_size):
        """Determine which blocks to attend to based on sparse pattern"""
        current_block = position // self.block_size
        
        # Always attend to local window
        local_blocks = list(range(
            max(0, current_block - self.local_windows_len // self.block_size),
            current_block + 1
        ))
        
        # Add top-k important blocks (simplified - in practice use importance scores)
        total_blocks = (position + block_size) // self.block_size
        stride = max(1, total_blocks // self.topk)
        important_blocks = list(range(0, current_block, stride))
        
        # Combine and deduplicate
        attend_blocks = sorted(set(local_blocks + important_blocks))
        
        return attend_blocks

class CXLAwareKTransformersExperts(nn.Module):
    """CXL-aware MoE implementation with zero-copy expert loading"""
    
    def __init__(self, config, expert_modules, **kwargs):
        super().__init__()
        self.config = config
        self.num_experts = len(expert_modules)
        self.expert_modules = expert_modules
        self.allocator = CXLMemoryAllocator()
        
        # Pre-allocate expert weight storage in CXL memory tiers
        self._allocate_expert_storage()
        
        # Track which experts are in which memory tier
        self.expert_locations: Dict[int, MemoryTier] = {}
        self.expert_tensors: Dict[int, Dict[str, torch.Tensor]] = {}
        
        # Thread pool for parallel expert computation
        self.thread_pool = ThreadPoolExecutor(max_workers=kwargs.get('num_threads', 8))
    
    def _allocate_expert_storage(self):
        """Pre-allocate storage for experts across CXL tiers"""
        # Distribute experts across memory tiers based on importance
        # Most important experts in LOCAL_DRAM, others in CXL tiers
        
        for idx, expert in enumerate(self.expert_modules):
            if idx < 2:  # Keep top experts in fast memory
                tier = MemoryTier.LOCAL_DRAM
            elif idx < 8:
                tier = MemoryTier.CXL_NEAR
            else:
                tier = MemoryTier.CXL_FAR
            
            self.expert_locations[idx] = tier
            
            # Pre-allocate weight tensors
            self.expert_tensors[idx] = {}
            for name, param in expert.named_parameters():
                cxl_tensor = self.allocator.allocate_tensor_cxl(
                    param.shape, param.dtype, tier, zero_copy=True
                )
                # Copy weights to CXL memory (one-time operation)
                cxl_tensor.copy_(param.data, non_blocking=True)
                self.expert_tensors[idx][name] = cxl_tensor
    
    def forward(self, hidden_states, router_logits, **kwargs):
        """Forward pass with zero-copy expert computation"""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Compute expert assignments
        expert_mask = router_logits.argmax(dim=-1)
        
        # Pre-allocate output buffer
        output = torch.zeros_like(hidden_states)
        
        # Process tokens in parallel using thread pool
        def process_expert_tokens(expert_idx):
            # Find tokens assigned to this expert
            expert_mask_idx = (expert_mask == expert_idx).nonzero(as_tuple=True)
            
            if len(expert_mask_idx[0]) == 0:
                return
            
            # Get input tokens for this expert (zero-copy view)
            expert_input = hidden_states[expert_mask_idx]
            
            # Compute expert output using pre-allocated weights
            expert_output = self._compute_expert_inplace(
                expert_idx, expert_input
            )
            
            # Write back to output buffer (in-place)
            output[expert_mask_idx] = expert_output
        
        # Submit all expert computations to thread pool
        futures = []
        for expert_idx in range(self.num_experts):
            future = self.thread_pool.submit(process_expert_tokens, expert_idx)
            futures.append(future)
        
        # Wait for all experts to complete
        for future in futures:
            future.result()
        
        return output
    
    def _compute_expert_inplace(self, expert_idx, input_tokens):
        """Compute expert output with zero-copy operations"""
        # Get pre-allocated expert weights
        expert_weights = self.expert_tensors[expert_idx]
        
        # Simple feedforward computation (customize based on expert architecture)
        # This is a simplified example - adapt to your expert structure
        x = input_tokens
        
        # First linear layer (in-place operations where possible)
        w1 = expert_weights.get('w1', None)
        if w1 is not None:
            x = torch.matmul(x, w1.t())
            x = torch.nn.functional.silu(x, inplace=True)  # In-place activation
        
        # Second linear layer
        w2 = expert_weights.get('w2', None)
        if w2 is not None:
            x = torch.matmul(x, w2.t())
        
        return x
    
    def migrate_expert_tier(self, expert_idx: int, new_tier: MemoryTier):
        """Migrate expert between CXL memory tiers"""
        if self.expert_locations[expert_idx] == new_tier:
            return
        
        # Allocate in new tier
        old_tensors = self.expert_tensors[expert_idx]
        new_tensors = {}
        
        for name, old_tensor in old_tensors.items():
            new_tensor = self.allocator.allocate_tensor_cxl(
                old_tensor.shape, old_tensor.dtype, new_tier, zero_copy=True
            )
            # Copy data (this is the only copy operation)
            new_tensor.copy_(old_tensor, non_blocking=True)
            new_tensors[name] = new_tensor
        
        # Update tracking
        self.expert_tensors[expert_idx] = new_tensors
        self.expert_locations[expert_idx] = new_tier

class CXLSparseMemoryManager:
    """Memory manager optimized for CXL systems with zero-copy operations"""
    
    def __init__(self, model):
        self.model = model
        self.allocator = CXLMemoryAllocator()
        
        # Replace modules with CXL-aware versions
        self._replace_modules_with_cxl_aware()
        
        # Track memory usage per tier
        self.tier_usage = {tier: 0 for tier in MemoryTier}
    
    def _replace_modules_with_cxl_aware(self):
        """Replace standard modules with CXL-aware versions"""
        for name, module in self.model.named_modules():
            # Replace attention modules
            if 'self_attn' in name and hasattr(module, 'dynamic_attn'):
                parent_name = '.'.join(name.split('.')[:-1])
                parent = self.model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                
                # Create CXL-aware attention
                cxl_attn = CXLAwareKCustomAttention(module.config)
                setattr(parent, name.split('.')[-1], cxl_attn)
            
            # Replace MoE modules
            elif isinstance(module, KTransformersExperts):
                parent_name = '.'.join(name.split('.')[:-1])
                parent = self.model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                
                # Create CXL-aware experts
                cxl_experts = CXLAwareKTransformersExperts(
                    module.config, module.expert_modules
                )
                setattr(parent, name.split('.')[-1], cxl_experts)
    
    def optimize_memory_placement(self):
        """Optimize placement of model components across CXL tiers"""
        # Analyze access patterns and optimize placement
        # This is a simplified version - in practice, use profiling data
        
        # Move frequently accessed layers to faster memory
        for i in range(min(8, len(self.model.model.layers))):
            layer = self.model.model.layers[i]
            
            # Ensure attention KV cache is in fast memory for early layers
            if hasattr(layer.self_attn, 'cache_k'):
                # These are already allocated optimally in __init__
                pass
            
            # Move important experts to faster memory
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                experts_module = layer.mlp.experts
                if isinstance(experts_module, CXLAwareKTransformersExperts):
                    # Migrate top experts to faster memory
                    for expert_idx in range(min(2, experts_module.num_experts)):
                        experts_module.migrate_expert_tier(
                            expert_idx, MemoryTier.LOCAL_DRAM
                        )
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics per tier"""
        stats = {}
        
        for tier in MemoryTier:
            used = 0
            total = 0
            
            for region in self.allocator.memory_regions:
                if region.tier == tier:
                    total += region.size
                    used += region.size - region.available
            
            stats[f"{tier.name}_used_gb"] = used / (1024**3)
            stats[f"{tier.name}_total_gb"] = total / (1024**3)
            stats[f"{tier.name}_utilization"] = (used / total * 100) if total > 0 else 0
        
        return stats

# Integration function
def enable_cxl_optimization(model):
    """Enable CXL optimization for a model"""
    manager = CXLSparseMemoryManager(model)
    manager.optimize_memory_placement()
    return manager