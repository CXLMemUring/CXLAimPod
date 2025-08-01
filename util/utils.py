#!/usr/bin/env python
# coding=utf-8
'''
Description  :  
Author       : Boxin Zhang, Azure-Tang
Version      : 0.1.0
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''
import torch
from torch import nn
import itertools
import time
import enum
from transformers import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    MinPLogitsWarper,
    TypicalLogitsWarper,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
)

from ktransformers.util.weight_loader import ModelLoaderFactory, ModelLoader, SafeTensorLoader, GGUFLoader
from ktransformers.util.custom_gguf import translate_name_to_gguf
from ktransformers.operators import base_operator
from ktransformers.models.custom_cache import StaticCache
from ktransformers.util.cuda_graph_runner import CUDAGraphRunner
from ktransformers.util.textstream import TextStreamer
if not torch.xpu.is_available():
    from ktransformers.operators.flashinfer_wrapper import MLAWrapperSingleton
import socket

warm_uped = False

def get_free_ports(n: int, continue_prot: list):
    sockets = []
    ports = []
    for _ in range(n):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0)) 
        port = s.getsockname()[1]
        if port in continue_prot:
            s.close()
            continue
        ports.append(port)
        sockets.append(s)
    for s in sockets:
        s.close()
    return ports

def get_compute_capability(device:torch.device = None):
    if torch.cuda.is_available():
        if device is None:
            num_gpus = torch.cuda.device_count()
            min_compute_capability_major = 100
            for gpu_id in range(num_gpus):
                gpu_props = torch.cuda.get_device_properties(gpu_id)
                min_compute_capability_major = min(min_compute_capability_major, gpu_props.major)
            return min_compute_capability_major
        else:
            return torch.cuda.get_device_properties(device)
    else:
        return 0

def set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        if hasattr(cur_mod, s):
            cur_mod = getattr(cur_mod, s)
        else: # nn.ModuleList or nn.ModuleList
            cur_mod=cur_mod[int(s)]
    if hasattr(cur_mod, tokens[-1]):
        setattr(cur_mod, tokens[-1], module)
    else: # nn.ModuleList or nn.ModuleList
        cur_mod[int(tokens[-1])] = module

def set_param(module: nn.Module, name: str, weights: torch.Tensor):
    
    param=nn.parameter.Parameter(weights, requires_grad=False)
    if isinstance(module, nn.Linear) and len(weights.shape)==1:
        param.unsqueeze_(0)
    setattr(module, name, param)

def get_device(gguf_module_key:str, device_map:dict):
    if gguf_module_key in device_map:
        return device_map[gguf_module_key]["generate_device"]
    else:
        return "cuda"

def get_all_used_cuda_device(device_map:dict):
    all_device_list = set()
    for key in device_map:
        all_device_list.add(device_map[key]["generate_device"]) if "generate_device" in device_map[key] else None
        all_device_list.add(device_map[key]["prefill_device"]) if "prefill_device" in device_map[key] else None
    if "cpu" in all_device_list:
        all_device_list.remove("cpu")
    all_device_list = list(all_device_list)
    return all_device_list

def load_cur_state_dict(module: nn.Module, gguf_loader: ModelLoader, prefix: str = "", device="cuda"):
    prefix = prefix.replace("orig_module.", "")
    persistent_buffers = {k: v for k, v in module._buffers.items() if k not in module._non_persistent_buffers_set}
    local_name_params = itertools.chain(module._parameters.items(), persistent_buffers.items())
    local_state = {k: v for k, v in local_name_params if v is not None}
    for name, param in local_state.items():
        key = prefix + name
        translated_key = key
        
        # TODO: Merge all loader.
        # I know this is ugly but lets do it for now.
        if isinstance(gguf_loader, SafeTensorLoader):
            load_dequantized_tensor = gguf_loader.load_dequantized_tensor
            translated_key = key
        else:
            load_dequantized_tensor = gguf_loader.load_gguf_tensor
            translated_key = translate_name_to_gguf(key)
            tensor_file_map = gguf_loader.tensor_file_map
        
        if hasattr(gguf_loader, 'has_tensor') and gguf_loader.has_tensor(translated_key):
            target_dtype = torch.get_default_dtype()
            device = "cpu"  # Force CPU loading
            print(f"loading {translated_key} to {device}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            weights = load_dequantized_tensor(translated_key, device=device).to(dtype=target_dtype)
            set_param(module, name, weights)
            del weights
        else:
            # Handle missing keys with graceful fallback options
            print(f"Warning: Couldn't find {translated_key} in GGUF file, trying alternatives...")
            
            # Try to find similar keys based on common patterns
            alternative_found = False
            
            # Option 1: Try without prefix
            base_key = key.split('.')[-1]
            if hasattr(gguf_loader, 'has_tensor') and gguf_loader.has_tensor(base_key):
                print(f"Found alternative key: {base_key}")
                weights = load_dequantized_tensor(base_key, device="cpu").to(dtype=torch.get_default_dtype())
                set_param(module, name, weights)
                alternative_found = True
                del weights
            
            # Option 2: Try with various transformations
            if not alternative_found:
                possible_alternatives = []
                
                # Try to find keys with similar patterns
                key_parts = translated_key.split('.')
                if len(key_parts) > 1:
                    # Try matching only the last part of the key
                    for existing_key in (gguf_loader.tensor_info.keys() if hasattr(gguf_loader, 'tensor_info') else []):
                        if existing_key.endswith('.' + key_parts[-1]) or existing_key.endswith('/' + key_parts[-1]):
                            possible_alternatives.append(existing_key)
                            
                    # Try matching keys that have the same shape
                    target_shape = param.shape
                    for existing_key in (gguf_loader.tensor_info.keys() if hasattr(gguf_loader, 'tensor_info') else []):
                        try:
                            if hasattr(tensor_file_map[existing_key], 'shape'):
                                if tensor_file_map[existing_key].shape == target_shape:
                                    possible_alternatives.append(existing_key)
                        except Exception:
                            pass  # Skip keys that don't have proper shape info
                
                if possible_alternatives:
                    # Find the most similar key by string matching
                    import difflib
                    closest_match = difflib.get_close_matches(translated_key, possible_alternatives, n=1, cutoff=0.3)
                    
                    if closest_match:
                        alternative_key = closest_match[0]
                        print(f"Found close match: {alternative_key}")
                        weights = load_dequantized_tensor(alternative_key, device="cpu").to(dtype=torch.get_default_dtype())
                        set_param(module, name, weights)
                        alternative_found = True
                        del weights
            
            # Option 3: Initialize with zeros or random values that match parameter shape
            if not alternative_found:
                print(f"No alternative found for {translated_key}, initializing with zeros")
                # Initialize with zeros of the correct shape
                zero_init = torch.zeros_like(param, device="cpu")
                set_param(module, name, zero_init)
                del zero_init

def load_weights(module:nn.Module, gguf_loader:GGUFLoader, prefix=''):
    #print(f"recursively loading weights {prefix}")
    if not isinstance(module, base_operator.BaseInjectedModule):
        load_cur_state_dict(module, gguf_loader, prefix)
        for name, child in module._modules.items():
            load_weights(child, gguf_loader, prefix+name+".")
    else:
        module.load()

def tf_logits_warper(generation_config):
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsWarper`] instances
        used for multinomial sampling.
        """

        # instantiate warpers list
        warpers = LogitsProcessorList()

        # In beam methods, we need to keep at least one non-eos token to explore continuations that might have a
        # better score (i.e. keep len(list(generation_config._eos_token_tensor)) + 1)
        if generation_config.num_beams > 1:
            if isinstance(generation_config._eos_token_tensor, list):
                min_tokens_to_keep = len(generation_config._eos_token_tensor) + 1
            elif isinstance(generation_config._eos_token_tensor, torch.Tensor):
                min_tokens_to_keep = generation_config._eos_token_tensor.shape[0] + 1
            else:
                min_tokens_to_keep = 2
        else:
            min_tokens_to_keep = 1

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        if generation_config.temperature is not None and generation_config.temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(generation_config.temperature))
        if generation_config.top_k is not None and generation_config.top_k != 0:
            warpers.append(TopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=min_tokens_to_keep))
        if generation_config.top_p is not None and generation_config.top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=min_tokens_to_keep))
        if generation_config.min_p is not None:
            # Applied after temperature scaling (see https://github.com/ggerganov/llama.cpp/pull/3841#issuecomment-2073826084)
            warpers.append(MinPLogitsWarper(min_p=generation_config.min_p, min_tokens_to_keep=min_tokens_to_keep))
        if generation_config.typical_p is not None and generation_config.typical_p < 1.0:
            warpers.append(
                TypicalLogitsWarper(mass=generation_config.typical_p, min_tokens_to_keep=min_tokens_to_keep)
            )
        if generation_config.epsilon_cutoff is not None and 0.0 < generation_config.epsilon_cutoff < 1.0:
            warpers.append(
                EpsilonLogitsWarper(epsilon=generation_config.epsilon_cutoff, min_tokens_to_keep=min_tokens_to_keep)
            )
        if generation_config.eta_cutoff is not None and 0.0 < generation_config.eta_cutoff < 1.0:
            warpers.append(
               EtaLogitsWarper(
                    epsilon=generation_config.eta_cutoff, min_tokens_to_keep=min_tokens_to_keep, device=device
                )
            )
        # `LogitNormalization` should always be the last logit processor, when present
        if generation_config.renormalize_logits is True:
            warpers.append(LogitNormalization())
        return warpers

def prefill_and_generate(model, tokenizer, inputs, max_new_tokens=10000, use_cuda_graph: bool = True,
                         mode = 'normal', force_think: bool = False, chunk_size = 16384, use_flashinfer_mla = False,
                         num_heads = None, head_dim_ckv = None, head_dim_kpe = None, q_head_dim = None):
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch._dynamo.config.suppress_errors = True
    batch_size, seq_length = inputs.shape
    device_map = model.gguf_loader.tensor_device_map
    torch_device = get_device('model.layers.0.self_attn', device_map)
    torch_device = torch_device_mapping[torch_device] if torch_device in torch_device_mapping else torch_device
    inputs = inputs.to(torch_device)
    all_cuda_device = get_all_used_cuda_device(device_map)

    tokens = []
    
    def decode_one_tokens(cuda_graph_runner, cur_token, position_ids, cache_position, past_key_values, logits_warper, generation_config, use_cuda_graph: bool = True):
        if cuda_graph_runner is None:
            use_cuda_graph = False
        if use_cuda_graph:
            logits = cuda_graph_runner(cur_token, position_ids, cache_position)
        else:
            # custom_stream = torch.cuda.Stream()
            if torch.cuda.is_available():
                torch.cuda.set_device(torch_device)
            elif torch.xpu.is_available():
                torch.xpu.set_device(torch_device)
            else:
                raise RuntimeError(f"The device: {torch_device} is not available")
            inputs_embeds = model.model.embed_tokens(cur_token.to("cpu")).to(torch_device)
            # with torch.cuda.stream(custom_stream):
            logits=model(inputs_embeds=inputs_embeds,
                        position_ids=position_ids,
                        cache_position=cache_position,
                        past_key_values=past_key_values,
                        return_dict=False, use_cache=True)[0]
        if past_key_values != None and isinstance(past_key_values, StaticCache):
            past_key_values.change_seq_length(1)
        sync_all_device(all_cuda_device)
        #print(logits)
        next_token_scores = logits_warper(inputs, logits[:, -1, :])
        if generation_config.do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_token = torch.argmax(next_token_scores, dim=-1)
        return next_token
    
    # TODO: use CUDA Graph for chunk prefill, may get small improvement
    def chunk_prefill(inputs, cache_position, past_key_values):
        # Add print statements to debug
        print(f"inputs shape: {inputs.shape}")
        print(f"cache_position shape: {cache_position.shape}")
        if mode == "long_context":
            inputs_embeds = model.model.embed_tokens(inputs.to("cpu"))
        else:
            inputs_embeds = model.model.embed_tokens(inputs.to("cpu")).to(torch_device)
        if use_flashinfer_mla:
            MLAWrapperSingleton.update_buffer(past_key_values.max_pages)
            MLAWrapperSingleton.need_plan_all()
            
        # Call the model
        output = model(
            inputs_embeds = inputs_embeds, cache_position=cache_position, past_key_values=past_key_values, return_dict=False, use_cache=True
        )
        
        # Check if output is a tuple (common with return_dict=False)
        if isinstance(output, tuple):
            output = output[0]
        
        # Now handle the tensor based on its dimensions
        print(f"model output shape: {output.shape}")
        
        if len(output.shape) == 3:
            # Original case: [batch, seq, vocab]
            logits = output[:,-1,:].unsqueeze(0).clone().to(torch_device)
        elif len(output.shape) == 2:
            # Handle 2D case: [batch, vocab] - already the last token logits
            logits = output.unsqueeze(0).clone().to(torch_device)
        else:
            # Unexpected shape - print more info and try to recover
            print(f"WARNING: Unexpected model output shape: {output.shape}")
            # Try to make it compatible by reshaping if possible
            if output.numel() > 0:
                # Reshape to have batch dimension and vocabulary dimension
                if output.shape[0] == 1:
                    # Single batch case
                    logits = output.view(1, -1).clone().to(torch_device)
                else:
                    # Try to infer vocab size from model config
                    vocab_size = getattr(model.config, "vocab_size", output.shape[-1])
                    logits = output.view(1, vocab_size).clone().to(torch_device)
            else:
                # Empty tensor - fill with zeros as fallback
                print("WARNING: Empty model output, using zeros")
                vocab_size = getattr(model.config, "vocab_size", 32000)  # Default fallback
                logits = torch.zeros((1, vocab_size), device=torch_device)
        
        return logits
    
    if torch.cuda.is_available():
        torch.cuda.set_device(torch_device)
    elif torch.xpu.is_available():
        torch.xpu.set_device(torch_device)
    else:
        raise RuntimeError(f"The device: {torch_device} is not available")
    with torch.no_grad():
        
        stream = TextStreamer(tokenizer)
        if torch.xpu.is_available():
            from ipex_llm.transformers.kv import DynamicUnbalancedFp8Cache, DynamicNormalCache
            if model.config.architectures[0] in ["DeepseekV3ForCausalLM", "DeepseekV2ForCausalLM"]:
                past_key_values = DynamicUnbalancedFp8Cache.from_legacy_cache(None)
            else:
                past_key_values = DynamicNormalCache.from_legacy_cache(None)
        elif mode != 'long_context':
            past_key_values = StaticCache(
                config = model.config, max_batch_size = 1, max_cache_len = seq_length + max_new_tokens, device = device_map, dtype = model.dtype
            )
        else:
            past_key_values = None
        
        generation_config, model_kwargs = model._prepare_generation_config(
            None, do_sample=True
            # change this to modify generate config
            #top_k=5, top_p=0.85, temperature=0.1
        )

        logits_warper = tf_logits_warper(generation_config)

        cache_position = torch.arange(seq_length, device=torch_device, dtype=torch.int32)
        generated_ids = torch.zeros(
            batch_size, seq_length + max_new_tokens + 1, dtype=torch.int, device=torch_device
        )
        generated_ids[:, cache_position] = inputs.to(torch_device).to(torch.int)
        start_time = time.time()

        chunk_start = 0
        while chunk_start < seq_length:
            chunk_end = min(chunk_start + chunk_size, seq_length)
            if past_key_values != None:
                past_key_values.cur_idx=cache_position[chunk_start:chunk_end]
            logits = chunk_prefill(inputs[:, chunk_start:chunk_end], cache_position[chunk_start:chunk_end], past_key_values)
            chunk_start += chunk_size

        next_token_scores = logits_warper(inputs, logits[:, -1, :])
        if generation_config.do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_token = torch.argmax(next_token_scores, dim=-1)

        first_token_time = time.time() - start_time
        
        if use_flashinfer_mla:
            MLAWrapperSingleton.reset_buffer()

        prefill_count = seq_length
        prefill_time = first_token_time
        if force_think:
            print("<think>")
        print(stream.put(next_token.item()), end="", flush=True)
        generated_ids[:, seq_length] = next_token
        tokens.append(int(next_token))
        inputs = torch.cat((inputs, next_token.unsqueeze(0)), dim=-1)
        cache_position = torch.tensor([seq_length], device=torch_device, dtype=torch.int32)
        position_ids = cache_position.unsqueeze(0)
        seq_length += 1
        
        cuda_graph_runner = None
            
        start_time = time.time()
        for i in range(1, max_new_tokens):
            if use_flashinfer_mla:
                MLAWrapperSingleton.plan_all(None,None,None,position_ids.squeeze(1)+1,None,
                                             num_heads, head_dim_ckv, head_dim_kpe, past_key_values.page_size,
                                             model.model.layers[0].self_attn.softmax_scale, torch.bfloat16, torch.bfloat16)
            global warm_uped
            if use_cuda_graph and ( (warm_uped == True and int(i) == 1) or (warm_uped == False and int(i) == 2) ):
                warm_uped = True
                cuda_graph_runner = CUDAGraphRunner()
                cuda_graph_runner.capture(model, next_token.unsqueeze(0), position_ids, cache_position, past_key_values, torch_device, return_dict=False, use_cache=True)
            next_token = decode_one_tokens(cuda_graph_runner, next_token.unsqueeze(0), position_ids, cache_position, past_key_values, logits_warper, generation_config, use_cuda_graph).to(torch_device)
            inputs = torch.cat((inputs, next_token.unsqueeze(0)), dim=-1)
            generated_ids[:, cache_position] = next_token.int()
            tokens.append(int(next_token))
            seq_length += 1
            
            if next_token[0].item() == tokenizer.eos_token_id or tokenizer.decode(next_token.tolist()) == '<|im_end|>':
                print(stream.end(), end="", flush=True)
                break
            else:
                print(stream.put(next_token.item()), end="", flush=True)
            cache_position += 1
            position_ids = cache_position.unsqueeze(0)
        

    total_time = time.time() - start_time
    tokens_generated = len(tokens)
    tokens_per_second = tokens_generated / total_time

    print("")

    print(f"prompt eval count:    {prefill_count} token(s)")
    print(f"prompt eval duration: {prefill_time}s")
    print(f"prompt eval rate:     {prefill_count/prefill_time} tokens/s")
    print(f"eval count:           {tokens_generated} token(s)")
    print(f"eval duration:        {total_time}s")
    print(f"eval rate:            {tokens_per_second} tokens/s")

    return tokens

class InferenceState(enum.Enum):
    UNLOAD = 0
    PREFILL = 1
    GENERATE = 2
    RESTORE = 3
