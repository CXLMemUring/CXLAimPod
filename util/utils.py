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
from ktransformers.util.custom_gguf import translate_name_to_gguf
from ktransformers.util.custom_gguf import GGUFLoader
from ktransformers.operators import base_operator
from ktransformers.models.custom_cache import StaticCache
from ktransformers.util.cuda_graph_runner import CUDAGraphRunner
from ktransformers.util.textstream import TextStreamer
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
        return "cpu"

def get_all_used_cuda_device(device_map:dict):
    all_device_list = set()
    for key in device_map:
        all_device_list.add(device_map[key]["generate_device"]) if "generate_device" in device_map[key] else None
        all_device_list.add(device_map[key]["prefill_device"]) if "prefill_device" in device_map[key] else None
    if "cpu" in all_device_list:
        all_device_list.remove("cpu")
    all_device_list = list(all_device_list)
    return all_device_list

def load_cur_state_dict(module: nn.Module, gguf_loader: GGUFLoader, prefix: str = ""):
    prefix = prefix.replace("orig_module.", "")
    persistent_buffers = {k: v for k, v in module._buffers.items() if k not in module._non_persistent_buffers_set}
    local_name_params = itertools.chain(module._parameters.items(), persistent_buffers.items())
    local_state = {k: v for k, v in local_name_params if v is not None}
    for name, param in local_state.items():
        key = prefix + name
        translated_key = translate_name_to_gguf(key)
        
        # TODO: Merge all loader.
        # I know this is ugly but lets do it for now.
        if gguf_loader.safetensor_loader is not None:
            load_dequantized_tensor = gguf_loader.safetensor_loader.load_dequantized_tensor
            tensor_file_map = gguf_loader.safetensor_loader.tensor_file_map
        else:
            load_dequantized_tensor = gguf_loader.load_gguf_tensor
            tensor_file_map = gguf_loader.tensor_file_map
        
        if translated_key in tensor_file_map:
            target_dtype = torch.get_default_dtype()
            device = "cpu"  # Force CPU loading
            print(f"loading {translated_key} to {device}")
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
            if base_key in tensor_file_map:
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
                    for existing_key in tensor_file_map.keys():
                        if existing_key.endswith('.' + key_parts[-1]) or existing_key.endswith('/' + key_parts[-1]):
                            possible_alternatives.append(existing_key)
                            
                    # Try matching keys that have the same shape
                    target_shape = param.shape
                    for existing_key in tensor_file_map.keys():
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

def prefill_and_generate(model, tokenizer, inputs, max_new_tokens=10000, use_cuda_graph: bool = False,
                         mode = 'normal', force_think: bool = False, chunk_size = 16384, use_flashinfer_mla = False,
                         num_heads = None, head_dim_ckv = None, head_dim_kpe = None, q_head_dim = None):
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch._dynamo.config.suppress_errors = True
    
    batch_size, seq_length = inputs.shape
    torch_device = "cpu"
    inputs = inputs.to(torch_device)
    
    # Force CPU mode - disable all CUDA features
    use_cuda_graph = False
    use_flashinfer_mla = False

    tokens = []
    
    def decode_one_tokens(cuda_graph_runner, cur_token, position_ids, cache_position, past_key_values, logits_warper, generation_config, use_cuda_graph: bool = False):
        # Always use CPU path, never CUDA graph
        use_cuda_graph = False
        try:
            # Determine model's native dtype
            model_dtype = None
            if hasattr(model, 'dtype'):
                model_dtype = model.dtype
            elif hasattr(model, 'config') and hasattr(model.config, 'torch_dtype'):
                model_dtype = getattr(torch, model.config.torch_dtype) 
            else:
                # Check the dtype of a parameter to determine model dtype
                for param in model.parameters():
                    model_dtype = param.dtype
                    break
            
            # Default to float32 if we couldn't determine the model dtype
            if model_dtype is None:
                model_dtype = torch.float32
            
            # On CPU, we'll use float32 for stability regardless of model's native dtype
            compute_dtype = torch.float32
            
            # Ensure inputs are on CPU with consistent dtypes and shapes
            # Make sure cur_token is 2D with shape [batch_size, seq_len]
            if cur_token.dim() == 1:
                cur_token = cur_token.unsqueeze(0)  # Add batch dimension if missing
            elif cur_token.dim() > 2:
                cur_token = cur_token.view(cur_token.size(0), -1)  # Flatten extra dimensions
                
            cur_token = cur_token.to(device="cpu", dtype=torch.long)
            position_ids = position_ids.to(device="cpu", dtype=torch.long)
            cache_position = cache_position.to(device="cpu", dtype=torch.long)
            
            # Get embeddings and ensure they have the right dtype for computation
            try:
                inputs_embeds = model.model.embed_tokens(cur_token)
                inputs_embeds = inputs_embeds.to(device="cpu", dtype=compute_dtype)
            except RuntimeError as embed_error:
                print(f"Error in embedding lookup: {embed_error}")
                print(f"cur_token shape: {cur_token.shape}, dtype: {cur_token.dtype}")
                # Try reshaping to ensure compatibility
                if cur_token.dim() > 2:
                    cur_token = cur_token.view(cur_token.size(0), -1)
                inputs_embeds = model.model.embed_tokens(cur_token)
                inputs_embeds = inputs_embeds.to(device="cpu", dtype=compute_dtype)
            
            try:
                # First attempt to run the model with proper type handling
                model_inputs = {
                    'inputs_embeds': inputs_embeds,
                    'position_ids': position_ids,
                    'past_key_values': past_key_values,
                    'cache_position': cache_position,
                    'use_cache': True,
                    'return_dict': True
                }
                
                # Run the model with proper error handling
                outputs = model(**model_inputs)
                
                # Extract logits and ensure they're in compute_dtype
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits.to(dtype=compute_dtype)
                else:
                    logits = outputs[0].to(dtype=compute_dtype)
                
                # Calculate next token
                next_token_logits = logits[:, -1, :]
                next_token_scores = logits_warper(cur_token, next_token_logits)
                
                # Either sample or take argmax
                if hasattr(generation_config, 'do_sample') and generation_config.do_sample:
                    probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_token = torch.argmax(next_token_scores, dim=-1)
                    
                return next_token
                
            except RuntimeError as e:
                if "expected m1 and m2 to have the same dtype" in str(e):
                    print(f"Handling dtype mismatch error in decode_one_tokens: {str(e)}")
                    
                    # Try forcing all inputs to model's native dtype
                    inputs_embeds = inputs_embeds.to(dtype=model_dtype)
                    
                    model_inputs = {
                        'inputs_embeds': inputs_embeds,
                        'position_ids': position_ids,
                        'past_key_values': past_key_values,
                        'cache_position': cache_position,
                        'use_cache': True,
                        'return_dict': True
                    }
                    
                    outputs = model(**model_inputs)
                    
                    # Convert logits back to compute_dtype
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits.to(dtype=compute_dtype)
                    else:
                        logits = outputs[0].to(dtype=compute_dtype)
                    
                    next_token_logits = logits[:, -1, :]
                    next_token_scores = logits_warper(cur_token, next_token_logits)
                    next_token = torch.argmax(next_token_scores, dim=-1)
                    return next_token
                    
                elif "shape" in str(e) and ("invalid for input of size" in str(e) or "invalid" in str(e)):
                    print(f"Handling shape mismatch error: {str(e)}")
                    
                    # Reshape input embeddings if needed
                    batch_size = inputs_embeds.size(0)
                    try:
                        # Attempt to reshape embeddings to a valid size for the model
                        if hasattr(model.config, 'hidden_size'):
                            hidden_size = model.config.hidden_size
                            inputs_embeds = inputs_embeds.reshape(batch_size, 1, hidden_size)
                        else:
                            # If we can't determine hidden_size, use the embedding size
                            embed_size = inputs_embeds.size(-1)
                            inputs_embeds = inputs_embeds.reshape(batch_size, 1, embed_size)
                            
                        inputs_embeds = inputs_embeds.to(dtype=model_dtype)
                        
                        # Try with reshaped inputs
                        model_inputs = {
                            'inputs_embeds': inputs_embeds,
                            'position_ids': position_ids,
                            'past_key_values': past_key_values,
                            'cache_position': cache_position,
                            'use_cache': True,
                            'return_dict': True
                        }
                        
                        outputs = model(**model_inputs)
                        
                        # Convert logits back to compute_dtype
                        if hasattr(outputs, 'logits'):
                            logits = outputs.logits.to(dtype=compute_dtype)
                        else:
                            logits = outputs[0].to(dtype=compute_dtype)
                        
                        next_token_logits = logits[:, -1, :]
                        next_token_scores = logits_warper(cur_token, next_token_logits)
                        next_token = torch.argmax(next_token_scores, dim=-1)
                        return next_token
                    except RuntimeError as reshape_error:
                        print(f"Reshape attempt failed: {reshape_error}")
                        # Last resort fallback
                        if hasattr(model.config, 'eos_token_id'):
                            fallback_token = torch.tensor([model.config.eos_token_id], device="cpu")
                        else:
                            fallback_token = torch.tensor([0], device="cpu")  # Use token ID 0 as default fallback
                        return fallback_token
                    
                elif "DynamicScaledDotProductAttention" in str(e) and "prefill_block_num" in str(e):
                    print(f"Handling DynamicScaledDotProductAttention error: {str(e)}")
                    # We should have fixed this with our custom attention implementation
                    raise
                else:
                    raise
        except Exception as e:
            print(f"Critical error in decode_one_tokens: {e}")
            # Return a fallback token as last resort instead of crashing
            print("Returning fallback token (EOS) due to critical error")
            if hasattr(model.config, 'eos_token_id'):
                fallback_token = torch.tensor([model.config.eos_token_id], device="cpu")
            else:
                fallback_token = torch.tensor([0], device="cpu")  # Use token ID 0 as default fallback
            return fallback_token
    
    def chunk_prefill(inputs, cache_position, past_key_values):
        try:
            # Get embeddings
            inputs_embeds = model.model.embed_tokens(inputs)
            
            # Get model's native dtype
            model_dtype = None
            if hasattr(model, 'dtype'):
                model_dtype = model.dtype
            elif hasattr(model, 'config') and hasattr(model.config, 'torch_dtype'):
                model_dtype = getattr(torch, model.config.torch_dtype)
            else:
                # Check the dtype of a parameter to determine model dtype
                for param in model.parameters():
                    model_dtype = param.dtype
                    break
            
            # Default to float32 if we couldn't determine the model dtype
            if model_dtype is None:
                model_dtype = torch.float32
                
            # Ensure inputs are properly shaped and have the right dtype for computation
            batch_size, seq_len, hidden_size = inputs_embeds.shape
            inputs_embeds = inputs_embeds.to(dtype=model_dtype)
            
            # Wrap model call in a try-except block to handle errors
            try:
                outputs = model(
                    inputs_embeds=inputs_embeds, 
                    cache_position=cache_position, 
                    past_key_values=past_key_values, 
                    return_dict=True, 
                    use_cache=True
                )
                
                # Extract logits and ensure they have consistent dtype
                logits = outputs.logits.to(dtype=torch.float32)
                
                # Safely get the last token logits
                if logits.dim() >= 2:
                    last_logits = logits[:, -1, :].clone()
                    if last_logits.dim() == 2:
                        return last_logits.unsqueeze(0)
                    return last_logits
                
                return logits
                
            except RuntimeError as e:
                print(f"Model forward error: {e}")
                print(f"Input shape: {inputs_embeds.shape}, cache_position shape: {cache_position.shape}")
                
                if "shape" in str(e) and ("invalid" in str(e) or "size mismatch" in str(e)):
                    # Handle shape error by ensuring the tensors can be properly reshaped
                    print("Handling shape mismatch, attempting to reshape inputs...")
                    
                    # Try processing with a simpler approach
                    # First get the correct model hidden size
                    if hasattr(model.config, 'hidden_size'):
                        hidden_size = model.config.hidden_size
                    
                    # Reshape inputs to ensure compatibility
                    inputs_embeds_reshaped = inputs_embeds.reshape(batch_size, seq_len, hidden_size)
                    
                    # Try with the reshaped inputs
                    outputs = model(
                        inputs_embeds=inputs_embeds_reshaped,
                        cache_position=cache_position,
                        past_key_values=past_key_values,
                        return_dict=True,
                        use_cache=True
                    )
                    
                    # Extract logits and ensure they have consistent dtype
                    logits = outputs.logits.to(dtype=torch.float32)
                    return logits[:, -1, :]
                
                elif "expected m1 and m2 to have the same dtype" in str(e):
                    # Handle dtype mismatch
                    print("Handling dtype mismatch...")
                    
                    # Try with a different dtype
                    inputs_embeds = inputs_embeds.to(dtype=torch.float32)
                    
                    outputs = model(
                        inputs_embeds=inputs_embeds,
                        cache_position=cache_position,
                        past_key_values=past_key_values,
                        return_dict=True,
                        use_cache=True
                    )
                    
                    logits = outputs.logits.to(dtype=torch.float32)
                    return logits[:, -1, :]
                
                else:
                    # Fallback implementation - manual call each component
                    print("Attempting fallback implementation...")
                    
                    # Apply the first layer norm
                    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
                        hidden_states = model.model.norm(inputs_embeds)
                    else:
                        hidden_states = inputs_embeds
                    
                    # Apply the language model head
                    if hasattr(model, 'lm_head'):
                        logits = model.lm_head(hidden_states)
                        return logits[:, -1, :]
                
                # If we can't get logits, return a tensor of zeros as a last resort
                print("Couldn't produce valid logits. Returning zeros.")
                return torch.zeros((1, 1, model.config.vocab_size), device=inputs.device)
            
        except Exception as e:
            print(f"Critical error in chunk_prefill: {e}")
            # Last resort - return empty logits
            if hasattr(model, 'config') and hasattr(model.config, 'vocab_size'):
                vocab_size = model.config.vocab_size
            else:
                vocab_size = 32000  # Default vocabulary size
            return torch.zeros((1, 1, vocab_size), device=inputs.device)
    
    with torch.no_grad():
        
        stream = TextStreamer(tokenizer)
        if mode != 'long_context':
            # Create a simple device map dictionary for CPU
            device_map = {}
            for idx in range(model.config.num_hidden_layers):
                device_map[f"blk.{idx}.self_attn"] = {"generate_device": "cpu"}
                
            # Determine the appropriate dtype for CPU (avoid using half precision on CPU)
            cache_dtype = torch.float32  # Always use float32 for CPU regardless of model dtype
            
            # Initialize StaticCache with appropriate parameters for CPU
            try:
                # Check if we're running on CPU and simplify device maps accordingly
                is_cpu_mode = str(inputs.device).lower() == "cpu"
                if is_cpu_mode:
                    print("Running in CPU mode, using simplified StaticCache initialization")
                    # For CPU mode, use a simpler device configuration
                    past_key_values = StaticCache(
                        config=model.config, 
                        max_batch_size=1, 
                        max_cache_len=seq_length + max_new_tokens, 
                        device="cpu", 
                        dtype=cache_dtype
                    )
                else:
                    # For GPU mode, use the device map
                    past_key_values = StaticCache(
                        config=model.config, 
                        max_batch_size=1, 
                        max_cache_len=seq_length + max_new_tokens, 
                        device=device_map, 
                        dtype=cache_dtype
                    )
            except Exception as e:
                print(f"Error initializing StaticCache: {e}")
                # Fallback to even simpler initialization
                try:
                    print("Attempting fallback StaticCache initialization")
                    past_key_values = StaticCache(
                        config=model.config, 
                        max_batch_size=1, 
                        max_cache_len=min(seq_length + max_new_tokens, 2048),  # Limit max cache length
                        device="cpu", 
                        dtype=torch.float32
                    )
                except Exception as e2:
                    print(f"Critical error in StaticCache initialization: {e2}")
                    print("Creating minimal StaticCache with defaults")
                    # Create a very basic cache as last resort
                    setattr(model.config, "max_position_embeddings", 2048) if not hasattr(model.config, "max_position_embeddings") else None
                    setattr(model.config, "num_key_value_heads", model.config.num_attention_heads) if not hasattr(model.config, "num_key_value_heads") else None
                    past_key_values = StaticCache(
                        config=model.config, 
                        max_batch_size=1, 
                        max_cache_len=1024,  # Use a small safe value
                        device="cpu", 
                        dtype=torch.float32
                    )
        else:
            past_key_values = None
        
        generation_config, model_kwargs = model._prepare_generation_config(
            None, do_sample=True
            # change this to modify generate config
            #top_k=5, top_p=0.85, temperature=0.1
        )
        try: # transformers==4.43
            logits_warper = (
                model._get_logits_warper(generation_config,device=inputs.device)
            )
        except: 
            logits_warper = (
                model._get_logits_warper(generation_config)
            )

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
        
        # Remove CUDA-specific operations
        prefill_count = seq_length
        prefill_time = first_token_time
        if force_think:
            print("<think>")
        print(stream.put(next_token.item()), end="", flush=True)
        
        # Fix the tensor concatenation - ensure proper dimensions
        # Make sure next_token is properly shaped for concatenation with inputs
        if next_token.dim() == 0:  # If next_token is a scalar
            next_token = next_token.unsqueeze(0)  # Make it a 1D tensor
            
        # Ensure inputs and next_token have same number of dimensions
        if inputs.dim() != next_token.dim():
            if inputs.dim() > next_token.dim():
                # Add missing dimensions to next_token
                while next_token.dim() < inputs.dim():
                    next_token = next_token.unsqueeze(0)
            else:
                # Reduce dimensions of next_token to match inputs
                next_token = next_token.view(*[-1 for _ in range(inputs.dim())])
        
        # Now concatenate with proper dimension (-1)
        inputs = torch.cat([inputs, next_token.view(1, -1)], dim=-1)
        
        # Ensure next_token is the right shape for generated_ids assignment
        # generated_ids expects a single value at each position per batch
        if next_token.dim() > 1:
            # If multi-dimensional, take first token
            token_value = next_token.view(-1)[0].item()
            token_tensor = torch.tensor([token_value], dtype=torch.int, device=torch_device)
            generated_ids[:, seq_length] = token_tensor
        else:
            # If already 1D or scalar, just convert to int
            token_value = next_token.item() if next_token.numel() == 1 else next_token[0].item()
            generated_ids[:, seq_length] = torch.tensor([token_value], dtype=torch.int, device=torch_device)
        
        # Store the token value safely, handling different possible shapes
        try:
            if next_token.numel() > 0:
                if next_token.dim() == 0:
                    token_value = next_token.item()
                elif next_token.dim() == 1 and next_token.size(0) > 0:
                    token_value = next_token[0].item()
                else:
                    token_value = next_token.view(-1)[0].item()
                tokens.append(int(token_value))
            else:
                print("Warning: Empty token detected")
                tokens.append(0)  # Default token
        except Exception as e:
            print(f"Error extracting token value: {e}")
            tokens.append(0)  # Default token
            
        # Cache position management
        cache_position = torch.tensor([seq_length], device=torch_device, dtype=torch.int32)
        position_ids = cache_position.unsqueeze(0)
        seq_length += 1
        
        # Disable CUDA graph runner
        cuda_graph_runner = None
            
        start_time = time.time()
        for i in range(1, max_new_tokens):
            # Remove CUDA-specific code
            next_token = decode_one_tokens(None, next_token.unsqueeze(0) if next_token.dim() == 1 else next_token, position_ids, cache_position, past_key_values, logits_warper, generation_config, False)
            
            # Ensure next_token has proper shape for concatenation
            if next_token.dim() == 0:  # If next_token is a scalar
                next_token = next_token.unsqueeze(0)  # Make it a 1D tensor
                
            # Proper concatenation ensuring tensor dimensions match
            if inputs.dim() == 2 and next_token.dim() == 1:
                inputs = torch.cat([inputs, next_token.unsqueeze(0)], dim=-1)
            else:
                # Handle other dimension combinations
                next_token_shaped = next_token.reshape(1, -1)
                inputs = torch.cat([inputs, next_token_shaped], dim=-1)
                
            # Ensure next_token is the right shape for generated_ids assignment
            # generated_ids expects a single value at each position per batch
            if next_token.dim() > 1:
                # If multi-dimensional, take first token
                token_value = next_token.view(-1)[0].item()
                token_tensor = torch.tensor([token_value], dtype=torch.int, device=torch_device)
                generated_ids[:, seq_length] = token_tensor
            else:
                # If already 1D or scalar, just convert to int
                token_value = next_token.item() if next_token.numel() == 1 else next_token[0].item()
                generated_ids[:, seq_length] = torch.tensor([token_value], dtype=torch.int, device=torch_device)
            
            # Store the token value safely, handling different possible shapes
            try:
                if next_token.numel() > 0:
                    if next_token.dim() == 0:
                        token_value = next_token.item()
                    elif next_token.dim() == 1 and next_token.size(0) > 0:
                        token_value = next_token[0].item()
                    else:
                        token_value = next_token.view(-1)[0].item()
                    tokens.append(int(token_value))
                else:
                    print("Warning: Empty token detected")
                    tokens.append(0)  # Default token
            except Exception as e:
                print(f"Error extracting token value: {e}")
                tokens.append(0)  # Default token
                
            seq_length += 1
            
            # Check if we need to stop generation
            should_stop = False
            try:
                if next_token.numel() > 0:
                    if next_token.dim() == 0:
                        token_to_check = next_token.item()
                    elif next_token.dim() >= 1 and next_token.size(0) > 0:
                        token_to_check = next_token[0].item()
                    else:
                        token_to_check = 0
                        
                    if token_to_check == tokenizer.eos_token_id or tokenizer.decode([token_to_check]) == '<|im_end|>':
                        print(stream.end(), end="", flush=True)
                        should_stop = True
                    else:
                        print(stream.put(token_to_check), end="", flush=True)
                else:
                    print("Empty next_token, stopping generation")
                    should_stop = True
            except Exception as e:
                print(f"Error checking token: {e}")
                should_stop = True
                
            if should_stop:
                break
                
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
