"""
Optimized version of local_chat.py for high-performance inference
Targets 20+ tokens/s on CPU by fixing critical performance bottlenecks
"""

import os
import platform
import sys

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)
import torch
import logging

# Suppress torch dynamo errors
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
    TextStreamer,
)
import json
import fire
from ktransformers.optimize.optimize import optimize_and_load_gguf
from ktransformers.models.modeling_deepseek import DeepseekV2ForCausalLM
from ktransformers.models.modeling_qwen2_moe import Qwen2MoeForCausalLM
from ktransformers.models.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from ktransformers.models.modeling_llama import LlamaForCausalLM
from ktransformers.models.modeling_mixtral import MixtralForCausalLM
from ktransformers.util.utils import get_compute_capability
from ktransformers.server.config.config import Config
from ktransformers.operators.flashinfer_wrapper import flashinfer_enabled
from ktransformers.util.vendors import device_manager, get_device, to_device, GPUVendor



custom_models = {
    "DeepseekV2ForCausalLM": DeepseekV2ForCausalLM,
    "DeepseekV3ForCausalLM": DeepseekV3ForCausalLM,
    "Qwen2MoeForCausalLM": Qwen2MoeForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "MixtralForCausalLM": MixtralForCausalLM,
}

ktransformer_rules_dir = (
    os.path.dirname(os.path.abspath(__file__)) + "/optimize/optimize_rules/"
)
default_optimize_rules = {
    "DeepseekV2ForCausalLM": ktransformer_rules_dir + "DeepSeek-V2-Chat.yaml",
    "DeepseekV3ForCausalLM": ktransformer_rules_dir + "DeepSeek-V3-Chat-memory-optimized.yaml",  # Memory-efficient config
    "Qwen2MoeForCausalLM": ktransformer_rules_dir + "Qwen2-57B-A14B-Instruct.yaml",
    "LlamaForCausalLM": ktransformer_rules_dir + "Internlm2_5-7b-Chat-1m.yaml",
    "MixtralForCausalLM": ktransformer_rules_dir + "Mixtral.yaml",
}


def setup_cpu_optimization():
    """Configure CPU for optimal performance"""
    # Set environment variables for optimal CPU performance
    num_cores = os.cpu_count()
    os.environ["OMP_NUM_THREADS"] = str(num_cores)
    os.environ["MKL_NUM_THREADS"] = str(num_cores)
    os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
    os.environ["KMP_BLOCKTIME"] = "0"
    os.environ["MALLOC_CONF"] = "oversize_threshold:1,background_thread:true"
    
    # Enable Intel MKL optimizations
    if hasattr(torch, 'set_num_threads'):
        torch.set_num_threads(num_cores)
    
    # Set inter-op threads for better parallelism
    torch.set_num_interop_threads(max(4, num_cores // 2))
    
    print(f"CPU optimization configured for {num_cores} cores")


def local_chat(
    model_path: str | None = None,
    optimize_config_path: str = None,
    gguf_path: str | None = None,
    max_new_tokens: int = 10,
    cpu_infer: int = Config().cpu_infer,
    use_cuda_graph: bool = False,
    prompt_file : str | None = None,
    mode: str = "normal",
    force_think: bool = True,
    chunk_size: int = 8192,
    enable_ipex: bool = True,  # New parameter to control IPEX optimization
):
    # Set up CPU optimization first
    setup_cpu_optimization()
    
    torch.set_grad_enabled(False)
    Config().cpu_infer = cpu_infer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if mode == 'long_context':
        assert config.architectures[0] == "LlamaForCausalLM", "only LlamaForCausalLM support long_context mode"
        torch.set_default_dtype(torch.float16)
    else:
        torch.set_default_dtype(config.torch_dtype)

    with torch.device("meta"):
        if config.architectures[0] in custom_models:
            print("using custom modeling_xxx.py.")
            if (
                "Qwen2Moe" in config.architectures[0]
            ):  # Qwen2Moe must use flash_attention_2 to avoid overflow.
                config._attn_implementation = "flash_attention_2"
            if "Llama" in config.architectures[0]:
                config._attn_implementation = "eager"
            if "Mixtral" in config.architectures[0]:
                config._attn_implementation = "flash_attention_2"

            model = custom_models[config.architectures[0]](config)
        else:
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True, attn_implementation="flash_attention_2"
            )

    if optimize_config_path is None:
        if config.architectures[0] in default_optimize_rules:
            print("using default_optimize_rule for", config.architectures[0])
            optimize_config_path = default_optimize_rules[config.architectures[0]]
        else:
            optimize_config_path = input(
                "please input the path of your rule file(yaml file containing optimize rules):"
            )

    if gguf_path is None:
        gguf_path = input(
            "please input the path of your gguf file(gguf file in the dir containing input gguf file must all belong to current model):"
        )
    
    optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
    
    # Apply IPEX optimization ONCE before inference, not in the decode loop
    if enable_ipex and cpu_infer > 0:
        try:
            import intel_extension_for_pytorch as ipex
            print("Applying IPEX optimization to model...")
            # Use appropriate dtype based on quantization
            opt_dtype = torch.int8 if hasattr(config, 'quantization_config') else torch.bfloat16
            model = ipex.optimize(model, dtype=opt_dtype, inplace=False)
            print(f"IPEX optimization applied with dtype={opt_dtype}")
        except ImportError:
            print("Warning: IPEX not available, continuing without IPEX optimization")
        except Exception as e:
            print(f"Warning: IPEX optimization failed: {e}, continuing without optimization")
    
    # Enable torch compile for additional optimization
    if hasattr(torch, 'compile') and cpu_infer > 0:
        try:
            print("Compiling model with torch.compile...")
            model = torch.compile(model, mode="reduce-overhead", backend="inductor")
            print("Model compiled successfully")
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}, continuing without compilation")
    
    try:
        model.generation_config = GenerationConfig.from_pretrained(model_path)
    except Exception as e:
        print(f"generation config can't auto create, make default. Message: {e}")
        gen_config = GenerationConfig(
            temperature=0.6,
            top_p=0.95,
            do_sample=True
        )
        model.generation_config = gen_config
    
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    
    model.eval()
    logging.basicConfig(level=logging.INFO)

    system = platform.system()
    if system == "Windows":
        os.system("cls")
    else:
        os.system("clear")

    content = "Please write a piece of quicksort code in C++."
    if content.startswith('"""'):  # prefix """
        # multi lines input
        content = content[3:] + "\n"
        while True:
            line = input("")
            if line.endswith('"""'):
                # end multi lines input
                line = line[:-3]  # suffix """
                if line:
                    content += line + "\n"
                break
            else:
                content += line + "\n"
    if content == "":
        if prompt_file != None:
            content = open(prompt_file, "r").read()
        else:
            content = "Please write a piece of quicksort code in C++."
    elif os.path.isfile(content):
        content = open(content, "r").read()
    messages = [{"role": "user", "content": content}]
    input_tensor = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    if mode == 'long_context':
        assert Config().long_context_config['max_seq_len'] > input_tensor.shape[1] + max_new_tokens, \
        "please change max_seq_len in  ~/.ktransformers/config.yaml"
    
    # Use optimized prefill_and_generate function
    generated = optimized_prefill_and_generate(
        model, tokenizer, input_tensor, max_new_tokens, use_cuda_graph, mode
    )
    return generated


def optimized_prefill_and_generate(model, tokenizer, inputs, max_new_tokens=10000, use_cuda_graph: bool = False,
                         mode = 'normal', force_think: bool = False, chunk_size = 16384):
    """
    Optimized version of prefill_and_generate specifically for CPU inference
    with performance optimizations:
    - Efficient tensor operations
    - Minimized memory allocations
    - Optimized decoding loop
    - Better cache management
    """
    import time
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch._dynamo.config.suppress_errors = True
    
    batch_size, seq_length = inputs.shape
    torch_device = "cpu"
    inputs = inputs.to(torch_device)
    
    # Force CPU mode - disable all CUDA features
    use_cuda_graph = False
    
    # Pre-allocate output buffer for better memory efficiency
    tokens = torch.zeros((batch_size, seq_length + max_new_tokens), dtype=torch.long, device=torch_device)
    tokens[:, :seq_length] = inputs
    
    def optimized_decode_one_token(model, cur_token, position_ids, cache_position, past_key_values, generation_config):
        """Optimized single token decoding for CPU"""
        try:
            # Ensure proper tensor shapes and dtypes
            if cur_token.dim() == 1:
                cur_token = cur_token.unsqueeze(0)
            elif cur_token.dim() > 2:
                cur_token = cur_token.view(cur_token.size(0), -1)
                
            cur_token = cur_token.to(device="cpu", dtype=torch.long)
            position_ids = position_ids.to(device="cpu", dtype=torch.long)
            cache_position = cache_position.to(device="cpu", dtype=torch.long)
            
            # Get embeddings
            with torch.no_grad():
                inputs_embeds = model.model.embed_tokens(cur_token)
                
                # Forward pass with minimal overhead
                outputs = model(
                    inputs_embeds=inputs_embeds,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    use_cache=True,
                    return_dict=True
                )
                
                # Extract next token efficiently
                logits = outputs.logits
                if logits.dim() == 3:
                    next_token_logits = logits[:, -1, :]
                else:
                    next_token_logits = logits
                
                # Simple argmax for greedy decoding (fastest)
                if generation_config.do_sample:
                    # Apply temperature
                    next_token_logits = next_token_logits / generation_config.temperature
                    # Top-p sampling
                    if generation_config.top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > generation_config.top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = -float('Inf')
                    
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1)
                
                return next_token, outputs.past_key_values
                
        except Exception as e:
            print(f"Error in optimized_decode_one_token: {e}")
            # Return EOS token as fallback
            if hasattr(model.config, 'eos_token_id'):
                return torch.tensor([model.config.eos_token_id], device="cpu"), past_key_values
            else:
                return torch.tensor([0], device="cpu"), past_key_values
    
    def optimized_chunk_prefill(model, inputs, chunk_size=8192):
        """Optimized prefill with chunking for large sequences"""
        try:
            batch_size, seq_len = inputs.shape
            
            # Initialize KV cache
            past_key_values = None
            position_ids = torch.arange(seq_len, dtype=torch.long, device=inputs.device).unsqueeze(0)
            
            # Process in chunks if sequence is long
            if seq_len > chunk_size:
                for start_idx in range(0, seq_len, chunk_size):
                    end_idx = min(start_idx + chunk_size, seq_len)
                    chunk_inputs = inputs[:, start_idx:end_idx]
                    chunk_position_ids = position_ids[:, start_idx:end_idx]
                    
                    with torch.no_grad():
                        inputs_embeds = model.model.embed_tokens(chunk_inputs)
                        outputs = model(
                            inputs_embeds=inputs_embeds,
                            position_ids=chunk_position_ids,
                            past_key_values=past_key_values,
                            use_cache=True,
                            return_dict=True
                        )
                        past_key_values = outputs.past_key_values
            else:
                # Process entire sequence at once for short sequences
                with torch.no_grad():
                    inputs_embeds = model.model.embed_tokens(inputs)
                    outputs = model(
                        inputs_embeds=inputs_embeds,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True
                    )
                    past_key_values = outputs.past_key_values
            
            # Extract last token logits
            logits = outputs.logits
            if logits.dim() == 3:
                last_logits = logits[:, -1, :]
            else:
                last_logits = logits
                
            return last_logits, past_key_values, seq_len
            
        except Exception as e:
            print(f"Error in optimized_chunk_prefill: {e}")
            raise
    
    # Start timing
    start_time = time.time()
    
    # Prefill phase
    print(f"Starting prefill for {seq_length} tokens...")
    prefill_start = time.time()
    last_logits, past_key_values, cache_len = optimized_chunk_prefill(model, inputs, chunk_size)
    prefill_time = time.time() - prefill_start
    print(f"Prefill completed in {prefill_time:.2f}s ({seq_length/prefill_time:.1f} tokens/s)")
    
    # Initialize generation variables
    generation_config = model.generation_config
    position_ids = torch.tensor([[cache_len]], dtype=torch.long, device=torch_device)
    cache_position = torch.arange(cache_len, cache_len + 1, device=torch_device)
    
    # Get first token from prefill
    if generation_config.do_sample:
        # Apply temperature
        last_logits = last_logits / generation_config.temperature
        probs = torch.softmax(last_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
    else:
        next_token = torch.argmax(last_logits, dim=-1)
    
    tokens[:, seq_length] = next_token
    
    # Decoding phase with optimizations
    print(f"Starting generation of up to {max_new_tokens} tokens...")
    decode_start = time.time()
    generated_tokens = 1
    
    # Pre-compile stop tokens for efficiency
    stop_tokens = set()
    if hasattr(generation_config, 'eos_token_id'):
        if isinstance(generation_config.eos_token_id, list):
            stop_tokens.update(generation_config.eos_token_id)
        else:
            stop_tokens.add(generation_config.eos_token_id)
    
    # Main generation loop
    for i in range(1, max_new_tokens):
        # Update position information
        position_ids = torch.tensor([[cache_len + i]], dtype=torch.long, device=torch_device)
        cache_position = torch.arange(cache_len + i, cache_len + i + 1, device=torch_device)
        
        # Generate next token
        next_token, past_key_values = optimized_decode_one_token(
            model, next_token, position_ids, cache_position, past_key_values, generation_config
        )
        
        # Store token
        tokens[:, seq_length + i] = next_token
        generated_tokens += 1
        
        # Check for stop conditions
        if next_token.item() in stop_tokens:
            print(f"\nEOS token generated at position {i}")
            break
        
        # Print periodic updates
        if i % 10 == 0:
            elapsed = time.time() - decode_start
            tokens_per_sec = i / elapsed
            print(f"\rGenerated {i}/{max_new_tokens} tokens ({tokens_per_sec:.1f} tokens/s)", end='', flush=True)
    
    # Final statistics
    total_time = time.time() - start_time
    decode_time = time.time() - decode_start
    
    print(f"\n\nGeneration complete:")
    print(f"  Prefill: {seq_length} tokens in {prefill_time:.2f}s ({seq_length/prefill_time:.1f} tokens/s)")
    print(f"  Decode: {generated_tokens} tokens in {decode_time:.2f}s ({generated_tokens/decode_time:.1f} tokens/s)")
    print(f"  Total: {total_time:.2f}s")
    
    # Return only the generated portion
    return tokens[:, :seq_length + generated_tokens]


if __name__ == "__main__":
    fire.Fire(local_chat)