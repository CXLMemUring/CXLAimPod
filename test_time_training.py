"""
Description  :  Test-time training implementation for ktransformers
Author       : Boxin Zhang, Azure-Tang
Version      : 0.1.0
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
"""

import os
import platform
import sys

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)
import torch
import logging
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
from ktransformers.util.utils import prefill_and_generate, get_compute_capability
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
    "DeepseekV3ForCausalLM": ktransformer_rules_dir + "DeepSeek-V3-Chat.yaml",
    "Qwen2MoeForCausalLM": ktransformer_rules_dir + "Qwen2-57B-A14B-Instruct.yaml",
    "LlamaForCausalLM": ktransformer_rules_dir + "Internlm2_5-7b-Chat-1m.yaml",
    "MixtralForCausalLM": ktransformer_rules_dir + "Mixtral.yaml",
}

# Default model path for TinyLlama which was successfully downloaded
DEFAULT_MODEL_PATH = "TinyLlama-1.1B-Chat-v1.0"

def train_step(model, input_ids, targets, optimizer, device="cuda"):
    """Perform a single training step on the model"""
    input_ids = input_ids.to(device)
    targets = targets.to(device)
    
    # Forward pass
    outputs = model(input_ids, labels=targets)
    loss = outputs.loss
    
    # Backward pass
    loss.backward()
    
    # Update weights
    optimizer.step()
    optimizer.zero_grad()
    
    return loss.item()

def test_time_training(
    model_path: str | None = None,
    optimize_config_path: str = None,
    gguf_path: str | None = None,
    max_new_tokens: int = 1000,
    cpu_infer: int = Config().cpu_infer,
    use_cuda_graph: bool = False,
    prompt_file: str | None = None,
    mode: str = "normal",
    force_think: bool = True,
    chunk_size: int = 8192,
    # Test-time training parameters
    enable_training: bool = True,
    learning_rate: float = 1e-5,
    num_train_steps: int = 5,
    train_data_file: str | None = None,
    trainable_layers: str = "lm_head,norm",  # Comma-separated list of layer names to train
    dtype: str = "float16"  # Training precision
):
    """
    Test-time training implementation for ktransformers.
    
    Args:
        model_path: Path to the model
        optimize_config_path: Path to optimization config
        gguf_path: Path to GGUF files
        max_new_tokens: Maximum number of new tokens to generate
        cpu_infer: Whether to use CPU for inference
        use_cuda_graph: Whether to use CUDA graph
        prompt_file: Path to prompt file
        mode: Inference mode
        force_think: Whether to force thinking
        chunk_size: Chunk size for processing
        enable_training: Whether to enable test-time training
        learning_rate: Learning rate for test-time training
        num_train_steps: Number of training steps
        train_data_file: Path to training data file (JSON format with prompt/completion pairs)
        trainable_layers: Comma-separated list of layer names to train
        dtype: Training precision
    """
    
    # Set up training dtype
    train_dtype = torch.float16
    if dtype == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        train_dtype = torch.bfloat16
    elif dtype == "float32":
        train_dtype = torch.float32
        
    # Only enable gradients if training is enabled
    torch.set_grad_enabled(enable_training)

    Config().cpu_infer = cpu_infer
    
    # Use TinyLlama as default if no model path provided
    if model_path is None:
        print(f"No model path provided, using default TinyLlama model: {DEFAULT_MODEL_PATH}")
        model_path = DEFAULT_MODEL_PATH
        
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        print("Please make sure the model is properly downloaded.")
        return

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
                config, trust_remote_code=True
            )

    if optimize_config_path is None:
        if config.architectures[0] in default_optimize_rules:
            print("using default_optimize_rule for", config.architectures[0])
            optimize_config_path = default_optimize_rules[config.architectures[0]]
        else:
            optimize_config_path = input(
                "please input the path of your rule file(yaml file containing optimize rules):"
            )

    # Check if gguf files exist for the model
    if gguf_path is None:
        model_gguf_path = os.path.join(os.path.dirname(model_path), "gguf")
        if os.path.exists(model_gguf_path) and any(f.endswith(".gguf") for f in os.listdir(model_gguf_path)):
            gguf_path = model_gguf_path
            print(f"Using GGUF files from: {gguf_path}")
        else:
            gguf_path = input(
                "please input the path of your gguf file(gguf file in the dir containing input gguf file must all belong to current model):"
            )
    
    try:
        optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
    except Exception as e:
        print(f"Error loading GGUF files: {e}")
        print("Please make sure the GGUF files are properly downloaded.")
        return
    
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
    
    # Configure model for training if enabled
    if enable_training:
        print("Enabling test-time training mode")
        # Move model to train mode
        model.train()
        
        # Set trainable parameters based on specified layers
        trainable_layer_names = trainable_layers.split(",")
        for name, param in model.named_parameters():
            param.requires_grad = False  # Freeze all parameters by default
            
            # Unfreeze specified layers
            if any(layer_name in name for layer_name in trainable_layer_names):
                param.requires_grad = True
                print(f"Setting {name} as trainable")
        
        # Set up optimizer
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            print("Warning: No trainable parameters found!")
            enable_training = False
        else:
            print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")
            optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
            
            # Load training data if provided
            training_samples = []
            if train_data_file and os.path.exists(train_data_file):
                try:
                    with open(train_data_file, 'r') as f:
                        training_data = json.load(f)
                    
                    for item in training_data:
                        if 'prompt' in item and 'completion' in item:
                            training_samples.append({
                                'prompt': item['prompt'],
                                'completion': item['completion']
                            })
                    print(f"Loaded {len(training_samples)} training samples")
                except Exception as e:
                    print(f"Error loading training data: {e}")
            
            # If no training data file or it's empty, create a default training sample
            if not training_samples:
                print("No training data provided, using prompt as training data")
                content = "Please write a piece of quicksort code in C++."
                if prompt_file and os.path.exists(prompt_file):
                    content = open(prompt_file, "r").read()
                
                # Create a simple completion for the prompt
                completion = "Here's a C++ implementation of quicksort:\n```cpp\n#include <vector>\n\nint partition(std::vector<int>& arr, int low, int high) {\n    int pivot = arr[high];\n    int i = low - 1;\n    \n    for (int j = low; j < high; j++) {\n        if (arr[j] <= pivot) {\n            i++;\n            std::swap(arr[i], arr[j]);\n        }\n    }\n    \n    std::swap(arr[i + 1], arr[high]);\n    return i + 1;\n}\n\nvoid quicksort(std::vector<int>& arr, int low, int high) {\n    if (low < high) {\n        int pi = partition(arr, low, high);\n        \n        quicksort(arr, low, pi - 1);\n        quicksort(arr, pi + 1, high);\n    }\n}\n```"
                
                training_samples.append({
                    'prompt': content,
                    'completion': completion
                })
            
            # Perform test-time training
            print(f"Starting test-time training for {num_train_steps} steps")
            device = "cuda" if torch.cuda.is_available() and not cpu_infer else "cpu"
            
            for step in range(num_train_steps):
                total_loss = 0
                for sample in training_samples:
                    # Prepare input and target
                    prompt = sample['prompt']
                    completion = sample['completion']
                    
                    # Format as chat message
                    messages = [{"role": "user", "content": prompt}]
                    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors=None)
                    
                    # Create target by concatenating input and completion
                    target_text = input_text + completion
                    
                    # Tokenize
                    input_tokens = tokenizer(input_text, return_tensors="pt")
                    target_tokens = tokenizer(target_text, return_tensors="pt")
                    
                    # Ensure targets are properly aligned for causal LM training
                    input_ids = input_tokens.input_ids
                    labels = target_tokens.input_ids
                    
                    # Train step
                    loss = train_step(model, input_ids, labels, optimizer, device)
                    total_loss += loss
                    
                avg_loss = total_loss / len(training_samples)
                print(f"Step {step+1}/{num_train_steps}, Loss: {avg_loss:.4f}")
    else:
        # Evaluation mode if not training
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
    
    # If we were in training mode, switch back to eval for generation
    if enable_training:
        model.eval()
        torch.set_grad_enabled(False)
    
    torch.set_default_dtype(
        torch.bfloat16
    )  # TODO: Remove this, replace dtype using config
    
    generated = prefill_and_generate(
        model, tokenizer, input_tensor, max_new_tokens, use_cuda_graph, mode
    )

if __name__ == "__main__":
    fire.Fire(test_time_training)
