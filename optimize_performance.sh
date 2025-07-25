#!/bin/bash

# Optimized performance script for KTransformers
# Target: 20+ tokens/s on CPU

echo "Setting up optimized CPU environment for KTransformers..."

# Get number of CPU cores
NUM_CORES=33
echo "Detected $NUM_CORES CPU cores"

# Set optimal environment variables for CPU inference
export OMP_NUM_THREADS=$NUM_CORES
export MKL_NUM_THREADS=$NUM_CORES
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=0
export MALLOC_CONF="oversize_threshold:1,background_thread:true"

# Enable Intel MKL optimizations
export MKL_DYNAMIC=FALSE
export MKL_CBWR=COMPATIBLE

# Optimize memory allocation
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=1073741824

# Control memory usage
export OMP_WAIT_POLICY=PASSIVE
export MKL_ENABLE_INSTRUCTIONS=AVX512_E1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

# Enable huge pages if available
if [ -d /sys/kernel/mm/hugepages ]; then
    export THP_DISABLE=0
    echo "Transparent Huge Pages enabled"
fi

# Check for Intel CPU and enable AMX if available
if lscpu | grep -q "amx"; then
    echo "Intel AMX detected - enabling AMX optimizations"
    export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
fi

echo "Environment configured for optimal performance"
echo "----------------------------------------"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "KMP_AFFINITY=$KMP_AFFINITY"
echo "----------------------------------------"

# Run the optimized local chat with provided arguments
echo "Starting optimized inference..."

# Default values
MODEL_PATH="${MODEL_PATH:-deepseek-ai/deepseek-v3-0324}"
GGUF_PATH="${GGUF_PATH:-/home/victoryang00/DeepSeek-V3-0324-Q4_K_M-00001-of-00009.gguf}"
OPTIMIZE_CONFIG="${OPTIMIZE_CONFIG:-optimize/optimize_rules/DeepSeek-V3-Chat-amx.yaml}"
MAX_TOKENS="${MAX_TOKENS:-10}"

# Run with optimized script
echo python optimized_local_chat.py \
    --model_path="$MODEL_PATH" \
    --gguf_path="$GGUF_PATH" \
    --optimize_config_path="$OPTIMIZE_CONFIG" \
    --max_new_tokens=$MAX_TOKENS \
    --cpu_infer=$NUM_CORES \
    "$@"
python optimized_local_chat.py \
    --model_path="$MODEL_PATH" \
    --gguf_path="$GGUF_PATH" \
    --optimize_config_path="$OPTIMIZE_CONFIG" \
    --max_new_tokens=$MAX_TOKENS \
    --cpu_infer=$NUM_CORES \
    "$@"