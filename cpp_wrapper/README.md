# KVCache C++ Wrapper

This project provides a C++ wrapper for the KVCache functionality, making it easier to use in C++ applications.

## Prerequisites

- CMake 3.10 or higher
- C++17 compatible compiler
- PyTorch
- CUDA
- Python 3.x development files

## Building the Project

1. Create a build directory:
```bash
mkdir build
cd build
```

2. Configure with CMake:
```bash
cmake ..
```

3. Build the project:
```bash
make
```

## Usage

The project provides two main classes:

### KVCache

The `KVCache` class provides functionality for managing key-value caches:

```cpp
#include "kvcache.hpp"

// Create configuration
kvcache::KVCacheConfig config(
    32,                    // layer_num
    8,                     // kv_head_num
    32,                    // q_head_num
    128,                   // head_dim
    256,                   // block_len
    4,                     // anchor_num
    kvcache::AnchorType::FIXED,  // anchor_type
    kvcache::GGMLType::FP16,     // kv_type
    kvcache::RetrievalType::LAYER, // retrieval_type
    1,                     // layer_step
    1,                     // token_step
    0,                     // layer_offset
    512,                   // max_block_num
    4,                     // max_batch_size
    32                     // max_thread_num
);

// Create KVCache instance
kvcache::KVCache cache(config);

// Use the cache
torch::Tensor q_in = torch::rand({4, 1, 32, 128}, torch::kFloat16);
torch::Tensor output = torch::zeros({4, 1, 32, 128}, torch::kFloat16);
torch::Tensor attn_lse = torch::zeros({4, 1, 32}, torch::kFloat32);

cache.attn(q_in, output, attn_lse, 0, 0);
```

### CPUInfer

The `CPUInfer` class provides functionality for CPU-based inference:

```cpp
#include "cpuinfer.hpp"

// Create CPUInfer instance with 4 threads
kvcache::CPUInfer infer(4);

// Submit a task
void* task = nullptr; // Your task object
infer.submit(task);

// Or submit with CUDA stream
cudaStream_t stream;
cudaStreamCreate(&stream);
infer.submit_with_cuda_stream(stream, task);

// Synchronize
infer.sync();
// Or synchronize with CUDA stream
infer.sync_with_cuda_stream(stream);
```

## License

This project is licensed under the same terms as the original KVCache project. 

# CPUInfer C++ Extension

This is a C++ extension for the CPUInfer project that provides high-performance implementations of key operations.

## Prerequisites

- Python 3.7+
- PyTorch 1.8+
- CUDA 11.0+
- CMake 3.10+
- C++17 compatible compiler

## Building the Extension

1. First, make sure you have all the prerequisites installed.

2. Clone the repository and navigate to the `cpp_wrapper` directory:
```bash
cd cpp_wrapper
```

3. Build the extension using pip:
```bash
pip install -e .
```

This will compile the C++ extension and install it in development mode.

## Usage

Here's an example of how to use the KVCache class:

```python
import torch
from cpuinfer_ext.kvcache import KVCache, KVCacheConfig, AnchorType, GGMLType, RetrievalType

# Create KVCacheConfig
config = KVCacheConfig(
    layer_num=32,
    kv_head_num=32,
    q_head_num=32,
    head_dim=128,
    block_len=8,
    anchor_num=4,
    anchor_type=AnchorType.FIXED,
    kv_type=GGMLType.FP16,
    retrieval_type=RetrievalType.LAYER,
    layer_step=1,
    token_step=1,
    layer_offset=0,
    max_block_num=256,
    max_batch_size=1,
    max_thread_num=32
)

# Create KVCache instance
kvcache = KVCache(config)

# Create test tensors
q_in = torch.randn(1, 1, config.q_head_num, config.head_dim, 
                   device=torch.device('cuda'), dtype=torch.float16)
output = torch.zeros(1, 1, config.q_head_num, config.head_dim,
                    device=torch.device('cuda'), dtype=torch.float16)
attn_lse = torch.zeros(1, 1, config.q_head_num,
                      device=torch.device('cuda'), dtype=torch.float32)

# Run attention operation
success = kvcache.attn(q_in, output, attn_lse, 0, 0)
if success:
    print("Attention operation successful!")
else:
    print("Attention operation failed!")
```

## API Reference

### KVCacheConfig

Configuration structure for the KVCache class.

```cpp
struct KVCacheConfig {
    int layer_num;           // Number of layers
    int kv_head_num;        // Number of KV heads
    int q_head_num;         // Number of query heads
    int head_dim;           // Dimension of each head
    int block_len;          // Length of each block
    int anchor_num;         // Number of anchors
    AnchorType anchor_type; // Type of anchor selection
    GGMLType kv_type;       // Type of KV cache
    RetrievalType retrieval_type; // Type of retrieval
    int layer_step;         // Step size for layer selection
    int token_step;         // Step size for token selection
    int layer_offset;       // Offset for layer selection
    int max_block_num;      // Maximum number of blocks
    int max_batch_size;     // Maximum batch size
    int max_thread_num;     // Maximum number of threads
};
```

### KVCache Class

Main class for KV cache operations.

```cpp
class KVCache {
public:
    explicit KVCache(const KVCacheConfig& config);
    ~KVCache();

    // Core operations
    bool load_kvcache(const std::string& tensor_file_path);
    bool dump_kvcache(torch::Tensor& block_table, int cache_total_len, 
                     const std::string& tensor_file_path);
    void update_cache_total_len(int cache_total_len);
    int get_cache_total_len() const;

    // Attention operations
    bool attn(torch::Tensor& q_in, torch::Tensor& output, torch::Tensor& attn_lse,
              int layer_idx, int generate_token_idx, torch::Tensor* block_table = nullptr,
              torch::Tensor* cache_seqlens = nullptr, int* pick_block_num = nullptr,
              int* init_block_num = nullptr, int* local_block_num = nullptr);

    // KV Cache operations
    bool update_kvcache_one_block_fp16(torch::Tensor& k_in, torch::Tensor& v_in,
                                      int layer_id, int block_idx);
    bool get_kvcache_one_block_fp16(torch::Tensor& k_in, torch::Tensor& v_in,
                                   int layer_id, int block_idx);

    // Importance operations
    bool update_importance_one_block(torch::Tensor& importance, int layer_id, int block_idx);
    bool get_importance_one_block(torch::Tensor& importance, int layer_id, int block_idx);

    // Anchor operations
    bool get_anchor_one_block(torch::Tensor& anchor, int layer_id, int block_idx);
    bool update_anchor_one_block(torch::Tensor& anchor, int layer_id, int block_idx);
};
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 