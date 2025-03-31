#include "kvcache.hpp"
#include <Python.h>
#include <c10/cuda/CUDAStream.h>
#include <iostream>
#include <stdexcept>
#include <torch/torch.h>

int main() {
  try {
    // Initialize Python interpreter
    Py_Initialize();

    // Create a KVCacheConfig with all required parameters
    kvcache::KVCacheConfig kvconfig(
        32,                            // layer_num
        32,                            // kv_head_num
        32,                            // q_head_num
        128,                           // head_dim
        8,                             // block_len
        4,                             // anchor_num
        kvcache::AnchorType::FIXED,    // anchor_type
        kvcache::GGMLType::FP16,       // kv_type
        kvcache::RetrievalType::LAYER, // retrieval_type
        1,                             // layer_step
        1,                             // token_step
        0,                             // layer_offset
        256,                           // max_block_num
        1,                             // max_batch_size
        32                             // max_thread_num
    );

    // Create KVCache instance
    kvcache::KVCache kvcache(kvconfig);

    // Create test tensors
    auto q_in = torch::randn(
        {1, 1, kvconfig.q_head_num, kvconfig.head_dim},
        torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16));
    auto output = torch::zeros(
        {1, 1, kvconfig.q_head_num, kvconfig.head_dim},
        torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16));
    auto attn_lse = torch::zeros(
        {1, 1, kvconfig.q_head_num},
        torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));

    // Test attention operation
    bool success = kvcache.attn(q_in, output, attn_lse, 0, 0, nullptr, nullptr,
                                nullptr, nullptr, nullptr);

    if (success) {
      std::cout << "Test passed: KVCache attention operation successful!"
                << std::endl;
      std::cout << "Output shape: " << output.sizes() << std::endl;
    } else {
      std::cerr << "Test failed: KVCache attention operation failed"
                << std::endl;
      return 1;
    }

    // Finalize Python interpreter
    Py_Finalize();
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Test failed with exception: " << e.what() << std::endl;
    return 1;
  }
}