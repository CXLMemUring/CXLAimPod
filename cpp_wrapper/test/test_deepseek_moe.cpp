#include "kvcache.hpp"
#include <Python.h>
#include <c10/cuda/CUDAStream.h>
#include <iostream>
#include <stdexcept>
#include <torch/torch.h>
#include <vector>

int main() {
  try {
    // Initialize Python interpreter
    Py_Initialize();
    std::cout << "Python version: " << Py_GetVersion() << std::endl;
    std::cout << "Python path: " << Py_GetPath() << std::endl;

    // Create a KVCacheConfig for DeepSeek 671B MoE
    kvcache::KVCacheConfig kvconfig(
        32,                         // layer_num (DeepSeek 671B has 32 layers)
        32,                         // kv_head_num
        32,                         // q_head_num
        128,                        // head_dim
        8,                          // block_len
        4,                          // anchor_num
        kvcache::AnchorType::FIXED, // anchor_type
        kvcache::GGMLType::FP16,    // kv_type (using FP16 for efficiency)
        kvcache::RetrievalType::LAYER, // retrieval_type
        1,                             // layer_step
        1,                             // token_step
        0,                             // layer_offset
        256,                           // max_block_num
        1,                             // max_batch_size
        32                             // max_thread_num
    );

    std::cout << "Creating KVCache instance for DeepSeek 671B MoE..."
              << std::endl;
    kvcache::KVCache kvcache(kvconfig);
    std::cout << "KVCache instance created successfully" << std::endl;

    // Create test tensors for MoE inference
    std::cout << "Creating test tensors for MoE inference..." << std::endl;

    // Input query tensor (batch_size=1, seq_len=1, num_heads=32, head_dim=128)
    auto q_in = torch::randn(
        {1, 1, kvconfig.q_head_num, kvconfig.head_dim},
        torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16));

    // Output tensor for attention results
    auto output = torch::zeros(
        {1, 1, kvconfig.q_head_num, kvconfig.head_dim},
        torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16));

    // Attention log-sum-exp tensor
    auto attn_lse = torch::zeros(
        {1, 1, kvconfig.q_head_num},
        torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));

    // Block table for MoE routing
    auto block_table = torch::zeros(
        {1, kvconfig.max_block_num},
        torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));

    // Cache sequence lengths
    auto cache_seqlens = torch::zeros(
        {1}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));

    std::cout << "Test tensors created successfully" << std::endl;

    // Test MoE inference with attention operation
    std::cout << "Testing MoE inference with attention operation..."
              << std::endl;

    // Initialize MoE-specific parameters
    int pick_block_num = 4;  // Number of expert blocks to pick
    int init_block_num = 0;  // Initial block number
    int local_block_num = 4; // Number of local blocks

    // Perform attention operation with MoE routing
    bool success =
        kvcache.attn(q_in, output, attn_lse, 0, 0, &block_table, &cache_seqlens,
                     &pick_block_num, &init_block_num, &local_block_num);

    if (success) {
      std::cout << "MoE inference test passed successfully!" << std::endl;
      std::cout << "Output shape: " << output.sizes() << std::endl;

      // Test KVCache update for MoE
      std::cout << "Testing KVCache update for MoE..." << std::endl;

      // Create K and V tensors for cache update
      auto k_in = torch::randn(
          {1, 1, kvconfig.kv_head_num, kvconfig.head_dim},
          torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16));
      auto v_in = torch::randn(
          {1, 1, kvconfig.kv_head_num, kvconfig.head_dim},
          torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16));

      // Update KVCache for the first expert block
      bool update_success =
          kvcache.update_kvcache_one_block_fp16(k_in, v_in, 0, 0);
      if (update_success) {
        std::cout << "KVCache update successful for expert block 0"
                  << std::endl;
      } else {
        std::cerr << "Failed to update KVCache for expert block 0" << std::endl;
      }
    } else {
      std::cerr << "MoE inference test failed" << std::endl;
      return 1;
    }

    // Finalize Python interpreter
    Py_Finalize();
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Test failed with exception: " << e.what() << std::endl;
    if (Py_IsInitialized()) {
      PyErr_Print(); // Print any Python errors
    }
    return 1;
  } catch (...) {
    std::cerr << "Test failed with unknown exception" << std::endl;
    if (Py_IsInitialized()) {
      PyErr_Print(); // Print any Python errors
    }
    return 1;
  }
}