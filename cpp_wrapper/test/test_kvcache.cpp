#include "kvcache.hpp"
#include <Python.h>
#include <c10/cuda/CUDAStream.h>
#include <iostream>
#include <stdexcept>
#include <torch/torch.h>
// void c10::detail::torchInternalAssertFail(
//     char const *, char const *, unsigned int, char const *,
//     std::__cxx11::basic_string<char, std::char_traits<char>,
//                                std::allocator<char>> const &) {}
// void c10::detail::torchCheckFail(
//     char const *, char const *, unsigned int,
//     std::__cxx11::basic_string<char, std::char_traits<char>,
//                                std::allocator<char>> const &) {}
int main() {
  try {
    // Initialize Python interpreter
    Py_Initialize();

    // Print Python version and paths
    std::cout << "Python version: " << Py_GetVersion() << std::endl;
    std::cout << "Python path: " << Py_GetPath() << std::endl;

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

    std::cout << "Creating KVCache instance..." << std::endl;

    // Create KVCache instance
    kvcache::KVCache kvcache(kvconfig);
    std::cout << "KVCache instance created successfully" << std::endl;

    // Create test tensors
    std::cout << "Creating test tensors..." << std::endl;
    auto q_in = torch::randn(
        {1, 1, kvconfig.q_head_num, kvconfig.head_dim},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat16));
    auto output = torch::zeros(
        {1, 1, kvconfig.q_head_num, kvconfig.head_dim},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat16));
    auto attn_lse = torch::zeros(
        {1, 1, kvconfig.q_head_num},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
    std::cout << "Test tensors created successfully" << std::endl;

    // Test attention operation
    std::cout << "Testing attention operation..." << std::endl;
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