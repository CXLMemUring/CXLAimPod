#pragma once

#include <memory>
#include <string>
#include <torch/torch.h>
#include <Python.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Size.h>
#include <torch/csrc/Types.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/python_cpp_function.h>
#include <torch/csrc/autograd/python_hook.h>
#include <torch/csrc/autograd/python_variable_indexing.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/autograd/utils/error_messages.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/tensor/python_tensor.h>
#include <pybind11/pybind11.h>

namespace kvcache {

enum class AnchorType { FIXED, QUEST, DYNAMIC, BLOCK_MEAN, BLOCK_MAX };

enum class GGMLType { FP16, FP32, Q4_0, Q8_0 };

enum class RetrievalType { LAYER, QHEAD, KVHEAD };

struct KVCacheConfig {
  int layer_num;
  int kv_head_num;
  int q_head_num;
  int head_dim;
  int block_len;
  int anchor_num;
  AnchorType anchor_type;
  GGMLType kv_type;
  RetrievalType retrieval_type;
  int layer_step;
  int token_step;
  int layer_offset;
  int max_block_num;
  int max_batch_size;
  int max_thread_num;

  KVCacheConfig(int layer_num_, int kv_head_num_, int q_head_num_,
                int head_dim_, int block_len_, int anchor_num_,
                AnchorType anchor_type_, GGMLType kv_type_,
                RetrievalType retrieval_type_, int layer_step_, int token_step_,
                int layer_offset_, int max_block_num_, int max_batch_size_,
                int max_thread_num_)
      : layer_num(layer_num_), kv_head_num(kv_head_num_),
        q_head_num(q_head_num_), head_dim(head_dim_), block_len(block_len_),
        anchor_num(anchor_num_), anchor_type(anchor_type_), kv_type(kv_type_),
        retrieval_type(retrieval_type_), layer_step(layer_step_),
        token_step(token_step_), layer_offset(layer_offset_),
        max_block_num(max_block_num_), max_batch_size(max_batch_size_),
        max_thread_num(max_thread_num_) {}
};

class KVCache {
public:
  explicit KVCache(const KVCacheConfig &config);
  ~KVCache();

  // Disable copy
  KVCache(const KVCache &) = delete;
  KVCache &operator=(const KVCache &) = delete;

  // Allow move
  KVCache(KVCache &&) noexcept;
  KVCache &operator=(KVCache &&) noexcept;

  // Core operations
  bool load_kvcache(const std::string &tensor_file_path);
  bool dump_kvcache(torch::Tensor &block_table, int cache_total_len,
                    const std::string &tensor_file_path);
  void update_cache_total_len(int cache_total_len);
  int get_cache_total_len() const;

  // Attention operations
  bool attn(torch::Tensor &q_in, torch::Tensor &output, torch::Tensor &attn_lse,
            int layer_idx, int generate_token_idx,
            torch::Tensor *block_table = nullptr,
            torch::Tensor *cache_seqlens = nullptr,
            int *pick_block_num = nullptr, int *init_block_num = nullptr,
            int *local_block_num = nullptr);

  // KV Cache operations
  bool update_kvcache_one_block_fp16(torch::Tensor &k_in, torch::Tensor &v_in,
                                     int layer_id, int block_idx);

  bool get_kvcache_one_block_fp16(torch::Tensor &k_in, torch::Tensor &v_in,
                                  int layer_id, int block_idx);

  // Importance operations
  bool update_importance_one_block(torch::Tensor &importance, int layer_id,
                                   int block_idx);

  bool get_importance_one_block(torch::Tensor &importance, int layer_id,
                                int block_idx);

  // Anchor operations
  bool get_anchor_one_block(torch::Tensor &anchor, int layer_id, int block_idx);

  bool update_anchor_one_block(torch::Tensor &anchor, int layer_id,
                               int block_idx);

private:
  class Impl;
  std::unique_ptr<Impl> pimpl;
};

} // namespace kvcache