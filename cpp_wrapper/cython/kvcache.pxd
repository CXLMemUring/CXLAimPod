# distutils: language = c++
# cython: language_level = 3

from libcpp.memory cimport unique_ptr
from libc.stdint cimport int64_t
import torch
cimport torch

from kvcache cimport KVCache, KVCacheConfig, AnchorType, GGMLType, RetrievalType

cdef extern from "kvcache.hpp" namespace "kvcache":
    cdef cppclass KVCache:
        KVCache(const KVCacheConfig& config) except +
        bint load_kvcache(const string& tensor_file_path) except +
        bint dump_kvcache(torch::Tensor& block_table, int cache_total_len, const string& tensor_file_path) except +
        void update_cache_total_len(int cache_total_len) except +
        int get_cache_total_len() except +
        bint attn(torch::Tensor& q_in, torch::Tensor& output, torch::Tensor& attn_lse,
                 int layer_idx, int generate_token_idx,
                 torch::Tensor* block_table, torch::Tensor* cache_seqlens,
                 int* pick_block_num, int* init_block_num, int* local_block_num) except +
        bint update_kvcache_one_block_fp16(torch::Tensor& k_in, torch::Tensor& v_in,
                                         int layer_id, int block_idx) except +
        bint get_kvcache_one_block_fp16(torch::Tensor& k_in, torch::Tensor& v_in,
                                      int layer_id, int block_idx) except +
        bint update_importance_one_block(torch::Tensor& importance,
                                       int layer_id, int block_idx) except +
        bint get_importance_one_block(torch::Tensor& importance,
                                    int layer_id, int block_idx) except +
        bint get_anchor_one_block(torch::Tensor& anchor,
                                int layer_id, int block_idx) except +
        bint update_anchor_one_block(torch::Tensor& anchor,
                                   int layer_id, int block_idx) except +

cdef class PyKVCache:
    cdef unique_ptr[KVCache] c_kvcache

    def __cinit__(self, int layer_num, int kv_head_num, int q_head_num, int head_dim,
                  int block_len, int anchor_num, str anchor_type, str kv_type,
                  str retrieval_type, int layer_step, int token_step, int layer_offset,
                  int max_block_num, int max_batch_size, int max_thread_num):
        cdef KVCacheConfig config
        config.layer_num = layer_num
        config.kv_head_num = kv_head_num
        config.q_head_num = q_head_num
        config.head_dim = head_dim
        config.block_len = block_len
        config.anchor_num = anchor_type
        config.anchor_type = self._convert_anchor_type(anchor_type)
        config.kv_type = self._convert_ggml_type(kv_type)
        config.retrieval_type = self._convert_retrieval_type(retrieval_type)
        config.layer_step = layer_step
        config.token_step = token_step
        config.layer_offset = layer_offset
        config.max_block_num = max_block_num
        config.max_batch_size = max_batch_size
        config.max_thread_num = max_thread_num

        self.c_kvcache = unique_ptr[KVCache](new KVCache(config))

    cdef AnchorType _convert_anchor_type(self, str anchor_type):
        if anchor_type == "FIXED":
            return AnchorType.FIXED
        elif anchor_type == "QUEST":
            return AnchorType.QUEST
        elif anchor_type == "DYNAMIC":
            return AnchorType.DYNAMIC
        elif anchor_type == "BLOCK_MEAN":
            return AnchorType.BLOCK_MEAN
        elif anchor_type == "BLOCK_MAX":
            return AnchorType.BLOCK_MAX
        else:
            raise ValueError(f"Unknown anchor type: {anchor_type}")

    cdef GGMLType _convert_ggml_type(self, str kv_type):
        if kv_type == "FP16":
            return GGMLType.FP16
        elif kv_type == "FP32":
            return GGMLType.FP32
        elif kv_type == "Q4_0":
            return GGMLType.Q4_0
        elif kv_type == "Q8_0":
            return GGMLType.Q8_0
        else:
            raise ValueError(f"Unknown kv type: {kv_type}")

    cdef RetrievalType _convert_retrieval_type(self, str retrieval_type):
        if retrieval_type == "SHARED":
            return RetrievalType.LAYER
        elif retrieval_type == "INDIVIDUAL":
            return RetrievalType.QHEAD
        elif retrieval_type == "SEPARATE":
            return RetrievalType.KVHEAD
        else:
            raise ValueError(f"Unknown retrieval type: {retrieval_type}")

    def load_kvcache(self, str tensor_file_path):
        return self.c_kvcache.get().load_kvcache(tensor_file_path.encode('utf-8'))

    def dump_kvcache(self, torch.Tensor block_table, int cache_total_len, str tensor_file_path):
        return self.c_kvcache.get().dump_kvcache(block_table, cache_total_len, tensor_file_path.encode('utf-8'))

    def update_cache_total_len(self, int cache_total_len):
        self.c_kvcache.get().update_cache_total_len(cache_total_len)

    def get_cache_total_len(self):
        return self.c_kvcache.get().get_cache_total_len()

    def attn(self, torch.Tensor q_in, torch.Tensor output, torch.Tensor attn_lse,
             int layer_idx, int generate_token_idx,
             torch.Tensor block_table=None, torch.Tensor cache_seqlens=None,
             int pick_block_num=None, int init_block_num=None, int local_block_num=None):
        cdef torch.Tensor* block_table_ptr = &block_table if block_table is not None else NULL
        cdef torch.Tensor* cache_seqlens_ptr = &cache_seqlens if cache_seqlens is not None else NULL
        cdef int* pick_block_num_ptr = &pick_block_num if pick_block_num is not None else NULL
        cdef int* init_block_num_ptr = &init_block_num if init_block_num is not None else NULL
        cdef int* local_block_num_ptr = &local_block_num if local_block_num is not None else NULL

        return self.c_kvcache.get().attn(q_in, output, attn_lse, layer_idx, generate_token_idx,
                                       block_table_ptr, cache_seqlens_ptr,
                                       pick_block_num_ptr, init_block_num_ptr, local_block_num_ptr)

    def update_kvcache_one_block_fp16(self, torch.Tensor k_in, torch.Tensor v_in,
                                    int layer_id, int block_idx):
        return self.c_kvcache.get().update_kvcache_one_block_fp16(k_in, v_in, layer_id, block_idx)

    def get_kvcache_one_block_fp16(self, torch.Tensor k_in, torch.Tensor v_in,
                                 int layer_id, int block_idx):
        return self.c_kvcache.get().get_kvcache_one_block_fp16(k_in, v_in, layer_id, block_idx)

    def update_importance_one_block(self, torch.Tensor importance,
                                  int layer_id, int block_idx):
        return self.c_kvcache.get().update_importance_one_block(importance, layer_id, block_idx)

    def get_importance_one_block(self, torch.Tensor importance,
                               int layer_id, int block_idx):
        return self.c_kvcache.get().get_importance_one_block(importance, layer_id, block_idx)

    def get_anchor_one_block(self, torch.Tensor anchor,
                           int layer_id, int block_idx):
        return self.c_kvcache.get().get_anchor_one_block(anchor, layer_id, block_idx)

    def update_anchor_one_block(self, torch.Tensor anchor,
                              int layer_id, int block_idx):
        return self.c_kvcache.get().update_anchor_one_block(anchor, layer_id, block_idx) 