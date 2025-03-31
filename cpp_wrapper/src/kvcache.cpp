#include "kvcache.hpp"
#include <Python.h>
#include <iostream>
#include <stdexcept>
#include <torch/torch.h>

// Import the cpuinfer_ext module
extern "C" {
PyObject *PyInit_cpuinfer_ext(void);
}

namespace kvcache {

class KVCache::Impl {
public:
  explicit Impl(const KVCacheConfig &config) {
    // Initialize Python if not already initialized
    if (!Py_IsInitialized()) {
      Py_Initialize();
    }

    // Import cpuinfer_ext module
    PyObject *module = PyImport_ImportModule("cpuinfer_ext");
    if (!module) {
      PyErr_Print(); // Print Python error details
      throw std::runtime_error("Failed to import cpuinfer_ext module");
    }

    // Get the kvcache submodule
    PyObject *kvcache_module = PyObject_GetAttrString(module, "kvcache");
    if (!kvcache_module) {
      Py_DECREF(module);
      PyErr_Print(); // Print Python error details
      throw std::runtime_error("Failed to get kvcache submodule");
    }

    // Get the enum classes
    PyObject *anchor_type_enum =
        PyObject_GetAttrString(kvcache_module, "AnchorType");
    PyObject *ggml_type_enum =
        PyObject_GetAttrString(kvcache_module, "ggml_type");
    PyObject *retrieval_type_enum =
        PyObject_GetAttrString(kvcache_module, "RetrievalType");

    if (!anchor_type_enum || !ggml_type_enum || !retrieval_type_enum) {
      Py_XDECREF(anchor_type_enum);
      Py_XDECREF(ggml_type_enum);
      Py_XDECREF(retrieval_type_enum);
      Py_DECREF(kvcache_module);
      Py_DECREF(module);
      PyErr_Print(); // Print Python error details
      throw std::runtime_error("Failed to get enum classes");
    }

    // Get the enum values
    PyObject *fixed_anchor = PyObject_GetAttrString(anchor_type_enum, "FIXED");
    PyObject *fp16_type = PyObject_GetAttrString(ggml_type_enum, "FP16");
    PyObject *layer_retrieval =
        PyObject_GetAttrString(retrieval_type_enum, "LAYER");

    if (!fixed_anchor || !fp16_type || !layer_retrieval) {
      Py_XDECREF(fixed_anchor);
      Py_XDECREF(fp16_type);
      Py_XDECREF(layer_retrieval);
      Py_DECREF(anchor_type_enum);
      Py_DECREF(ggml_type_enum);
      Py_DECREF(retrieval_type_enum);
      Py_DECREF(kvcache_module);
      Py_DECREF(module);
      PyErr_Print(); // Print Python error details
      throw std::runtime_error("Failed to get enum values");
    }

    // Get the KVCacheConfig class
    PyObject *config_class =
        PyObject_GetAttrString(kvcache_module, "KVCacheConfig");
    if (!config_class) {
      Py_DECREF(fixed_anchor);
      Py_DECREF(fp16_type);
      Py_DECREF(layer_retrieval);
      Py_DECREF(anchor_type_enum);
      Py_DECREF(ggml_type_enum);
      Py_DECREF(retrieval_type_enum);
      Py_DECREF(kvcache_module);
      Py_DECREF(module);
      PyErr_Print(); // Print Python error details
      throw std::runtime_error("Failed to get KVCacheConfig class");
    }

    // Create KVCacheConfig object
    PyObject *config_args = PyTuple_New(15);
    if (!config_args) {
      Py_DECREF(config_class);
      Py_DECREF(fixed_anchor);
      Py_DECREF(fp16_type);
      Py_DECREF(layer_retrieval);
      Py_DECREF(anchor_type_enum);
      Py_DECREF(ggml_type_enum);
      Py_DECREF(retrieval_type_enum);
      Py_DECREF(kvcache_module);
      Py_DECREF(module);
      PyErr_Print(); // Print Python error details
      throw std::runtime_error("Failed to create config arguments tuple");
    }

    // Set config parameters in the tuple
    PyTuple_SetItem(config_args, 0, PyLong_FromLong(config.layer_num));
    PyTuple_SetItem(config_args, 1, PyLong_FromLong(config.kv_head_num));
    PyTuple_SetItem(config_args, 2, PyLong_FromLong(config.q_head_num));
    PyTuple_SetItem(config_args, 3, PyLong_FromLong(config.head_dim));
    PyTuple_SetItem(config_args, 4, PyLong_FromLong(config.block_len));
    PyTuple_SetItem(config_args, 5, PyLong_FromLong(config.anchor_num));
    PyTuple_SetItem(config_args, 6, fixed_anchor);
    PyTuple_SetItem(config_args, 7, fp16_type);
    PyTuple_SetItem(config_args, 8, layer_retrieval);
    PyTuple_SetItem(config_args, 9, PyLong_FromLong(config.layer_step));
    PyTuple_SetItem(config_args, 10, PyLong_FromLong(config.token_step));
    PyTuple_SetItem(config_args, 11, PyLong_FromLong(config.layer_offset));
    PyTuple_SetItem(config_args, 12, PyLong_FromLong(config.max_block_num));
    PyTuple_SetItem(config_args, 13, PyLong_FromLong(config.max_batch_size));
    PyTuple_SetItem(config_args, 14, PyLong_FromLong(config.max_thread_num));

    // Create KVCacheConfig instance
    PyObject *config_obj = PyObject_Call(config_class, config_args, nullptr);
    if (!config_obj) {
      Py_DECREF(config_args);
      Py_DECREF(config_class);
      Py_DECREF(fixed_anchor);
      Py_DECREF(fp16_type);
      Py_DECREF(layer_retrieval);
      Py_DECREF(anchor_type_enum);
      Py_DECREF(ggml_type_enum);
      Py_DECREF(retrieval_type_enum);
      Py_DECREF(kvcache_module);
      Py_DECREF(module);
      PyErr_Print(); // Print Python error details
      throw std::runtime_error("Failed to create KVCacheConfig instance");
    }

    // Get the KVCache class
    PyObject *kvcache_class = PyObject_GetAttrString(kvcache_module, "KVCache");
    if (!kvcache_class) {
      Py_DECREF(config_obj);
      Py_DECREF(config_args);
      Py_DECREF(config_class);
      Py_DECREF(fixed_anchor);
      Py_DECREF(fp16_type);
      Py_DECREF(layer_retrieval);
      Py_DECREF(anchor_type_enum);
      Py_DECREF(ggml_type_enum);
      Py_DECREF(retrieval_type_enum);
      Py_DECREF(kvcache_module);
      Py_DECREF(module);
      PyErr_Print(); // Print Python error details
      throw std::runtime_error("Failed to get KVCache class");
    }

    // Create KVCache instance with the config object
    PyObject *kvcache_args = PyTuple_New(1);
    if (!kvcache_args) {
      Py_DECREF(kvcache_class);
      Py_DECREF(config_obj);
      Py_DECREF(config_args);
      Py_DECREF(config_class);
      Py_DECREF(fixed_anchor);
      Py_DECREF(fp16_type);
      Py_DECREF(layer_retrieval);
      Py_DECREF(anchor_type_enum);
      Py_DECREF(ggml_type_enum);
      Py_DECREF(retrieval_type_enum);
      Py_DECREF(kvcache_module);
      Py_DECREF(module);
      PyErr_Print(); // Print Python error details
      throw std::runtime_error("Failed to create KVCache arguments tuple");
    }

    PyTuple_SetItem(kvcache_args, 0, config_obj);

    kvcache_obj = PyObject_Call(kvcache_class, kvcache_args, nullptr);
    if (!kvcache_obj) {
      Py_DECREF(kvcache_args);
      Py_DECREF(kvcache_class);
      Py_DECREF(config_args);
      Py_DECREF(config_class);
      Py_DECREF(fixed_anchor);
      Py_DECREF(fp16_type);
      Py_DECREF(layer_retrieval);
      Py_DECREF(anchor_type_enum);
      Py_DECREF(ggml_type_enum);
      Py_DECREF(retrieval_type_enum);
      Py_DECREF(kvcache_module);
      Py_DECREF(module);
      PyErr_Print(); // Print Python error details
      throw std::runtime_error("Failed to create KVCache instance");
    }

    // Cleanup
    Py_DECREF(kvcache_args);
    Py_DECREF(kvcache_class);
    Py_DECREF(config_args);
    Py_DECREF(config_class);
    Py_DECREF(fixed_anchor);
    Py_DECREF(fp16_type);
    Py_DECREF(layer_retrieval);
    Py_DECREF(anchor_type_enum);
    Py_DECREF(ggml_type_enum);
    Py_DECREF(retrieval_type_enum);
    Py_DECREF(kvcache_module);
    Py_DECREF(module);
  }

  ~Impl() {
    if (kvcache_obj) {
      Py_DECREF(kvcache_obj);
    }
  }

  bool load_kvcache(const std::string &tensor_file_path) {
    PyObject *result = PyObject_CallMethod(kvcache_obj, "load_kvcache", "s",
                                           tensor_file_path.c_str());
    if (!result) {
      return false;
    }
    bool success = PyObject_IsTrue(result);
    Py_DECREF(result);
    return success;
  }

  bool dump_kvcache(torch::Tensor &block_table, int cache_total_len,
                    const std::string &tensor_file_path) {
    PyObject *block_table_obj = THPVariable_Wrap(block_table);
    PyObject *result =
        PyObject_CallMethod(kvcache_obj, "dump_kvcache", "Ois", block_table_obj,
                            cache_total_len, tensor_file_path.c_str());
    Py_DECREF(block_table_obj);
    if (!result) {
      return false;
    }
    bool success = PyObject_IsTrue(result);
    Py_DECREF(result);
    return success;
  }

  void update_cache_total_len(int cache_total_len) {
    PyObject *result = PyObject_CallMethod(
        kvcache_obj, "update_cache_total_len", "i", cache_total_len);
    if (result) {
      Py_DECREF(result);
    }
  }

  int get_cache_total_len() const {
    PyObject *result =
        PyObject_CallMethod(kvcache_obj, "get_cache_total_len", NULL);
    if (!result) {
      return 0;
    }
    int length = PyLong_AsLong(result);
    Py_DECREF(result);
    return length;
  }

  bool attn(torch::Tensor &q_in, torch::Tensor &output, torch::Tensor &attn_lse,
            int layer_idx, int generate_token_idx, torch::Tensor *block_table,
            torch::Tensor *cache_seqlens, int *pick_block_num,
            int *init_block_num, int *local_block_num) {
    PyObject *q_in_obj = THPVariable_Wrap(q_in);
    PyObject *output_obj = THPVariable_Wrap(output);
    PyObject *attn_lse_obj = THPVariable_Wrap(attn_lse);
    PyObject *block_table_obj =
        block_table ? THPVariable_Wrap(*block_table) : Py_None;
    PyObject *cache_seqlens_obj =
        cache_seqlens ? THPVariable_Wrap(*cache_seqlens) : Py_None;

    PyObject *result = PyObject_CallMethod(
        kvcache_obj, "attn", "OOOiiOOiii", q_in_obj, output_obj, attn_lse_obj,
        layer_idx, generate_token_idx, block_table_obj, cache_seqlens_obj,
        pick_block_num ? *pick_block_num : -1,
        init_block_num ? *init_block_num : -1,
        local_block_num ? *local_block_num : -1);

    Py_DECREF(q_in_obj);
    Py_DECREF(output_obj);
    Py_DECREF(attn_lse_obj);
    if (block_table_obj != Py_None)
      Py_DECREF(block_table_obj);
    if (cache_seqlens_obj != Py_None)
      Py_DECREF(cache_seqlens_obj);

    if (!result) {
      return false;
    }
    bool success = PyObject_IsTrue(result);
    Py_DECREF(result);
    return success;
  }

  bool update_kvcache_one_block_fp16(torch::Tensor &k_in, torch::Tensor &v_in,
                                     int layer_id, int block_idx) {
    PyObject *k_in_obj = THPVariable_Wrap(k_in);
    PyObject *v_in_obj = THPVariable_Wrap(v_in);
    PyObject *result =
        PyObject_CallMethod(kvcache_obj, "update_kvcache_one_block_fp16",
                            "OOii", k_in_obj, v_in_obj, layer_id, block_idx);
    Py_DECREF(k_in_obj);
    Py_DECREF(v_in_obj);
    if (!result) {
      return false;
    }
    bool success = PyObject_IsTrue(result);
    Py_DECREF(result);
    return success;
  }

  bool get_kvcache_one_block_fp16(torch::Tensor &k_in, torch::Tensor &v_in,
                                  int layer_id, int block_idx) {
    PyObject *k_in_obj = THPVariable_Wrap(k_in);
    PyObject *v_in_obj = THPVariable_Wrap(v_in);
    PyObject *result =
        PyObject_CallMethod(kvcache_obj, "get_kvcache_one_block_fp16", "OOii",
                            k_in_obj, v_in_obj, layer_id, block_idx);
    Py_DECREF(k_in_obj);
    Py_DECREF(v_in_obj);
    if (!result) {
      return false;
    }
    bool success = PyObject_IsTrue(result);
    Py_DECREF(result);
    return success;
  }

  bool update_importance_one_block(torch::Tensor &importance, int layer_id,
                                   int block_idx) {
    PyObject *importance_obj = THPVariable_Wrap(importance);
    PyObject *result =
        PyObject_CallMethod(kvcache_obj, "update_importance_one_block", "Oii",
                            importance_obj, layer_id, block_idx);
    Py_DECREF(importance_obj);
    if (!result) {
      return false;
    }
    bool success = PyObject_IsTrue(result);
    Py_DECREF(result);
    return success;
  }

  bool get_importance_one_block(torch::Tensor &importance, int layer_id,
                                int block_idx) {
    PyObject *importance_obj = THPVariable_Wrap(importance);
    PyObject *result =
        PyObject_CallMethod(kvcache_obj, "get_importance_one_block", "Oii",
                            importance_obj, layer_id, block_idx);
    Py_DECREF(importance_obj);
    if (!result) {
      return false;
    }
    bool success = PyObject_IsTrue(result);
    Py_DECREF(result);
    return success;
  }

  bool get_anchor_one_block(torch::Tensor &anchor, int layer_id,
                            int block_idx) {
    PyObject *anchor_obj = THPVariable_Wrap(anchor);
    PyObject *result =
        PyObject_CallMethod(kvcache_obj, "get_anchor_one_block", "Oii",
                            anchor_obj, layer_id, block_idx);
    Py_DECREF(anchor_obj);
    if (!result) {
      return false;
    }
    bool success = PyObject_IsTrue(result);
    Py_DECREF(result);
    return success;
  }

  bool update_anchor_one_block(torch::Tensor &anchor, int layer_id,
                               int block_idx) {
    PyObject *anchor_obj = THPVariable_Wrap(anchor);
    PyObject *result =
        PyObject_CallMethod(kvcache_obj, "update_anchor_one_block", "Oii",
                            anchor_obj, layer_id, block_idx);
    Py_DECREF(anchor_obj);
    if (!result) {
      return false;
    }
    bool success = PyObject_IsTrue(result);
    Py_DECREF(result);
    return success;
  }

private:
  PyObject *kvcache_obj;
};

KVCache::KVCache(const KVCacheConfig &config)
    : pimpl(std::make_unique<Impl>(config)) {}
KVCache::~KVCache() = default;

KVCache::KVCache(KVCache &&) noexcept = default;
KVCache &KVCache::operator=(KVCache &&) noexcept = default;

bool KVCache::load_kvcache(const std::string &tensor_file_path) {
  return pimpl->load_kvcache(tensor_file_path);
}

bool KVCache::dump_kvcache(torch::Tensor &block_table, int cache_total_len,
                           const std::string &tensor_file_path) {
  return pimpl->dump_kvcache(block_table, cache_total_len, tensor_file_path);
}

void KVCache::update_cache_total_len(int cache_total_len) {
  pimpl->update_cache_total_len(cache_total_len);
}

int KVCache::get_cache_total_len() const {
  return pimpl->get_cache_total_len();
}

bool KVCache::attn(torch::Tensor &q_in, torch::Tensor &output,
                   torch::Tensor &attn_lse, int layer_idx,
                   int generate_token_idx, torch::Tensor *block_table,
                   torch::Tensor *cache_seqlens, int *pick_block_num,
                   int *init_block_num, int *local_block_num) {
  return pimpl->attn(q_in, output, attn_lse, layer_idx, generate_token_idx,
                     block_table, cache_seqlens, pick_block_num, init_block_num,
                     local_block_num);
}

bool KVCache::update_kvcache_one_block_fp16(torch::Tensor &k_in,
                                            torch::Tensor &v_in, int layer_id,
                                            int block_idx) {
  return pimpl->update_kvcache_one_block_fp16(k_in, v_in, layer_id, block_idx);
}

bool KVCache::get_kvcache_one_block_fp16(torch::Tensor &k_in,
                                         torch::Tensor &v_in, int layer_id,
                                         int block_idx) {
  return pimpl->get_kvcache_one_block_fp16(k_in, v_in, layer_id, block_idx);
}

bool KVCache::update_importance_one_block(torch::Tensor &importance,
                                          int layer_id, int block_idx) {
  return pimpl->update_importance_one_block(importance, layer_id, block_idx);
}

bool KVCache::get_importance_one_block(torch::Tensor &importance, int layer_id,
                                       int block_idx) {
  return pimpl->get_importance_one_block(importance, layer_id, block_idx);
}

bool KVCache::get_anchor_one_block(torch::Tensor &anchor, int layer_id,
                                   int block_idx) {
  return pimpl->get_anchor_one_block(anchor, layer_id, block_idx);
}

bool KVCache::update_anchor_one_block(torch::Tensor &anchor, int layer_id,
                                      int block_idx) {
  return pimpl->update_anchor_one_block(anchor, layer_id, block_idx);
}

} // namespace kvcache