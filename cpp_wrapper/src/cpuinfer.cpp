#include "cpuinfer.hpp"
#include <Python.h>
#include <stdexcept>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/python_engine.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/cuda/Stream.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/torch.h>

namespace kvcache {

class CPUInfer::Impl {
public:
  explicit Impl(int thread_num) {
    // Initialize Python if not already initialized
    if (!Py_IsInitialized()) {
      Py_Initialize();
    }

    // Import cpuinfer_ext module
    PyObject *module = PyImport_ImportModule("cpuinfer_ext");
    if (!module) {
      throw std::runtime_error("Failed to import cpuinfer_ext module");
    }

    // Get the CPUInfer class
    PyObject *cpuinfer_class = PyObject_GetAttrString(module, "CPUInfer");
    if (!cpuinfer_class) {
      Py_DECREF(module);
      throw std::runtime_error("Failed to get CPUInfer class");
    }

    // Create CPUInfer instance
    PyObject *args = PyTuple_Pack(1, PyLong_FromLong(thread_num));
    cpuinfer_obj = PyObject_Call(cpuinfer_class, args, nullptr);
    Py_DECREF(args);
    Py_DECREF(cpuinfer_class);
    Py_DECREF(module);

    if (!cpuinfer_obj) {
      throw std::runtime_error("Failed to create CPUInfer instance");
    }
  }

  ~Impl() {
    if (cpuinfer_obj) {
      Py_DECREF(cpuinfer_obj);
    }
  }

  void submit(void *task) {
    PyObject *result = PyObject_CallMethod(cpuinfer_obj, "submit", "O",
                                           PyLong_FromVoidPtr(task));
    if (!result) {
      throw std::runtime_error("Failed to submit task");
    }
    Py_DECREF(result);
  }

  void submit_with_cuda_stream(cudaStream_t stream, void *task) {
    // Convert CUDA stream to PyObject
    PyObject *stream_capsule = PyCapsule_New(stream, "cudaStream_t", nullptr);
    if (!stream_capsule) {
      throw std::runtime_error("Failed to wrap CUDA stream");
    }

    PyObject *result =
        PyObject_CallMethod(cpuinfer_obj, "submit_with_cuda_stream", "OO",
                            stream_capsule, PyLong_FromVoidPtr(task));
    Py_DECREF(stream_capsule);

    if (!result) {
      throw std::runtime_error("Failed to submit task with CUDA stream");
    }
    Py_DECREF(result);
  }

  void sync() {
    PyObject *result = PyObject_CallMethod(cpuinfer_obj, "sync", nullptr);
    if (!result) {
      throw std::runtime_error("Failed to sync");
    }
    Py_DECREF(result);
  }

  void sync_with_cuda_stream(cudaStream_t stream) {
    // Convert CUDA stream to PyObject
    PyObject *stream_capsule = PyCapsule_New(stream, "cudaStream_t", nullptr);
    if (!stream_capsule) {
      throw std::runtime_error("Failed to wrap CUDA stream");
    }

    PyObject *result = PyObject_CallMethod(
        cpuinfer_obj, "sync_with_cuda_stream", "O", stream_capsule);
    Py_DECREF(stream_capsule);

    if (!result) {
      throw std::runtime_error("Failed to sync with CUDA stream");
    }
    Py_DECREF(result);
  }

private:
  PyObject *cpuinfer_obj;
};

CPUInfer::CPUInfer(int thread_num)
    : pimpl(std::make_unique<Impl>(thread_num)) {}

CPUInfer::~CPUInfer() = default;

CPUInfer::CPUInfer(CPUInfer &&) noexcept = default;
CPUInfer &CPUInfer::operator=(CPUInfer &&) noexcept = default;

void CPUInfer::submit(void *task) { pimpl->submit(task); }

void CPUInfer::submit_with_cuda_stream(cudaStream_t stream, void *task) {
  pimpl->submit_with_cuda_stream(stream, task);
}

void CPUInfer::sync() { pimpl->sync(); }

void CPUInfer::sync_with_cuda_stream(cudaStream_t stream) {
  pimpl->sync_with_cuda_stream(stream);
}

} // namespace kvcache