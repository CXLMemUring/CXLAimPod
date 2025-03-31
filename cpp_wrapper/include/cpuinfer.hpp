#pragma once

#include <cuda_runtime.h>
#include <memory>

namespace kvcache {

class CPUInfer {
public:
  explicit CPUInfer(int thread_num);
  ~CPUInfer();

  // Disable copy
  CPUInfer(const CPUInfer &) = delete;
  CPUInfer &operator=(const CPUInfer &) = delete;

  // Allow move
  CPUInfer(CPUInfer &&) noexcept;
  CPUInfer &operator=(CPUInfer &&) noexcept;

  // Core operations
  void submit(void *task);
  void submit_with_cuda_stream(cudaStream_t stream, void *task);
  void sync();
  void sync_with_cuda_stream(cudaStream_t stream);

private:
  class Impl;
  std::unique_ptr<Impl> pimpl;
};

} // namespace kvcache