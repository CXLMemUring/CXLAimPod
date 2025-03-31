# distutils: language = c++
# cython: language_level = 3

from libcpp.memory cimport unique_ptr
from libc.stdint cimport int64_t
import torch
cimport torch

from cpuinfer cimport CPUInfer

cdef extern from "cpuinfer.hpp" namespace "kvcache":
    cdef cppclass CPUInfer:
        CPUInfer(int thread_num) except +
        void submit(void* task) except +
        void submit_with_cuda_stream(cudaStream_t stream, void* task) except +
        void sync() except +
        void sync_with_cuda_stream(cudaStream_t stream) except +

cdef class PyCPUInfer:
    cdef unique_ptr[CPUInfer] c_cpuinfer

    def __cinit__(self, int thread_num):
        self.c_cpuinfer = unique_ptr[CPUInfer](new CPUInfer(thread_num))

    def submit(self, void* task):
        self.c_cpuinfer.get().submit(task)

    def submit_with_cuda_stream(self, cudaStream_t stream, void* task):
        self.c_cpuinfer.get().submit_with_cuda_stream(stream, task)

    def sync(self):
        self.c_cpuinfer.get().sync()

    def sync_with_cuda_stream(self, cudaStream_t stream):
        self.c_cpuinfer.get().sync_with_cuda_stream(stream) 