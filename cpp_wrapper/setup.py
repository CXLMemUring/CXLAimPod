from setuptools import setup, Extension
from Cython.Build import cythonize
import torch
import os
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def get_extensions():
    extensions = [
        Extension(
            "kvcache_cpp.kvcache",
            ["cython/kvcache.pyx"],
            include_dirs=[
                os.path.dirname(os.path.abspath(__file__)),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "include"),
                torch.utils.cpp_extension.CUDA_HOME,
            ],
            library_dirs=[
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "build"),
            ],
            libraries=["kvcache_cpp"],
            language="c++",
            extra_compile_args={
                "cxx": ["-std=c++17", "-O3"],
                "nvcc": ["-std=c++17", "-O3"],
            },
        ),
        Extension(
            "kvcache_cpp.cpuinfer",
            ["cython/cpuinfer.pyx"],
            include_dirs=[
                os.path.dirname(os.path.abspath(__file__)),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "include"),
                torch.utils.cpp_extension.CUDA_HOME,
            ],
            library_dirs=[
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "build"),
            ],
            libraries=["kvcache_cpp"],
            language="c++",
            extra_compile_args={
                "cxx": ["-std=c++17", "-O3"],
                "nvcc": ["-std=c++17", "-O3"],
            },
        ),
    ]
    return extensions

setup(
    name="kvcache_cpp",
    ext_modules=cythonize(get_extensions()),
    zip_safe=False,
)

# Get PyTorch include paths
torch_include_paths = torch.utils.cpp_extension.include_paths()
torch_library_paths = torch.utils.cpp_extension.library_paths()

setup(
    name='cpuinfer_ext',
    ext_modules=[
        CUDAExtension('cpuinfer_ext', [
            'src/kvcache.cpp',
            'test/test_kvcache.cpp'
        ],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++17'],
            'nvcc': ['-O3', '-std=c++17']
        },
        include_dirs=[
            'include',
            *torch_include_paths,
            os.path.join(torch.utils.cpp_extension.CUDA_HOME, 'include')
        ],
        library_dirs=[
            'lib',
            *torch_library_paths,
            os.path.join(torch.utils.cpp_extension.CUDA_HOME, 'lib64')
        ],
        libraries=[
            'torch',
            'torch_cpu',
            'torch_python',
            'c10',
            'c10_cuda',
            'caffe2_module_test_dynamic',
            'caffe2_observers',
            'caffe2_detectron_ops_gpu',
            'caffe2_detectron_ops',
            'caffe2_serialize',
            'caffe2_utils',
            'caffe2_protos',
            'caffe2_core',
            'caffe2_operators',
            'caffe2_common',
            'cudart'
        ],
        define_macros=[('WITH_CUDA', None)],
        extra_link_args=['-L/usr/local/cuda/lib64'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }) 