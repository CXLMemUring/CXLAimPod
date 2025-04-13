cmake_minimum_required(VERSION 3.10)
project(cpuinfer_ext)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable testing
enable_testing()

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0 -g")
# Get PyTorch paths using Python
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.__path__[0])"
    OUTPUT_VARIABLE TORCH_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get PyTorch include paths
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; from torch.utils.cpp_extension import include_paths; print(';'.join(include_paths()))"
    OUTPUT_VARIABLE TORCH_INCLUDE_PATHS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get PyTorch library paths
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; from torch.utils.cpp_extension import library_paths; print(';'.join(library_paths()))"
    OUTPUT_VARIABLE TORCH_LIBRARY_PATHS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get PyTorch CXX flags
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(' '.join(torch._C._get_cmake_cxx_flags()))"
    OUTPUT_VARIABLE TORCH_CXX_FLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Find CUDA
find_package(CUDA REQUIRED)

# Add include directories
include_directories(
    ${Python3_INCLUDE_DIRS}
    ${TORCH_PATH}/include
    ${TORCH_PATH}/include/torch/csrc/api/include
    ${TORCH_PATH}/include/TH
    ${TORCH_PATH}/include/THC
    ${TORCH_INCLUDE_PATHS}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Add library directories
link_directories(
    ${TORCH_LIBRARY_PATHS}
)

# Add source files
set(SOURCES
    src/kvcache.cpp
    test/test_kvcache.cpp
)

# Create library
add_library(cpuinfer_ext SHARED ${SOURCES})

# Set compile flags
separate_arguments(TORCH_CXX_FLAGS)
target_compile_options(cpuinfer_ext PRIVATE ${TORCH_CXX_FLAGS})

# Link libraries
target_link_libraries(cpuinfer_ext
    PRIVATE
    ${Python3_LIBRARIES}
    ${CUDA_LIBRARIES}
    torch
    torch_cpu
    torch_python
    c10
    c10_cuda
    cudart
)

# Set include directories for the target
target_include_directories(cpuinfer_ext
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Python3_INCLUDE_DIRS}
    ${TORCH_PATH}/include
    ${TORCH_PATH}/include/torch/csrc/api/include
    ${TORCH_PATH}/include/TH
    ${TORCH_PATH}/include/THC
    ${TORCH_INCLUDE_PATHS}
    ${CUDA_INCLUDE_DIRS}
)

# Set output directories
set_target_properties(cpuinfer_ext PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
)

# Add test executable
add_executable(test_kvcache test/test_kvcache.cpp)

# Link test executable
target_link_libraries(test_kvcache
    PRIVATE
    cpuinfer_ext
    ${Python3_LIBRARIES}
    ${CUDA_LIBRARIES}
    torch
    torch_cpu
    torch_python
    c10
    c10_cuda
    cudart
)

target_include_directories(test_kvcache
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_PATHS}
    ${CUDA_INCLUDE_DIRS}
)

# Add test executable
add_executable(test_deepseek_moe test/test_deepseek_moe.cpp)

# Link test executable
target_link_libraries(test_deepseek_moe
        PRIVATE
        cpuinfer_ext
        ${Python3_LIBRARIES}
        ${CUDA_LIBRARIES}
        torch
        torch_cpu
        torch_python
        c10
        c10_cuda
        cudart
)

target_include_directories(test_deepseek_moe
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${Python3_INCLUDE_DIRS}
        ${TORCH_INCLUDE_PATHS}
        ${CUDA_INCLUDE_DIRS}
)

# Set compile options
if(MSVC)
    target_compile_options(cpuinfer_ext PRIVATE /W4)
else()
    target_compile_options(cpuinfer_ext PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Add test
add_test(NAME test_kvcache COMMAND test_kvcache)

# Install targets
install(TARGETS cpuinfer_ext
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp") 