cmake_minimum_required(VERSION 3.10)
project(CXL_Microbenchmarks CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pthread")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Add the double_bandwidth executable
add_executable(double_bandwidth double_bandwidth.cpp)

# Install targets
install(TARGETS double_bandwidth DESTINATION bin)
