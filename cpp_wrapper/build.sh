#!/bin/bash

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build the project
make -j$(nproc)

# Go back to root directory
cd ..

# Build the Python package
python setup.py build_ext --inplace 