cmake_minimum_required(VERSION 3.20.0)
project(cuda_club VERSION 0.1.0 LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_subdirectory(reduce)
