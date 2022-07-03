cmake_minimum_required(VERSION 3.22)
project(arnoldi)
set(CMAKE_CXX_STANDARD 14)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -g")

find_package(OpenMP)

add_executable(arnoldi main.cpp)
target_include_directories(arnoldi PRIVATE OpenBLAS/include)
target_link_directories(arnoldi PRIVATE OpenBLAS/lib)
target_link_libraries(arnoldi PRIVATE openblas OpenMP::OpenMP_CXX)