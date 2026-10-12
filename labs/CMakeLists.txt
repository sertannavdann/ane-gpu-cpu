cmake_minimum_required(VERSION 3.20)
project(pseries_labs CXX)

# Use Ninja: the parent checkout's path contains colons, which break Make.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# -mcpu=native turns on everything this core has (NEON fp16, dotprod, i8mm).
# No -ffast-math: the labs depend on the compiler NOT reordering float sums.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mcpu=native")

find_library(ACCELERATE Accelerate REQUIRED)

foreach(lab peak bandwidth latency gemm gemv)
  add_executable(p1_${lab} p1/${lab}.cpp)
  target_link_libraries(p1_${lab} ${ACCELERATE})
endforeach()

add_executable(p2_quant p2/quant.cpp)
target_link_libraries(p2_quant ${ACCELERATE})

add_executable(test_kernels tests/test_kernels.cpp)
target_link_libraries(test_kernels ${ACCELERATE})

enable_testing()
add_test(NAME kernels COMMAND test_kernels)
