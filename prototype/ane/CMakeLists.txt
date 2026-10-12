cmake_minimum_required(VERSION 3.20)
project(ane_probe LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum macOS deployment target")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_definitions(ACCELERATE_NEW_LAPACK)

set(ANE_VENDOR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/ane.cpp)
if(NOT EXISTS ${ANE_VENDOR}/core/ane_runtime.cpp)
  message(FATAL_ERROR "vendor/ane.cpp missing; run prototype/ane/vendor.sh first")
endif()

# Only the ANE runtime from upstream. Model/tokenizer sources (and their Rust/network deps) are not built.
add_library(ane_runtime STATIC ${ANE_VENDOR}/core/ane_runtime.cpp compat.cpp)
target_include_directories(ane_runtime PUBLIC ${ANE_VENDOR}/core ${ANE_VENDOR}/include)
target_link_libraries(ane_runtime PUBLIC "-framework Foundation" "-framework IOSurface" "-framework Accelerate" objc)
target_compile_options(ane_runtime PRIVATE -O2 -Wno-deprecated-declarations)

add_executable(probe probe.cpp)
target_link_libraries(probe PRIVATE ane_runtime)
target_compile_options(probe PRIVATE -O2)

add_library(lowerings STATIC lowerings.cpp)
target_link_libraries(lowerings PUBLIC ane_runtime)
target_compile_options(lowerings PRIVATE -O2)
add_executable(bench_main bench_main.cpp)
target_link_libraries(bench_main PRIVATE lowerings)
target_compile_options(bench_main PRIVATE -O2)

add_executable(ffn_main ffn_main.cpp)
target_link_libraries(ffn_main PRIVATE lowerings)
target_compile_options(ffn_main PRIVATE -O2)

add_executable(mil_main mil_main.cpp)
target_link_libraries(mil_main PRIVATE ane_runtime)
target_compile_options(mil_main PRIVATE -O2)
