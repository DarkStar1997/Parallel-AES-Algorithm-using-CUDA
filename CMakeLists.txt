cmake_minimum_required(VERSION 3.15)

project(CUDA-AES)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

find_program(CCACHE_PROGRAM ccache)

if(CCACHE_PROGRAM)
    MESSAGE("CCACHE support enabled")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CCACHE_PROGRAM}")
endif()

set(CMAKE_BUILD_TYPE "RELEASE")

function (add_code name)
    add_executable(rt_${name} ${name}.cpp)
endfunction()

function (add_cuda_code name)
    add_executable(rt_${name} ${name}.cu)
endfunction()

add_cuda_code(AES)
