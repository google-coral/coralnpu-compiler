# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

cmake_minimum_required(VERSION 3.21)

# Set default compilers
set(CMAKE_C_COMPILER "/usr/bin/clang-19" CACHE PATH "C compiler" FORCE)
set(CMAKE_CXX_COMPILER "/usr/bin/clang++-19" CACHE PATH "C++ compiler" FORCE)

set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type" FORCE)

# Enable ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE PATH "C compiler launcher" FORCE)
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE PATH "C++ compiler launcher" FORCE)
endif()

project(CoralNPU LANGUAGES C CXX)

# we don't checkout all submodules of iree as we don't use them
set(IREE_ERROR_ON_MISSING_SUBMODULES OFF)
# we want to have assertions even if we are building for Release or
# RelWithDebInfo
set(IREE_ENABLE_ASSERTIONS ON)
# add the cmake for our IREE plugin
set(IREE_CMAKE_PLUGIN_PATHS ${CMAKE_CURRENT_LIST_DIR})
# disable IREE compiler backend plugins by default
set(IREE_TARGET_BACKEND_DEFAULTS OFF)
# enable IREE llvm-cpu backed plugin that is used to compile code for CSS
set(IREE_TARGET_BACKEND_LLVM_CPU ON)
set(IREE_TARGET_BACKEND_VMVX ON)
set(IREE_TARGET_BACKEND_CORALNPU ON)
# disable the default runtime HAL drivers
set(IREE_HAL_DRIVER_DEFAULTS OFF)
# Explicitly enable local-sync driver (needed for VMVX tests)
set(IREE_HAL_DRIVER_LOCAL_SYNC ON CACHE BOOL "Enable local-sync driver" FORCE)
# Explicitly disable local-task driver
set(IREE_HAL_DRIVER_LOCAL_TASK OFF CACHE BOOL "Disable local-task driver" FORCE)
# enable our hal driver in the runtime
set(IREE_EXTERNAL_HAL_DRIVERS coralnpu)
if(NOT DEFINED IREE_BUILD_TESTS)
  set(IREE_BUILD_TESTS OFF)
endif()
if(IREE_BUILD_TESTS)
  enable_testing()
endif()
if(NOT DEFINED IREE_BUILD_SAMPLES)
  set(IREE_BUILD_SAMPLES OFF)
endif()

# Enable Split DWARF and Thin Archives for faster development builds
set(IREE_ENABLE_SPLIT_DWARF ON)
set(IREE_ENABLE_THIN_ARCHIVES ON)

set(IREE_COMPILER_PLUGINS
    "coralnpu"
    CACHE STRING "Statically linked IREE compiler plugins" FORCE)

set(IREE_ENABLE_LLD ON CACHE BOOL "Override the system's default linker to lld" FORCE)

add_subdirectory(third_party/iree)

list(APPEND CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_LIST_DIR}/third_party/iree/build_tools/cmake"
)
include(iree_copts)

add_subdirectory(tools)
add_subdirectory(tests)
