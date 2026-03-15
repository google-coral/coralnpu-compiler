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
# disable the default runtime HAL drivers
set(IREE_HAL_DRIVER_DEFAULTS OFF)
# enable our hal driver in the runtime
set(IREE_EXTERNAL_HAL_DRIVERS coralnpu)
set(IREE_BUILD_TESTS OFF)
set(IREE_BUILD_SAMPLES OFF)

set(IREE_COMPILER_PLUGINS
    "coralnpu"
    CACHE STRING "Statically linked IREE compiler plugins" FORCE)

add_subdirectory(third_party/iree)
