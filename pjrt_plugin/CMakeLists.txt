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

iree_cc_library(
  NAME
    client
  HDRS
    "client.h"
  SRCS
    "client.cc"
  DEPS
    runtime::driver::coralnpu
    iree::integrations::pjrt::common
    iree::base
    iree::hal::local::loaders::registration
    iree::hal::local::plugins::registration
  PUBLIC)

iree_cc_library(
  SHARED
  NAME
    dylib
  DEFINES
    # Causes PJRT dynamic linking entry points to be made visible.
    PJRT_PLUGIN_BUILDING_LIBRARY
  SRCS
    "dylib_entry_point.cc"
  DEPS
    ::client
    iree_pjrt::common
    iree_pjrt::common::dylib_platform)

# Output to the project wide python binary directory tree.
set(_NATIVE_PYTHON_DIR
    "${IREE_PJRT_PYTHON_BINARY_DIR}/iree/_pjrt_libs/coralnpu")
file(WRITE "${_NATIVE_PYTHON_DIR}/__init__.py" "")
set_target_properties(
  iree_pjrt_coralnpu_dylib
  PROPERTIES PREFIX "" # Disable "lib" prefix.
             LIBRARY_OUTPUT_NAME iree_pjrt_coralnpu
             RUNTIME_OUTPUT_DIRECTORY "${_NATIVE_PYTHON_DIR}"
             LIBRARY_OUTPUT_DIRECTORY "${_NATIVE_PYTHON_DIR}")

if(NOT IREE_ENABLE_ASAN)
  if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    target_link_options(iree_pjrt_coralnpu_dylib PRIVATE "-Wl,-undefined,error")
  else()
    target_link_options(iree_pjrt_coralnpu_dylib PRIVATE "-Wl,--no-undefined")
  endif()
endif()
