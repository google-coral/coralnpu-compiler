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

get_filename_component(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
list(APPEND CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/iree/integrations/pjrt/cmake"
)
include(protobuf_cc_library)

add_subdirectory("../third_party/iree/integrations/pjrt/third_party/pjrt_c_api" pjrt_c_api)
add_subdirectory("../third_party/iree/integrations/pjrt/src" integrations_pjrt)

iree_cc_library(
  NAME
    client
  HDRS
    "client.h"
  SRCS
    "client.cc"
  DEPS
    coralnpu_runtime::driver::coralnpu
    iree_pjrt::common
    iree::base
    iree::hal
    iree::hal::drivers::local_sync::sync_driver
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
if(DEFINED IREE_PJRT_PYTHON_BINARY_DIR)
  set(_NATIVE_PYTHON_DIR
      "${IREE_PJRT_PYTHON_BINARY_DIR}/iree/_pjrt_libs/coralnpu")
  file(MAKE_DIRECTORY "${_NATIVE_PYTHON_DIR}")
  file(WRITE "${_NATIVE_PYTHON_DIR}/__init__.py" "")
  set_target_properties(
    pjrt_plugin_dylib
    PROPERTIES PREFIX "" # Disable "lib" prefix.
               LIBRARY_OUTPUT_NAME iree_pjrt_coralnpu
               RUNTIME_OUTPUT_DIRECTORY "${_NATIVE_PYTHON_DIR}"
               LIBRARY_OUTPUT_DIRECTORY "${_NATIVE_PYTHON_DIR}")
endif()

if(NOT IREE_ENABLE_ASAN)
  if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    target_link_options(pjrt_plugin_dylib PRIVATE "-Wl,-undefined,error")
  else()
    target_link_options(pjrt_plugin_dylib PRIVATE "-fuse-ld=lld" "-Wl,--no-undefined")
  endif()
endif()
