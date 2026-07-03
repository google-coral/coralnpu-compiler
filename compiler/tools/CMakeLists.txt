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
    coralnpu_compile_lib
  HDRS
    "coralnpu_compile_lib.h"
  SRCS
    "coralnpu_compile_lib.cc"
  INCLUDES
    "${CMAKE_SOURCE_DIR}"
  DEPS
    LLVMSupport
    MLIRIR
    MLIRPass
    MLIRCAPIIR
    iree::compiler::API::Internal::CompilerDriver
    iree::compiler::Pipelines
    iree::compiler::bindings::c::headers
  PUBLIC
)

iree_cc_binary(
  NAME
    coralnpu-compile
  SRCS
    "coralnpu-compile-main.cc"
  DEPS
    ::coralnpu_compile_lib
    iree::compiler::API::StaticImpl
    iree::compiler::bindings::c::headers
  HOSTONLY
  SETUP_INSTALL_RPATH
)

if(CORALNPU_LINKER_PATH AND EXISTS "${CORALNPU_LINKER_PATH}")
  add_custom_command(
    TARGET coralnpu_compiler_tools_coralnpu-compile POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            "${CORALNPU_LINKER_PATH}"
            "$<TARGET_FILE_DIR:coralnpu_compiler_tools_coralnpu-compile>/"
  )
endif()
