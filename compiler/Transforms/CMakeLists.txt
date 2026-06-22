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

iree_tablegen_library(
  NAME
    PassesIncGen
  TD_FILE
    "Passes.td"
  OUTS
    --gen-pass-decls compiler/Transforms/Passes.h.inc
)

iree_cc_library(
  NAME
    Transforms
  HDRS
    "Passes.h"
  TEXTUAL_HDRS
    "compiler/Transforms/Passes.h.inc"
  SRCS
    "CoralNPUAffinityAnnotation.cpp"
    "CoralNPUDumpAffinityExecutionProfile.cpp"
    "CoralNPUTileSizeSelection.cpp"
    "Passes.cpp"
  INCLUDES
    "${CMAKE_SOURCE_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}"
  DEPS
    ::PassesIncGen
    iree::compiler::Codegen::Dialect::CPU::IR::IREECPUDialect
    iree::compiler::Codegen::Dialect::Codegen::IR::IREECodegenDialect
    iree::compiler::Codegen::Utils
    iree::compiler::Dialect::HAL::Analysis
    iree::compiler::Dialect::HAL::IR
    iree::compiler::Dialect::Stream::IR
    iree::compiler::Dialect::TensorExt::IR
    iree::compiler::Dialect::Util::IR
    LLVMSupport
    MLIRAffineDialect
    MLIRFunctionInterfaces
    MLIRIR
    MLIRLinalgDialect
    MLIRLinalgUtils
    MLIRPass
    MLIRTilingInterface
  PUBLIC
)
