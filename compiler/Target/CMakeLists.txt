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
    CoralNPUTarget
  HDRS
    "CoralNPULinkerTool.h"
    "CoralNPUTargetBackend.h"
  SRCS
    "CoralNPULinkerTool.cpp"
    "CoralNPUTarget.cpp"
    "CoralNPUTargetBackend.cpp"
  INCLUDES
    "${CMAKE_SOURCE_DIR}"
  DEPS
    coralnpu_compiler::Transforms
    iree::compiler::plugins::target::LLVMCPU::LLVMCPU
    iree::compiler::plugins::target::LLVMCPU::LLVMIRPasses
    iree::compiler::plugins::target::LLVMCPU::LLVMTargetOptions
    iree::compiler::plugins::target::LLVMCPU::LinkerTool
    iree::compiler::plugins::target::LLVMCPU::StaticLibraryGenerator
    iree::compiler::plugins::target::LLVMCPU::Builtins
    iree::compiler::Codegen::Dialect::CPU::IR::IREECPUDialect
    iree::compiler::Codegen::Dialect::Codegen::IR::IREECodegenDialect
    iree::compiler::Codegen::Dialect::VectorExt::IR::IREEVectorExtDialect
    iree::compiler::Codegen::LLVMCPU
    iree::compiler::Codegen::Utils
    iree::compiler::Dialect::Encoding::IR
    iree::compiler::Dialect::HAL::IR
    iree::compiler::Dialect::HAL::Target
    iree::compiler::Dialect::HAL::Utils::LLVMLinkerUtils
    iree::compiler::Dialect::LinalgExt::IR
    iree::compiler::Dialect::Util::IR
    iree::compiler::PluginAPI
    iree::compiler::Utils
    LLVMBitWriter
    LLVMCore
    LLVMLinker
    LLVMSupport
    LLVMTarget
    LLVMTargetParser
    MLIRBuiltinToLLVMIRTranslation
    MLIRIR
    MLIRLLVMDialect
    MLIRLLVMToLLVMIRTranslation
    MLIRLinalgDialect
    MLIRPDLDialect
    MLIRPDLInterpDialect
    MLIRPass
    MLIRTargetLLVMIRExport
    MLIRTransformDialect
  DEFINES
    "CORALNPU_LINKER_SCRIPT_PATH=\"${CORALNPU_LINKER_SCRIPT_PATH}\""
    "CORALNPU_LINKER_PATH=\"${CORALNPU_LINKER_PATH}\""
  PUBLIC
)

iree_compiler_register_plugin(
  PLUGIN_ID
    coralnpu
  TARGET
    ::CoralNPUTarget
)
