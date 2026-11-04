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


# Compiler metrics of the CoralNPU target backend: a metrics bundle per
# executable, which the hooks of the backend
# (coralnpu_compiler::Target::TargetHooks) fill as it compiles the executable.
# Depends on these hooks, not on the backend.

iree_setup_c_src_root(
  PACKAGE_ROOT_PREFIX
    coralnpu_metrics_compiler
)

# The --coralnpu-dump-metrics-* options.
iree_cc_library(
  NAME
    MetricsOptions
  HDRS
    "MetricsOptions.h"
  INCLUDES
    "${CMAKE_SOURCE_DIR}"
  DEPS
    iree::compiler::Utils
    LLVMSupport
  PUBLIC
)

# Size report of the linked firmware ELF.
iree_cc_library(
  NAME
    MetricsElf
  HDRS
    "MetricsElf.h"
  SRCS
    "MetricsElf.cpp"
  INCLUDES
    "${CMAKE_SOURCE_DIR}"
  DEPS
    LLVMBinaryFormat
    LLVMObject
    LLVMSupport
  PUBLIC
)

# The metrics bundle of one executable and its JSON and text writers.
iree_cc_library(
  NAME
    MetricsBundle
  HDRS
    "MetricsBundle.h"
  SRCS
    "MetricsBundle.cpp"
  INCLUDES
    "${CMAKE_SOURCE_DIR}"
  DEPS
    ::MetricsElf
    LLVMSupport
  PUBLIC
)

# The `coralnpu.metrics` attribute of the dispatch functions.
iree_cc_library(
  NAME
    MetricsAttr
  HDRS
    "MetricsAttr.h"
  SRCS
    "MetricsAttr.cpp"
  INCLUDES
    "${CMAKE_SOURCE_DIR}"
  DEPS
    LLVMSupport
    MLIRIR
  PUBLIC
)

# Records the tile grid of each dispatch after the configuration pipeline.
iree_cc_library(
  NAME
    RecordTileGridPass
  HDRS
    "RecordTileGridPass.h"
  SRCS
    "RecordTileGridPass.cpp"
  INCLUDES
    "${CMAKE_SOURCE_DIR}"
  DEPS
    ::MetricsAttr
    coralnpu_compiler::Transforms
    iree::compiler::Codegen::Dialect::CPU::IR::IREECPUDialect
    iree::compiler::Codegen::Dialect::Codegen::IR::IREECodegenDialect
    iree::compiler::Codegen::Utils
    LLVMSupport
    MLIRFunctionInterfaces
    MLIRIR
    MLIRLinalgDialect
    MLIRPass
    MLIRTilingInterface
  PUBLIC
)

# Records whether the matrix codegen lowered each dispatch to Zvt.
iree_cc_library(
  NAME
    RecordZvtLoweringPass
  HDRS
    "RecordZvtLoweringPass.h"
  SRCS
    "RecordZvtLoweringPass.cpp"
  INCLUDES
    "${CMAKE_SOURCE_DIR}"
  DEPS
    ::MetricsAttr
    coralnpu_compiler::Target::Utils
    LLVMSupport
    MLIRFunctionInterfaces
    MLIRIR
    MLIRLLVMDialect
    MLIRPass
  PUBLIC
)

# The hooks that add the metrics passes and fill and write the metrics
# bundles.
iree_cc_library(
  NAME
    MetricsTargetHooks
  HDRS
    "MetricsTargetHooks.h"
  SRCS
    "MetricsTargetHooks.cpp"
  INCLUDES
    "${CMAKE_SOURCE_DIR}"
  DEPS
    ::MetricsAttr
    ::MetricsBundle
    ::MetricsElf
    ::MetricsOptions
    ::RecordTileGridPass
    ::RecordZvtLoweringPass
    coralnpu_compiler::Target::TargetHooks
    coralnpu_compiler::Target::Utils
    iree::compiler::plugins::target::LLVMCPU::LLVMTargetOptions
    iree::compiler::Dialect::HAL::IR
    LLVMSupport
    MLIRIR
    MLIRPass
  PUBLIC
)
