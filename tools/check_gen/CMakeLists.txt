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

# Special case package namespace for tools/ -> "" (empty string)
get_filename_component(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

iree_cc_binary(
  NAME
    check_gen
  SRCS
    "CheckTestGenerator.cpp"
    "CheckTestGenerator.h"
    "CompileUtils.cpp"
    "CompileUtils.h"
    "ParseUtils.cpp"
    "ParseUtils.h"
    "PrecompiledBinary.cpp"
    "PrecompiledBinary.h"
    "main.cpp"
  DEPS
    iree::compiler::bindings::c::headers
    iree::compiler::API::StaticImpl
    iree::compiler::ConstEval::Runtime
    iree::compiler::Tools::init_passes_and_dialects
    LLVMSupport
    MLIRCAPIIR
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRTensorDialect
    MLIRTransforms
    ChloOps
    StablehloOps
    StablehloPasses
)

add_subdirectory(generators)
