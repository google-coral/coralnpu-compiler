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

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "affinity_annotation.mlir"
    "approximate_math.mlir"
    "convolution_to_igemm.mlir"
    "dump_affinity_execution_profile.mlir"
    "dump_affinity_execution_profile_empty.mlir"
    "matrix_codegen.mlir"
    "optimize_vector_loops.mlir"
    "promote_bf16_accumulator.mlir"
    "raise_apply_scale.mlir"
    "register_allocation_report.mlir"
    "restore_conv_input_channel.mlir"
    "target_options.mlir"
    "tile_size_selection_dtcm.mlir"
    "tile_size_selection_footprint.mlir"
  TOOLS
    FileCheck
    iree-compile
    iree-opt
  LABELS
    "ci"
)
