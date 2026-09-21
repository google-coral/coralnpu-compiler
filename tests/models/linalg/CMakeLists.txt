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

include(defs.cmake)

set(LINALG_MANUAL_PATTERNS
    "pack_padding_.*_(7_15|17_33)"
    # BF16 manual instances (due to large accumulation divergence or ITCM size)
    "matmul_bf16_(120_256-256_300|300_100-100_450)"
    "batch_matmul_bf16_3_12_25-3_25_30"
    "batch_reduce_matmul_bf16_3_12_25-3_25_30"
    "mmt4d_bf16_"
    "batch_mmt4d_bf16_"
    "generic_reduction_2d_bf16_(120_256|300_450)"
    "reduce_2d_bf16_(120_256|300_450)"
    "reduce_3d_dim1_2_bf16_(10_20_30|5_100_2)"
    "contract_matmul_bf16_(120_256-256_300|300_100-100_450)"
    "matvec_bf16_(120_256-256|300_450-450)"
    "vecmat_bf16_(256-256_120|450-450_300)"
    "batch_matvec_bf16_3_120_256-3_256"
    "batch_vecmat_bf16_3_256-3_256_120"
    "dot_bf16_(256-256|450-450)"
)

foreach(TYPE f32 bf16 i16 i32 i8)
  coralnpu_add_static_check_tests(
    TYPE ${TYPE}
    COMPILER_FLAGS ${_COMMON_COMPILER_FLAGS}
    RUNNER_ARGS ${_COMMON_RUNNER_ARGS}
    MANUAL_PATTERNS ${LINALG_MANUAL_PATTERNS}
  )
endforeach()

if(CORALNPU_ENABLE_VERILATOR)
  foreach(_VERILATOR_TEST
      "generated_f32/abs_f32_4_8_check.mlir"
      "generated_i32/add_i32_4_8-4_8_check.mlir"
      "generated_f32/matmul_f32_4_8-8_4_check.mlir"
      "generated_i8/matmul_i8_4_8-8_4_check.mlir")
    get_filename_component(_BASE_NAME "${_VERILATOR_TEST}" NAME_WE)
    string(REGEX REPLACE "_check$" "" _BASE_NAME "${_BASE_NAME}")
    coralnpu_check_test(
      NAME
        "${_BASE_NAME}_verilator_test"
      SRC
        "${_VERILATOR_TEST}"
      COMPILER_FLAGS
        ${_COMMON_COMPILER_FLAGS}
      RUNNER_ARGS
        "--device=coralnpu"
        "--simulator=verilator"
      LABELS
        "ci"
    )
  endforeach()
endif()

coralnpu_check_generated_sync_test()

