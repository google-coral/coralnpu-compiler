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

set(STABLEHLO_MANUAL_PATTERNS
    "case_"
    "compare_eq_"
    "concatenate_rank[234]_"
    "is_finite_"
    "pad_rank2_"
    "reverse_rank2_i32_(120_256|300_450)"
    "scatter_"
    "slice_rank[234]_"
)

foreach(TYPE f32 i16 i32 i8)
  coralnpu_add_static_check_tests(
    TYPE ${TYPE}
    COMPILER_FLAGS ${_COMMON_COMPILER_FLAGS}
    RUNNER_ARGS ${_COMMON_RUNNER_ARGS}
    MANUAL_PATTERNS ${STABLEHLO_MANUAL_PATTERNS}
  )
endforeach()

coralnpu_check_test(
  NAME
    matmul_highmem_test
  SRC
    "matmul_highmem.mlir"
  COMPILER_FLAGS
    "--iree-hal-target-device=coralnpu"
    "--coralnpu-dtcm-size-kb=1024"
    "--coralnpu-linker-script-path=${CMAKE_BINARY_DIR}/crt/coralnpu_tcm_highmem.ld"
  RUNNER_ARGS
    "--device=coralnpu"
  LABELS
    "ci"
)

coralnpu_check_generated_sync_test()

