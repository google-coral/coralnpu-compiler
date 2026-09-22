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

if(CORALNPU_ENABLE_VERILATOR)
  foreach(_VERILATOR_TEST
      "abs_f32_4_8.mlir"
      "add_i32_4_8-4_8.mlir"
      "matmul_f32_4_8-8_4.mlir"
      "matmul_i8_4_8-8_4.mlir")
    get_filename_component(_BASE_NAME "${_VERILATOR_TEST}" NAME_WE)
    coralnpu_check_test(
      NAME
        "${_BASE_NAME}_verilator_test"
      SRC
        "${_VERILATOR_TEST}"
      COMPILER_FLAGS
        "--iree-hal-target-device=coralnpu"
      RUNNER_ARGS
        "--device=coralnpu"
        "--simulator=verilator"
      LABELS
        "ci"
    )
  endforeach()
endif()
