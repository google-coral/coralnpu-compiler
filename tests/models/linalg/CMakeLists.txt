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
    "softmax_"
    "select_"
    "generic_multi_output_"
    "1_6_6_4-3_3_4"
    "pack_padding_.*_7_15"
    "pack_padding_.*_17_33"
    "pooling_nchw_.*_1_8_120_256-5_7"
    "1_4_6_6-4_4_3_3"
)

foreach(TYPE f32 i16 i32 i8)
  coralnpu_add_static_check_tests(
    TYPE ${TYPE}
    COMPILER_FLAGS ${_COMMON_COMPILER_FLAGS}
    RUNNER_ARGS ${_COMMON_RUNNER_ARGS}
    MANUAL_PATTERNS ${LINALG_MANUAL_PATTERNS}
  )
endforeach()
