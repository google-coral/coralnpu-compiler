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

include(CMakeParseArguments)

set(_COMMON_COMPILER_FLAGS
    "--iree-hal-target-device=local"
    "--iree-hal-local-target-device-backends=vmvx"
    "--iree-hal-target-device=coralnpu"
    "--coralnpu-target-abi=ilp32"
    "--coralnpu-target-cpu-features=+m,+f,+zvl128b,+zve32f"
)

set(_COMMON_RUNNER_ARGS
    "--device=local-sync"
    "--device=coralnpu"
)

# op_tests()
function(op_tests)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;TEST;TIMEOUT;DEFAULT_GEN"
    "INSTANCES;ARG_GENS;COMPILER_FLAGS;RUNNER_ARGS"
    ${ARGN}
  )

  if(NOT DEFINED _RULE_TIMEOUT)
    set(_RULE_TIMEOUT "short")
  endif()

  set(COMPILER_FLAGS ${_COMMON_COMPILER_FLAGS})
  if(DEFINED _RULE_COMPILER_FLAGS)
    list(APPEND COMPILER_FLAGS ${_RULE_COMPILER_FLAGS})
  endif()

  set(RUNNER_ARGS ${_COMMON_RUNNER_ARGS})
  if(DEFINED _RULE_RUNNER_ARGS)
    list(APPEND RUNNER_ARGS ${_RULE_RUNNER_ARGS})
  endif()

  coralnpu_check_gen_tests(
    NAME
      "${_RULE_NAME}"
    TEST
      "${_RULE_TEST}"
    INSTANCES
      ${_RULE_INSTANCES}
    ARG_GENS
      ${_RULE_ARG_GENS}
    DEFAULT_GEN
      "${_RULE_DEFAULT_GEN}"
    COMPILER_FLAGS
      ${COMPILER_FLAGS}
    RUNNER_ARGS
      ${RUNNER_ARGS}
    TIMEOUT
      "${_RULE_TIMEOUT}"
  )
endfunction()
