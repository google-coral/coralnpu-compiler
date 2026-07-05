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
include(${CMAKE_CURRENT_LIST_DIR}/../tools/check_gen/def.cmake)

# Standard default generators matching tests/defs.bzl
set(CORALNPU_STANDARD_DEFAULT_GEN "tools_check_gen_generators_sequential_vmfb")

# coralnpu_check_gen_tests()
function(coralnpu_check_gen_tests)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;TEST;TIMEOUT;DEFAULT_GEN"
    "INSTANCES;ARG_GENS;COMPILER_FLAGS;RUNNER_ARGS;LABELS"
    ${ARGN}
  )

  if("manual" IN_LIST _RULE_LABELS)
    message(STATUS "Skipping manual check gen test: ${_RULE_NAME}")
    return()
  endif()


  # 1. Resolve DEFAULT_GEN to path and target
  if(NOT DEFINED _RULE_DEFAULT_GEN)
    set(_RULE_DEFAULT_GEN ${CORALNPU_STANDARD_DEFAULT_GEN})
  endif()

  set(DEFAULT_GEN_FILE)
  set(DEFAULT_GEN_TARGET)
  if(_RULE_DEFAULT_GEN)
    if(TARGET ${_RULE_DEFAULT_GEN})
      set(DEFAULT_GEN_TARGET ${_RULE_DEFAULT_GEN})
      if(${_RULE_DEFAULT_GEN} MATCHES "^tools_check_gen_generators_")
        string(REPLACE "tools_check_gen_generators_" "" GEN_NAME "${_RULE_DEFAULT_GEN}")
        set(DEFAULT_GEN_FILE "${CMAKE_BINARY_DIR}/tools/check_gen/generators/${GEN_NAME}.vmfb")
      else()
        message(FATAL_ERROR "Target ${_RULE_DEFAULT_GEN} is not a supported generator target. Pass file path instead.")
      endif()
    else()
      get_filename_component(DEFAULT_GEN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${_RULE_DEFAULT_GEN}" REALPATH)
    endif()
  endif()

  # 2. Resolve ARG_GENS to paths and targets
  set(GEN_FILES)
  set(GEN_TARGETS)
  foreach(GEN IN LISTS _RULE_ARG_GENS)
    if(GEN STREQUAL "default")
      list(APPEND GEN_FILES "default")
    else()
      if(TARGET ${GEN})
        list(APPEND GEN_TARGETS ${GEN})
        if(GEN MATCHES "^tools_check_gen_generators_")
          string(REPLACE "tools_check_gen_generators_" "" GEN_NAME "${GEN}")
          list(APPEND GEN_FILES "${CMAKE_BINARY_DIR}/tools/check_gen/generators/${GEN_NAME}.vmfb")
        else()
          message(FATAL_ERROR "Target ${GEN} is not a supported generator target. Pass file path instead.")
        endif()
      else()
        get_filename_component(GEN_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${GEN}" REALPATH)
        list(APPEND GEN_FILES "${GEN_PATH}")
      endif()
    endif()
  endforeach()


  # 4. Call generic check_gen_tests
  check_gen_tests(
    NAME
      ${_RULE_NAME}
    TEST
      ${_RULE_TEST}
    DEFAULT_GEN
      ${DEFAULT_GEN_FILE}
    DEFAULT_GEN_TARGET
      ${DEFAULT_GEN_TARGET}
    ARG_GENS
      ${GEN_FILES}
    ARG_GEN_TARGETS
      ${GEN_TARGETS}
    INSTANCES
      ${_RULE_INSTANCES}
    COMPILER_FLAGS
      ${_RULE_COMPILER_FLAGS}
    RUNNER_ARGS
      ${_RULE_RUNNER_ARGS}
    LABELS
      "driver=coralnpu"
      "target=coralnpu"
      ${_RULE_LABELS}
    TIMEOUT
      ${_RULE_TIMEOUT}
  )
endfunction()
