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

# coralnpu_add_static_check_tests()
function(coralnpu_add_static_check_tests)
  cmake_parse_arguments(
    _RULE
    ""
    "TYPE"
    "COMPILER_FLAGS;RUNNER_ARGS;MANUAL_PATTERNS"
    ${ARGN}
  )

  file(GLOB_RECURSE CHECK_FILES "${CMAKE_CURRENT_SOURCE_DIR}/generated_${_RULE_TYPE}/*_check.mlir")
  list(SORT CHECK_FILES)

  foreach(CHECK_FILE IN LISTS CHECK_FILES)
    get_filename_component(FILE_NAME "${CHECK_FILE}" NAME_WE)
    string(REGEX REPLACE "_check$" "" TEST_NAME "${FILE_NAME}")

    set(TEST_LABELS "driver=coralnpu" "target=coralnpu" "${_RULE_TYPE}")
    set(IS_MANUAL FALSE)
    foreach(PATTERN IN LISTS _RULE_MANUAL_PATTERNS)
      if(TEST_NAME MATCHES "${PATTERN}")
        set(IS_MANUAL TRUE)
        break()
      endif()
    endforeach()

    if(IS_MANUAL)
      list(APPEND TEST_LABELS "manual")
    else()
      list(APPEND TEST_LABELS "ci")
    endif()

    coralnpu_check_test(
      NAME
        "${TEST_NAME}"
      SRC
        "${CHECK_FILE}"
      COMPILER_FLAGS
        "--mlir-disable-threading"
        ${_RULE_COMPILER_FLAGS}
      RUNNER_ARGS
        ${_RULE_RUNNER_ARGS}
      LABELS
        ${TEST_LABELS}
      TIMEOUT
        "short"
    )
  endforeach()
endfunction()

# coralnpu_check_generated_sync_test()
function(coralnpu_check_generated_sync_test)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "LABELS"
    ${ARGN}
  )

  if(NOT _RULE_NAME)
    set(_RULE_NAME "check_generated")
  endif()

  iree_package_path(_PACKAGE_PATH)
  set(_TEST_NAME "${_PACKAGE_PATH}/${_RULE_NAME}")

  add_test(
    NAME
      "${_TEST_NAME}"
    COMMAND
      "${CMAKE_COMMAND}" -E echo "All generated check test files for ${_PACKAGE_PATH} are in sync."
  )

  set_property(TEST "${_TEST_NAME}" PROPERTY LABELS "driver=coralnpu" "target=coralnpu" "ci" ${_RULE_LABELS} "${_PACKAGE_PATH}")
  set_property(TEST "${_TEST_NAME}" PROPERTY TIMEOUT 60)
endfunction()


