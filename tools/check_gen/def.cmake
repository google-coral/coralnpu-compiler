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

# Helper function to parse instance to suffix
function(parse_instance_to_suffix INSTANCE_STR OUT_VAR)
  string(FIND "${INSTANCE_STR}" ", [" TAG_POS)
  if(NOT TAG_POS EQUAL -1)
    string(SUBSTRING "${INSTANCE_STR}" 0 ${TAG_POS} INST_CLI)
  else()
    set(INST_CLI "${INSTANCE_STR}")
  endif()
  string(REPLACE ")(" "-" TEMP "${INST_CLI}")
  string(REPLACE "(" "" TEMP "${TEMP}")
  string(REPLACE ")" "" TEMP "${TEMP}")
  string(REPLACE "," "_" TEMP "${TEMP}")
  set(${OUT_VAR} "${TEMP}" PARENT_SCOPE)
endfunction()

# check_gen_tests()
function(check_gen_tests)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;TEST;TIMEOUT;DEFAULT_GEN;DEFAULT_GEN_TARGET"
    "INSTANCES;ARG_GENS;ARG_GEN_TARGETS;COMPILER_FLAGS;RUNNER_ARGS;LABELS;DEPENDS"
    ${ARGN}
  )

  # Declare outputs
  get_filename_component(TEST_FILE_BASE "${_RULE_TEST}" NAME_WE)
  set(OUTPUT_FILES)
  set(OUTPUT_SUFFIXES)
  foreach(INST IN LISTS _RULE_INSTANCES)
    parse_instance_to_suffix("${INST}" SUFFIX)
    list(APPEND OUTPUT_SUFFIXES "${SUFFIX}")
    list(APPEND OUTPUT_FILES "${CMAKE_CURRENT_BINARY_DIR}/${TEST_FILE_BASE}_${SUFFIX}_check.mlir")
  endforeach()

  # Construct check_gen command
  set(CHECK_GEN_ARGS)
  list(APPEND CHECK_GEN_ARGS "-o" "${CMAKE_CURRENT_BINARY_DIR}")

  if(_RULE_DEFAULT_GEN)
    list(APPEND CHECK_GEN_ARGS "--default-gen" "${_RULE_DEFAULT_GEN}")
  endif()

  foreach(INST IN LISTS _RULE_INSTANCES)
    string(FIND "${INST}" ", [" TAG_POS)
    if(NOT TAG_POS EQUAL -1)
      string(SUBSTRING "${INST}" 0 ${TAG_POS} INST_CLI)
    else()
      set(INST_CLI "${INST}")
    endif()
    list(APPEND CHECK_GEN_ARGS "--instance" "${INST_CLI}")
  endforeach()

  get_filename_component(TEST_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${_RULE_TEST}" REALPATH)
  list(APPEND CHECK_GEN_ARGS "${TEST_PATH}")
  list(APPEND CHECK_GEN_ARGS ${_RULE_ARG_GENS})

  # Run check_gen at build time
  add_custom_command(
    OUTPUT
      ${OUTPUT_FILES}
    COMMAND
      tools_check_gen_check_gen
      ${CHECK_GEN_ARGS}
    DEPENDS
      tools_check_gen_check_gen
      ${TEST_PATH}
      ${_RULE_DEFAULT_GEN_TARGET}
      ${_RULE_ARG_GEN_TARGETS}
      ${_RULE_DEPENDS}
    COMMENT
      "Generating check tests from template ${_RULE_TEST}"
    VERBATIM
  )

  set(GEN_TARGET "generate_${_RULE_NAME}")
  add_custom_target(${GEN_TARGET} DEPENDS ${OUTPUT_FILES})

  list(LENGTH _RULE_INSTANCES INST_COUNT)
  math(EXPR LAST_IDX "${INST_COUNT} - 1")

  foreach(IDX RANGE 0 ${LAST_IDX})
    list(GET OUTPUT_FILES ${IDX} OUT_FILE)
    list(GET OUTPUT_SUFFIXES ${IDX} SUFFIX)
    list(GET _RULE_INSTANCES ${IDX} INST_ENTRY)

    set(EXTRA_LABELS)
    string(FIND "${INST_ENTRY}" ", [" TAG_POS)
    if(NOT TAG_POS EQUAL -1)
      math(EXPR TAG_START "${TAG_POS} + 4")
      string(LENGTH "${INST_ENTRY}" INST_LEN)
      math(EXPR TAG_LEN "${INST_LEN} - ${TAG_START} - 1")
      string(SUBSTRING "${INST_ENTRY}" ${TAG_START} ${TAG_LEN} TAG_STR)
      string(REPLACE "," ";" EXTRA_LABELS "${TAG_STR}")
    endif()

    set(TEST_NAME "${_RULE_NAME}_${SUFFIX}")

    iree_check_test(
      NAME
        "${TEST_NAME}"
      SRC
        "${OUT_FILE}"
      TARGET_BACKEND
        "vmvx"
      DRIVER
        "local-sync"
      COMPILER_FLAGS
        ${_RULE_COMPILER_FLAGS}
      RUNNER_ARGS
        ${_RULE_RUNNER_ARGS}
      LABELS
        ${_RULE_LABELS}
        ${EXTRA_LABELS}
      DEPENDS
        ${GEN_TARGET}
      TIMEOUT
        "${_RULE_TIMEOUT}"
    )
  endforeach()
endfunction()
