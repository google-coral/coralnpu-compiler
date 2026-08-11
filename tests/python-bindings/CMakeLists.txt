# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if(NOT IREE_BUILD_TESTS OR NOT IREE_BUILD_PYTHON_BINDINGS)
  return()
endif()

include(iree_python)

# Ensure Python modules are built when building test dependencies
add_dependencies(iree-test-deps
  IREECompilerPythonModules
  iree_runtime_bindings_python_runtime
)

function(coralnpu_py_test)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;SRC;TIMEOUT"
    "ARGS;LABELS"
    ${ARGN}
  )

  iree_package_ns(_PACKAGE_NS)
  string(REPLACE "::" "/" _PACKAGE_PATH ${_PACKAGE_NS})
  set(_NAME_PATH "${_PACKAGE_PATH}/${_RULE_NAME}")
  list(APPEND _RULE_LABELS "${_PACKAGE_PATH}")

  set(_PYTHONPATH
    "${CMAKE_BINARY_DIR}/third_party/iree/compiler/bindings/python"
    "${CMAKE_BINARY_DIR}/third_party/iree/runtime/bindings/python"
    "$ENV{PYTHONPATH}"
  )
  if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    list(JOIN _PYTHONPATH "\\;" _PYTHONPATH_JOINED)
  else()
    list(JOIN _PYTHONPATH ":" _PYTHONPATH_JOINED)
  endif()

  set(_LD_LIBRARY_PATH
    "${CMAKE_SOURCE_DIR}/runtime/sim"
    "$ENV{LD_LIBRARY_PATH}"
  )
  if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    list(JOIN _LD_LIBRARY_PATH "\\;" _LD_LIBRARY_PATH_JOINED)
  else()
    list(JOIN _LD_LIBRARY_PATH ":" _LD_LIBRARY_PATH_JOINED)
  endif()

  add_test(
    NAME ${_NAME_PATH}
    COMMAND
      "${Python3_EXECUTABLE}"
      "${_RULE_SRC}"
      ${_RULE_ARGS}
  )

  set_property(TEST ${_NAME_PATH} PROPERTY LABELS "ci" ${_RULE_LABELS})
  set_property(TEST ${_NAME_PATH} PROPERTY ENVIRONMENT
    "PYTHONPATH=${_PYTHONPATH_JOINED}"
    "LD_LIBRARY_PATH=${_LD_LIBRARY_PATH_JOINED}"
  )

  if(NOT DEFINED _RULE_TIMEOUT)
    set(_RULE_TIMEOUT 60)
  endif()
  set_property(TEST ${_NAME_PATH} PROPERTY TIMEOUT ${_RULE_TIMEOUT})

  iree_configure_test(${_NAME_PATH})
endfunction()

coralnpu_py_test(
  NAME
    hal_test
  SRC
    "${CMAKE_SOURCE_DIR}/third_party/iree/runtime/bindings/python/tests/hal_test.py"
  LABELS
    "ci"
)

coralnpu_py_test(
  NAME
    import_test
  SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/import_test.py"
  LABELS
    "ci"
)
