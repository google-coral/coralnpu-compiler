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

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")

  set(SIMULATOR_LIB "${CMAKE_CURRENT_LIST_DIR}/libcoralnpu_simulator_mpact.so")

  iree_cc_library(
    NAME
      mpact_simulator
    HDRS
      "simulator_api.h"
    SRCS
      "simulator_mpact.cpp"
    INCLUDES
      "${PROJECT_SOURCE_DIR}"
    DEPS
      ${SIMULATOR_LIB}
    PUBLIC
  )

  iree_cc_library(
    NAME
      simulation
    HDRS
      "simulator_api.h"
      "simulator_elf_loader.h"
      "simulator_format.h"
      "simulator_inline.h"
      "simulator_executable.h"
    SRCS
      "simulator_elf_loader.c"
      "simulator_inline.c"
      "simulator_executable.c"
    INCLUDES
      "${PROJECT_SOURCE_DIR}"
    DEPS
      coralnpu_runtime::sim::mpact_simulator
      iree::base
      iree::base::internal
      iree::hal::local::executable_environment
      iree::hal::local::executable_library
      iree::hal::local::executable_loader
    PUBLIC
  )

endif()
