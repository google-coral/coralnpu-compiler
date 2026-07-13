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

  set(CORALNPU_SIMULATOR_TYPE "mpact" CACHE STRING "Simulator backend: mpact or verilator")

  set(SIMULATOR_LIB "${CMAKE_CURRENT_LIST_DIR}/libcoralnpu_simulator_mpact.so")

  if(CORALNPU_SIMULATOR_TYPE STREQUAL "verilator")
    set(CORALNPU_VERILATOR_SIMULATOR_LIB "" CACHE FILEPATH "Path to libcoralnpu_simulator.so for verilator")
    if(NOT CORALNPU_VERILATOR_SIMULATOR_LIB OR NOT EXISTS "${CORALNPU_VERILATOR_SIMULATOR_LIB}")
      set(DEFAULT_VERILATOR_SO "${CMAKE_SOURCE_DIR}/bazel-bin/external/coralnpu_hw+/hw_sim/libcoralnpu_simulator.so")
      if(NOT EXISTS "${DEFAULT_VERILATOR_SO}")
        set(DEFAULT_VERILATOR_SO "${CMAKE_SOURCE_DIR}/bazel-bin/external/coralnpu_hw/hw_sim/libcoralnpu_simulator.so")
      endif()
      if(NOT EXISTS "${DEFAULT_VERILATOR_SO}")
        message(STATUS "Verilator simulator library not found. Building with Bazel...")
        execute_process(
          COMMAND sh -c "cd ${CMAKE_SOURCE_DIR} && export PWD=${CMAKE_SOURCE_DIR} && USE_BAZEL_VERSION=8.6.0 bazel build --config=dev @coralnpu_hw//hw_sim:libcoralnpu_simulator.so"
          RESULT_VARIABLE BAZEL_BUILD_RES
        )
        if(NOT BAZEL_BUILD_RES EQUAL 0)
          message(FATAL_ERROR "Failed to build Verilator simulator library via Bazel.")
        endif()
      endif()
      set(CORALNPU_VERILATOR_SIMULATOR_LIB "${DEFAULT_VERILATOR_SO}")
    endif()

    set(VERILATOR_CMAKE_LIB "${CMAKE_CURRENT_BINARY_DIR}/libcoralnpu_simulator.so")
    configure_file("${CORALNPU_VERILATOR_SIMULATOR_LIB}" "${VERILATOR_CMAKE_LIB}" COPYONLY)

    set(SIM_DEFINES "USE_VERILATOR")
    set(SIM_DEPS "${VERILATOR_CMAKE_LIB}")
    set(SIM_INCLUDES "${CMAKE_SOURCE_DIR}" "${CMAKE_SOURCE_DIR}/third_party/coralnpu")
  else()
    set(SIM_DEFINES "USE_MPACT_PREBUILT")
    set(SIM_DEPS "${SIMULATOR_LIB}")
    set(SIM_INCLUDES "${CMAKE_SOURCE_DIR}")
  endif()

  iree_cc_library(
    NAME
      mpact_simulator
    HDRS
      "simulator_api.h"
    SRCS
      "simulator.cpp"
    DEFINES
      ${SIM_DEFINES}
    INCLUDES
      ${SIM_INCLUDES}
    DEPS
      ${SIM_DEPS}
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
      "${CMAKE_SOURCE_DIR}"
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
