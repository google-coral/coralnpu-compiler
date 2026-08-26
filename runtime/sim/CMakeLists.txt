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

  # ==============================================================================
  # CoralNPU Runtime Simulation Backend
  #
  # NOTE: The CoralNPU compiler tools and passes do NOT depend on Bazel or the
  # simulator library.
  #
  # The functional simulator library (MPACT or Verilator) is only needed when
  # building the runtime HAL driver simulation backend (coralnpu_runtime::sim).
  # If a pre-built simulator library is not explicitly supplied via
  # CORALNPU_MPACT_SIMULATOR_LIB or CORALNPU_VERILATOR_SIMULATOR_LIB, CMake
  # invokes Bazel as a build-time fallback to build the simulator from source.
  # ==============================================================================

  set(CORALNPU_SIMULATOR_TYPE "mpact" CACHE STRING "Simulator backend: mpact or verilator")

  if(CORALNPU_SIMULATOR_TYPE STREQUAL "verilator")
    set(CORALNPU_VERILATOR_SIMULATOR_LIB "" CACHE FILEPATH "Path to libcoralnpu_simulator.so for verilator")
    if(NOT CORALNPU_VERILATOR_SIMULATOR_LIB OR NOT EXISTS "${CORALNPU_VERILATOR_SIMULATOR_LIB}")
      set(VERILATOR_CANDIDATES
        "${CMAKE_SOURCE_DIR}/bazel-bin/external/coralnpu_hw/hw_sim/libcoralnpu_simulator.so"
        "${CMAKE_SOURCE_DIR}/bazel-bin/external/coralnpu_hw+/hw_sim/libcoralnpu_simulator.so"
      )
      set(FOUND_VERILATOR_SO "")
      foreach(CANDIDATE IN LISTS VERILATOR_CANDIDATES)
        if(EXISTS "${CANDIDATE}")
          set(FOUND_VERILATOR_SO "${CANDIDATE}")
          break()
        endif()
      endforeach()

      if(NOT FOUND_VERILATOR_SO)
        message(STATUS "[CoralNPU Runtime] Verilator simulator library not found. Building with Bazel (@coralnpu_hw//hw_sim:libcoralnpu_simulator.so)...")
        execute_process(
          COMMAND sh -c "cd ${CMAKE_SOURCE_DIR} && export PWD=${CMAKE_SOURCE_DIR} && USE_BAZEL_VERSION=8.6.0 bazel build --config=dev @coralnpu_hw//hw_sim:libcoralnpu_simulator.so"
          RESULT_VARIABLE BAZEL_BUILD_RES
        )
        if(NOT BAZEL_BUILD_RES EQUAL 0)
          message(FATAL_ERROR "[CoralNPU Runtime] Failed to build Verilator simulator library via Bazel.")
        endif()

        foreach(CANDIDATE IN LISTS VERILATOR_CANDIDATES)
          if(EXISTS "${CANDIDATE}")
            set(FOUND_VERILATOR_SO "${CANDIDATE}")
            break()
          endif()
        endforeach()

        if(NOT FOUND_VERILATOR_SO)
          file(GLOB GLOBBED_VERILATOR_SO "${CMAKE_SOURCE_DIR}/bazel-bin/external/*coralnpu_hw*/hw_sim/libcoralnpu_simulator.so")
          if(GLOBBED_VERILATOR_SO)
            list(GET GLOBBED_VERILATOR_SO 0 FOUND_VERILATOR_SO)
          endif()
        endif()
      endif()

      if(NOT FOUND_VERILATOR_SO OR NOT EXISTS "${FOUND_VERILATOR_SO}")
        message(FATAL_ERROR "[CoralNPU Runtime] Verilator simulator library was built, but could not locate libcoralnpu_simulator.so in bazel-bin.")
      endif()

      set(CORALNPU_VERILATOR_SIMULATOR_LIB "${FOUND_VERILATOR_SO}")
    endif()

    set(VERILATOR_CMAKE_LIB "${CMAKE_CURRENT_BINARY_DIR}/libcoralnpu_simulator.so")
    configure_file("${CORALNPU_VERILATOR_SIMULATOR_LIB}" "${VERILATOR_CMAKE_LIB}" COPYONLY)

    set(SIM_DEPS "${VERILATOR_CMAKE_LIB}")
  else()
    set(CORALNPU_MPACT_SIMULATOR_LIB "" CACHE FILEPATH "Path to libcoralnpu_simulator_mpact.so for mpact")
    if(NOT CORALNPU_MPACT_SIMULATOR_LIB OR NOT EXISTS "${CORALNPU_MPACT_SIMULATOR_LIB}")
      set(MPACT_CANDIDATES
        "${CMAKE_SOURCE_DIR}/bazel-bin/external/coralnpu_mpact+/sim/hw_sim/libcoralnpu_simulator_mpact.so"
        "${CMAKE_SOURCE_DIR}/bazel-bin/external/coralnpu_mpact/sim/hw_sim/libcoralnpu_simulator_mpact.so"
      )
      set(FOUND_MPACT_SO "")
      foreach(CANDIDATE IN LISTS MPACT_CANDIDATES)
        if(EXISTS "${CANDIDATE}")
          set(FOUND_MPACT_SO "${CANDIDATE}")
          break()
        endif()
      endforeach()

      if(NOT FOUND_MPACT_SO)
        message(STATUS "[CoralNPU Runtime] MPACT simulator library not found. Building with Bazel (@coralnpu_mpact//sim/hw_sim:libcoralnpu_simulator_mpact.so)...")
        execute_process(
          COMMAND sh -c "cd ${CMAKE_SOURCE_DIR} && export PWD=${CMAKE_SOURCE_DIR} && USE_BAZEL_VERSION=8.6.0 bazel build --config=dev @coralnpu_mpact//sim/hw_sim:libcoralnpu_simulator_mpact.so"
          RESULT_VARIABLE BAZEL_BUILD_RES
        )
        if(NOT BAZEL_BUILD_RES EQUAL 0)
          message(FATAL_ERROR "[CoralNPU Runtime] Failed to build MPACT simulator library via Bazel.")
        endif()

        foreach(CANDIDATE IN LISTS MPACT_CANDIDATES)
          if(EXISTS "${CANDIDATE}")
            set(FOUND_MPACT_SO "${CANDIDATE}")
            break()
          endif()
        endforeach()

        if(NOT FOUND_MPACT_SO)
          file(GLOB GLOBBED_MPACT_SO "${CMAKE_SOURCE_DIR}/bazel-bin/external/*coralnpu_mpact*/sim/hw_sim/libcoralnpu_simulator_mpact.so")
          if(GLOBBED_MPACT_SO)
            list(GET GLOBBED_MPACT_SO 0 FOUND_MPACT_SO)
          endif()
        endif()
      endif()

      if(NOT FOUND_MPACT_SO OR NOT EXISTS "${FOUND_MPACT_SO}")
        message(FATAL_ERROR "[CoralNPU Runtime] MPACT simulator library was built, but could not locate libcoralnpu_simulator_mpact.so in bazel-bin.")
      endif()

      set(CORALNPU_MPACT_SIMULATOR_LIB "${FOUND_MPACT_SO}")
    endif()

    set(MPACT_CMAKE_LIB "${CMAKE_CURRENT_BINARY_DIR}/libcoralnpu_simulator_mpact.so")
    configure_file("${CORALNPU_MPACT_SIMULATOR_LIB}" "${MPACT_CMAKE_LIB}" COPYONLY)

    set(SIM_DEPS "${MPACT_CMAKE_LIB}")
  endif()

  iree_cc_library(
    NAME
      simulator
    HDRS
      "simulator_api.h"
    SRCS
      "simulator.cpp"
    INCLUDES
      "${CMAKE_SOURCE_DIR}"
      "${CMAKE_SOURCE_DIR}/third_party/coralnpu"
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
      coralnpu_runtime::sim::simulator
      iree::base
      iree::base::internal
      iree::hal::local::executable_environment
      iree::hal::local::executable_library
      iree::hal::local::executable_loader
    PUBLIC
  )

endif()
