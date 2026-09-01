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

  # ------------------------------------------------------------------------------
  # MPACT Simulator
  # ------------------------------------------------------------------------------
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

  set(MPACT_INCLUDE_DIR "")
  execute_process(
    COMMAND sh -c "USE_BAZEL_VERSION=8.6.0 bazel info output_base 2>/dev/null"
    OUTPUT_VARIABLE BAZEL_OUT_BASE
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(BAZEL_OUT_BASE)
    file(GLOB GLOBBED_MPACT_HDR "${BAZEL_OUT_BASE}/external/*coralnpu_mpact*/sim/hw_sim/coralnpu_simulator.h")
    if(GLOBBED_MPACT_HDR)
      list(GET GLOBBED_MPACT_HDR 0 FOUND_MPACT_HDR)
      get_filename_component(MPACT_HW_SIM_DIR "${FOUND_MPACT_HDR}" DIRECTORY)
      get_filename_component(MPACT_SIM_DIR "${MPACT_HW_SIM_DIR}" DIRECTORY)
      get_filename_component(MPACT_INCLUDE_DIR "${MPACT_SIM_DIR}" DIRECTORY)
    endif()
  endif()

  iree_cc_library(
    NAME
      simulator_mpact
    INCLUDES
      "${MPACT_INCLUDE_DIR}"
    DEPS
      "${MPACT_CMAKE_LIB}"
    PUBLIC
  )

  if(CORALNPU_ENABLE_VERILATOR)

    # ------------------------------------------------------------------------------
    # Verilator Simulator
    # ------------------------------------------------------------------------------
    set(CORALNPU_VERILATOR_SIMULATOR_LIB "" CACHE FILEPATH "Path to libcoralnpu_simulator_vme.so for verilator")
    if(NOT CORALNPU_VERILATOR_SIMULATOR_LIB OR NOT EXISTS "${CORALNPU_VERILATOR_SIMULATOR_LIB}")
      set(VERILATOR_CANDIDATES
        "${CMAKE_SOURCE_DIR}/bazel-bin/external/coralnpu_hw+/hw_sim/libcoralnpu_simulator_vme.so"
        "${CMAKE_SOURCE_DIR}/bazel-bin/external/coralnpu_hw/hw_sim/libcoralnpu_simulator_vme.so"
      )
      set(FOUND_VERILATOR_SO "")
      foreach(CANDIDATE IN LISTS VERILATOR_CANDIDATES)
        if(EXISTS "${CANDIDATE}")
          set(FOUND_VERILATOR_SO "${CANDIDATE}")
          break()
        endif()
      endforeach()

      if(NOT FOUND_VERILATOR_SO)
        message(STATUS "[CoralNPU Runtime] Verilator simulator library not found. Building with Bazel (@coralnpu_hw//hw_sim:libcoralnpu_simulator_vme.so)...")
        execute_process(
          COMMAND sh -c "cd ${CMAKE_SOURCE_DIR} && export PWD=${CMAKE_SOURCE_DIR} && USE_BAZEL_VERSION=8.6.0 bazel build --config=dev @coralnpu_hw//hw_sim:libcoralnpu_simulator_vme.so"
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
          file(GLOB GLOBBED_VERILATOR_SO "${CMAKE_SOURCE_DIR}/bazel-bin/external/*coralnpu_hw*/hw_sim/libcoralnpu_simulator_vme.so")
          if(GLOBBED_VERILATOR_SO)
            list(GET GLOBBED_VERILATOR_SO 0 FOUND_VERILATOR_SO)
          endif()
        endif()
      endif()

      if(NOT FOUND_VERILATOR_SO OR NOT EXISTS "${FOUND_VERILATOR_SO}")
        message(FATAL_ERROR "[CoralNPU Runtime] Verilator simulator library was built, but could not locate libcoralnpu_simulator_vme.so in bazel-bin.")
      endif()

      set(CORALNPU_VERILATOR_SIMULATOR_LIB "${FOUND_VERILATOR_SO}")
    endif()

    set(VERILATOR_CMAKE_LIB "${CMAKE_CURRENT_BINARY_DIR}/libcoralnpu_simulator_vme.so")
    configure_file("${CORALNPU_VERILATOR_SIMULATOR_LIB}" "${VERILATOR_CMAKE_LIB}" COPYONLY)
    configure_file("${CORALNPU_VERILATOR_SIMULATOR_LIB}" "${CMAKE_BINARY_DIR}/runtime/sim/libcoralnpu_simulator_vme.so" COPYONLY)

    iree_cc_library(
      NAME
        simulator_verilator
      INCLUDES
        "${CMAKE_SOURCE_DIR}"
        "${CMAKE_SOURCE_DIR}/third_party/coralnpu"
      DEPS
        "${VERILATOR_CMAKE_LIB}"
      PUBLIC
    )

  endif()

  if(CORALNPU_ENABLE_FPGA)

    # ------------------------------------------------------------------------------
    # FPGA Simulator
    # ------------------------------------------------------------------------------
    find_package(PkgConfig QUIET)
    set(FPGA_INCLUDE_DIRS
      "${CMAKE_SOURCE_DIR}"
      "${CMAKE_SOURCE_DIR}/third_party/coralnpu"
      "${CMAKE_SOURCE_DIR}/third_party/coralnpu/sw/utils/nexus_loader"
    )
    set(FPGA_DEPS "")
    if(PKG_CONFIG_FOUND)
      pkg_check_modules(LIBFTDI1 QUIET libftdi1)
      pkg_check_modules(LIBUSB1 QUIET libusb-1.0)
    endif()

    if(LIBFTDI1_FOUND)
      list(APPEND FPGA_INCLUDE_DIRS ${LIBFTDI1_INCLUDE_DIRS})
      list(APPEND FPGA_DEPS ${LIBFTDI1_LIBRARIES})
    else()
      list(APPEND FPGA_INCLUDE_DIRS "/usr/include/libftdi1")
      list(APPEND FPGA_DEPS "ftdi1")
    endif()

    if(LIBUSB1_FOUND)
      list(APPEND FPGA_INCLUDE_DIRS ${LIBUSB1_INCLUDE_DIRS})
      list(APPEND FPGA_DEPS ${LIBUSB1_LIBRARIES})
    else()
      list(APPEND FPGA_INCLUDE_DIRS "/usr/include/libusb-1.0")
      list(APPEND FPGA_DEPS "usb-1.0")
    endif()
    list(REMOVE_DUPLICATES FPGA_INCLUDE_DIRS)
    list(REMOVE_DUPLICATES FPGA_DEPS)

    iree_cc_library(
      SHARED
      NAME
        simulator_fpga
      SRCS
        "fpga/simulator.cpp"
        "${CMAKE_SOURCE_DIR}/third_party/coralnpu/sw/utils/nexus_loader/spi_master.cc"
      INCLUDES
        ${FPGA_INCLUDE_DIRS}
      COPTS
        "-fvisibility=default"
      DEPS
        ${FPGA_DEPS}
      PUBLIC
    )
    set_target_properties(
      coralnpu_runtime_sim_simulator_fpga
      PROPERTIES
        OUTPUT_NAME "coralnpu_simulator_fpga"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
    add_custom_command(
      TARGET coralnpu_runtime_sim_simulator_fpga POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/runtime/sim"
      COMMAND ${CMAKE_COMMAND} -E copy
        "$<TARGET_FILE:coralnpu_runtime_sim_simulator_fpga>"
        "${CMAKE_BINARY_DIR}/runtime/sim/libcoralnpu_simulator_fpga.so"
    )

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
    PUBLIC
  )

  iree_cc_library(
    NAME
      simulation
    HDRS
      "simulator_api.h"
      "simulator_backend.h"
      "simulator_elf_loader.h"
      "simulator_inline.h"
    SRCS
      "simulator_backend.c"
      "simulator_elf_loader.c"
      "simulator_inline.c"
    INCLUDES
      "${CMAKE_SOURCE_DIR}"
    DEPS
      coralnpu_runtime::driver::coralnpu
      coralnpu_runtime::sim::simulator
      iree::base
      iree::hal::local::executable_library
    PUBLIC
  )

endif()
