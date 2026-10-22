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

  # MPACT is always required; Spike, Verilator and FPGA are opt-in. Prebuilt
  # MPACT/Spike/Verilator libraries come from CORALNPU_<NAME>_SIMULATOR_LIB or
  # bazel-bin and are built with Bazel if missing; FPGA is built from source.

  # Sets |var| to the library matching |glob|, building Bazel |target| first
  # if needed. A preset cache value for |var| wins if that file exists.
  function(coralnpu_find_bazel_simulator var name target glob)
    set(${var} "" CACHE FILEPATH "Path to the ${name} simulator library")
    if(${var} AND EXISTS "${${var}}")
      return()
    endif()
    foreach(_attempt RANGE 1)
      file(GLOB _found "${glob}")
      if(_found)
        list(GET _found 0 _found)
        set(${var} "${_found}" PARENT_SCOPE)
        return()
      endif()
      if(_attempt EQUAL 1)
        message(FATAL_ERROR "[CoralNPU Runtime] ${name} simulator library was built, but ${glob} does not exist.")
      endif()
      message(STATUS "[CoralNPU Runtime] ${name} simulator library not found. Building with Bazel (${target})...")
      execute_process(
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR} && export PWD=${CMAKE_SOURCE_DIR} && USE_BAZEL_VERSION=9.1.0 bazel build --config=dev ${target}"
        RESULT_VARIABLE _result
      )
      if(NOT _result EQUAL 0)
        message(FATAL_ERROR "[CoralNPU Runtime] Failed to build ${name} simulator library via Bazel.")
      endif()
    endforeach()
  endfunction()

  # MPACT
  coralnpu_find_bazel_simulator(CORALNPU_MPACT_SIMULATOR_LIB MPACT
    "@coralnpu_mpact//sim/hw_sim:libcoralnpu_simulator_mpact.so"
    "${CMAKE_SOURCE_DIR}/bazel-bin/external/*coralnpu_mpact*/sim/hw_sim/libcoralnpu_simulator_mpact.so")

  set(MPACT_CMAKE_LIB "${CMAKE_CURRENT_BINARY_DIR}/libcoralnpu_simulator_mpact.so")
  configure_file("${CORALNPU_MPACT_SIMULATOR_LIB}" "${MPACT_CMAKE_LIB}" COPYONLY)

  iree_cc_library(
    NAME
      simulator_mpact
    DEPS
      "${MPACT_CMAKE_LIB}"
    PUBLIC
  )

  # Spike, Verilator and FPGA are dlopen'd from LD_LIBRARY_PATH; copy them to
  # ${CMAKE_BINARY_DIR}/runtime/sim.
  if(CORALNPU_ENABLE_SPIKE)
    # Spike
    coralnpu_find_bazel_simulator(CORALNPU_SPIKE_SIMULATOR_LIB Spike
      "//runtime/sim/spike:libcoralnpu_simulator_spike.so"
      "${CMAKE_SOURCE_DIR}/bazel-bin/runtime/sim/spike/libcoralnpu_simulator_spike.so")
    configure_file("${CORALNPU_SPIKE_SIMULATOR_LIB}" "${CMAKE_BINARY_DIR}/runtime/sim/libcoralnpu_simulator_spike.so" COPYONLY)
  endif()

  if(CORALNPU_ENABLE_VERILATOR)
    # Verilator
    coralnpu_find_bazel_simulator(CORALNPU_VERILATOR_SIMULATOR_LIB Verilator
      "@coralnpu_hw//hw_sim:libcoralnpu_simulator_vme.so"
      "${CMAKE_SOURCE_DIR}/bazel-bin/external/*coralnpu_hw*/hw_sim/libcoralnpu_simulator_vme.so")
    configure_file("${CORALNPU_VERILATOR_SIMULATOR_LIB}" "${CMAKE_BINARY_DIR}/runtime/sim/libcoralnpu_simulator_vme.so" COPYONLY)
  endif()

  if(CORALNPU_ENABLE_FPGA)
    # FPGA
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
      coralnpu_runtime::driver::run_hooks
      coralnpu_runtime::sim::simulator
      iree::base
      iree::base::threading
      iree::hal::local::executable_library
      iree::hal::local::elf::elf_module
    PUBLIC
  )

endif()
