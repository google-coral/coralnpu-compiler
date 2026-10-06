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


# Per-dispatch execution statistics of the CoralNPU HAL driver: the record,
# the collector that fills it through the run hooks of the execution backends
# (coralnpu_runtime::driver::run_hooks), and the writer that the driver calls
# as its dispatch observer (coralnpu_runtime::driver::dispatch_observer).
# Depends on these hook interfaces and on the ELF reader of the driver, not on
# the simulators.

iree_setup_c_src_root(
  PACKAGE_ROOT_PREFIX
    "coralnpu_metrics_runtime"
  DEFAULT_EXPORT_SET
    Runtime
  DEFAULT_INSTALL_COMPONENT
    IREEDevLibraries-Runtime
)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../..")

if(NOT IREE_ENABLE_CORALNPU_DRIVER)
  return()
endif()

# The statistics of a single dispatch.
iree_cc_library(
  NAME
    dispatch_stats
  HDRS
    "dispatch_stats.h"
  INCLUDES
    "${CMAKE_SOURCE_DIR}"
  PUBLIC
)

# Reads the memory layout of the firmware of a dispatch from its ELF.
iree_cc_library(
  NAME
    firmware_memory
  HDRS
    "firmware_memory.h"
  SRCS
    "firmware_memory.c"
  INCLUDES
    "${CMAKE_SOURCE_DIR}"
  DEPS
    ::dispatch_stats
    coralnpu_runtime::driver::coralnpu
    iree::base
    iree::hal::local::elf::elf_module
  PUBLIC
)

# Measures the peak stack use of a run through the run hooks.
iree_cc_library(
  NAME
    stack_peak
  HDRS
    "stack_peak.h"
  SRCS
    "stack_peak.c"
  INCLUDES
    "${CMAKE_SOURCE_DIR}"
  DEPS
    ::dispatch_stats
    coralnpu_runtime::driver::run_hooks
    iree::base
  PUBLIC
)

# Collects the statistics of a dispatch through the run hooks.
iree_cc_library(
  NAME
    stats_collector
  HDRS
    "stats_collector.h"
  SRCS
    "stats_collector.c"
  INCLUDES
    "${CMAKE_SOURCE_DIR}"
  DEPS
    ::dispatch_stats
    ::stack_peak
    coralnpu_runtime::driver::run_hooks
    iree::base
  PUBLIC
)

# Writes the statistics as JSON Lines or as a summary table
# (--coralnpu_stats_format).
iree_cc_library(
  NAME
    stats_writer
  HDRS
    "stats_writer.h"
  SRCS
    "stats_writer.c"
  INCLUDES
    "${CMAKE_SOURCE_DIR}"
  DEPS
    ::dispatch_stats
    ::firmware_memory
    ::stats_collector
    coralnpu_runtime::driver::dispatch_observer
    iree::base
    iree::base::threading
  PUBLIC
)

# The --coralnpu_stats_* flags, which configure the statistics writer.
iree_cc_library(
  NAME
    stats_flags
  HDRS
    "stats_flags.h"
  SRCS
    "stats_flags.c"
  INCLUDES
    "${CMAKE_SOURCE_DIR}"
  DEPS
    ::stats_writer
    coralnpu_runtime::driver::dispatch_observer
    iree::base
    iree::base::tooling::flags
  PUBLIC
)
