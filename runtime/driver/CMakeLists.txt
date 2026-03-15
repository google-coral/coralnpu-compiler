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

iree_add_all_subdirs()

iree_register_external_hal_driver(
  NAME
    coralnpu
  DRIVER_TARGET
    coralnpu_runtime::driver::registration
  REGISTER_FN
    iree_hal_coralnpu_driver_module_register
)

set(coralnpu_hdrs
    coralnpu_command_buffer.h
    coralnpu_device.h
    coralnpu_driver.h
    coralnpu_event.h
    coralnpu_executable.h
    coralnpu_executable_cache.h
    coralnpu_semaphore.h
)
set(coralnpu_srcs
    coralnpu_command_buffer.c
    coralnpu_device.c
    coralnpu_driver.c
    coralnpu_event.c
    coralnpu_executable.c
    coralnpu_executable_cache.c
    coralnpu_semaphore.c
)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  set(CORALNPU_SIM_DEP "coralnpu_runtime::sim::simulation")
endif()

set(coralnpu_deps
    iree::base
    iree::base::internal
    iree::base::internal::arena
    iree::base::internal::cpu
    iree::base::internal::synchronization
    iree::hal
    iree::hal::local
    iree::hal::local::executable_environment
    iree::hal::utils::deferred_command_buffer
    iree::hal::utils::file_transfer
    iree::hal::utils::files
    iree::hal::utils::queue_emulation
    iree::hal::utils::semaphore_base
    ${CORALNPU_SIM_DEP}
)
set(coralnpu_defs)

iree_cc_library(
  NAME
    coralnpu
  HDRS
    ${coralnpu}
  INCLUDES
    "${PROJECT_SOURCE_DIR}"
  SRCS
    ${coralnpu_srcs}
  DEPS
    ${coralnpu_deps}
  INCLUDES
    "${CMAKE_SOURCE_DIR}"
  DEFINES
    ${coralnpu_defs}
  PUBLIC
)
