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

if(RISCV_TOOLCHAIN_INCLUDED)
  return()
endif()
set(RISCV_TOOLCHAIN_INCLUDED true)

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR riscv32)

set(RISCV_TOOLCHAIN_ROOT "${CMAKE_CURRENT_LIST_DIR}/../toolchain_rv32/" CACHE PATH
    "Path to the RISC-V toolchain installation")
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES RISCV_TOOLCHAIN_ROOT)

if(NOT RISCV_TOOLCHAIN_ROOT)
  message(FATAL_ERROR
      "Set RISCV_TOOLCHAIN_ROOT to the RISC-V toolchain directory")
endif()

set(RISCV_TOOLCHAIN_PREFIX
    "${RISCV_TOOLCHAIN_ROOT}/bin/riscv32-unknown-elf")

set(CMAKE_C_COMPILER
    "${RISCV_TOOLCHAIN_PREFIX}-clang")
set(CMAKE_CXX_COMPILER
    "${RISCV_TOOLCHAIN_PREFIX}-clang++")
set(CMAKE_ASM_COMPILER
    "${RISCV_TOOLCHAIN_PREFIX}-clang")
set(CMAKE_AR
    "${RISCV_TOOLCHAIN_PREFIX}-ar")
set(CMAKE_RANLIB
    "${RISCV_TOOLCHAIN_PREFIX}-ranlib")

set(CLANG_TARGET_FLAGS "--gcc-toolchain=${RISCV_TOOLCHAIN_ROOT}")
set(CMAKE_C_FLAGS_INIT "${CLANG_TARGET_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${CLANG_TARGET_FLAGS}")
set(CMAKE_ASM_FLAGS_INIT "${CLANG_TARGET_FLAGS}")

foreach(tool
    CMAKE_C_COMPILER
    CMAKE_CXX_COMPILER
    CMAKE_ASM_COMPILER
    CMAKE_AR
    CMAKE_RANLIB)
  if(NOT EXISTS "${${tool}}")
    message(FATAL_ERROR "${tool} not found: ${${tool}}")
  endif()
endforeach()

set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
