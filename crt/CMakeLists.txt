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

cmake_minimum_required(VERSION 3.20)

project(coralnpu_crt LANGUAGES C CXX ASM)

# Generate coralnpu_tcm.ld without modifying the upstream template.
file(READ
  "${CMAKE_CURRENT_LIST_DIR}/../third_party/coralnpu/toolchain/coralnpu_tcm.ld.tpl"
  CORALNPU_LINKER_SCRIPT
)

string(REPLACE "@@ITCM_LENGTH@@" "8"
  CORALNPU_LINKER_SCRIPT "${CORALNPU_LINKER_SCRIPT}")
string(REPLACE "@@DTCM_ORIGIN@@" "0x00010000"
  CORALNPU_LINKER_SCRIPT "${CORALNPU_LINKER_SCRIPT}")
string(REPLACE "@@DTCM_LENGTH@@" "32"
  CORALNPU_LINKER_SCRIPT "${CORALNPU_LINKER_SCRIPT}")
string(REPLACE "@@STACK_SIZE@@" "4096"
  CORALNPU_LINKER_SCRIPT "${CORALNPU_LINKER_SCRIPT}")
string(REPLACE "@@HEAP_LOCATION@@" "DTCM"
  CORALNPU_LINKER_SCRIPT "${CORALNPU_LINKER_SCRIPT}")
string(REPLACE "@@HEAP_SIZE_SPEC@@"
  ". = ORIGIN(DTCM) + LENGTH(DTCM) - STACK_SIZE;"
  CORALNPU_LINKER_SCRIPT "${CORALNPU_LINKER_SCRIPT}")
string(REPLACE "@@STACK_START_SPEC@@" ""
  CORALNPU_LINKER_SCRIPT "${CORALNPU_LINKER_SCRIPT}")

file(WRITE
  "${CMAKE_CURRENT_BINARY_DIR}/coralnpu_tcm.ld"
  "${CORALNPU_LINKER_SCRIPT}"
)

add_library(coralnpu_crt STATIC
  "${CMAKE_CURRENT_LIST_DIR}/../third_party/coralnpu/toolchain/crt/crt.S"
  "${CMAKE_CURRENT_LIST_DIR}/../third_party/coralnpu/toolchain/crt/coralnpu_start.S"
  "${CMAKE_CURRENT_LIST_DIR}/../third_party/coralnpu/toolchain/crt/coralnpu_exceptions.cc"
  "${CMAKE_CURRENT_LIST_DIR}/../third_party/coralnpu/toolchain/crt/cxx_guards.cc"
)

add_library(coralnpu_iree STATIC
  coralnpu_main.c
)

foreach(target
    coralnpu_crt
    coralnpu_iree
)
  target_include_directories(${target}
    PRIVATE
      ${CMAKE_CURRENT_LIST_DIR}/..
      "${CMAKE_CURRENT_LIST_DIR}/../third_party/iree/runtime/src"
  )

  target_compile_options(${target}
    PRIVATE
      -march=rv32imf_zicsr_zifencei_zbb
      -mabi=ilp32
      -mcmodel=medany
      -ffreestanding
      -fno-builtin
  )
endforeach()

set_target_properties(coralnpu_crt
  PROPERTIES
    OUTPUT_NAME coralnpu_crt
)

set_target_properties(coralnpu_iree
  PROPERTIES
    OUTPUT_NAME coralnpu_iree
)
