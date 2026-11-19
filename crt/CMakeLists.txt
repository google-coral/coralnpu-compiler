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

# Generate linker scripts without modifying the upstream template.
function(generate_coralnpu_linker_script out_file itcm_length dtcm_origin dtcm_length csr_origin)
  file(READ
    "${CMAKE_CURRENT_LIST_DIR}/../third_party/coralnpu/toolchain/coralnpu_tcm.ld.tpl"
    script
  )

  string(REPLACE "@@ITCM_LENGTH@@" "${itcm_length}" script "${script}")
  string(REPLACE "@@DTCM_ORIGIN@@" "${dtcm_origin}" script "${script}")
  string(REPLACE "@@DTCM_LENGTH@@" "${dtcm_length}" script "${script}")
  string(REPLACE "@@STACK_SIZE@@" "4096" script "${script}")
  string(REPLACE "@@HEAP_LOCATION@@" "DTCM" script "${script}")
  string(REPLACE "@@HEAP_SIZE_SPEC@@"
    ". = ORIGIN(DTCM) + LENGTH(DTCM) - STACK_SIZE;"
    script "${script}")
  string(REPLACE "@@STACK_START_SPEC@@" "" script "${script}")

  string(APPEND script
    "\n__itcm_start = ORIGIN(ITCM);\n__itcm_size = LENGTH(ITCM);\n__dtcm_start = ORIGIN(DTCM);\n__dtcm_size = LENGTH(DTCM);\n__dma_base = DEFINED(__dma_base) ? __dma_base : ${csr_origin} + 0x2000;\n"
  )

  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${out_file}" "${script}")
endfunction()

generate_coralnpu_linker_script("coralnpu_tcm.ld" "8" "0x00010000" "32" "0x00030000")
generate_coralnpu_linker_script("coralnpu_tcm_highmem.ld" "1024" "0x00100000" "1024" "0x00200000")

add_library(coralnpu_crt STATIC
  "${CMAKE_CURRENT_LIST_DIR}/../third_party/coralnpu/toolchain/crt/crt.S"
  "${CMAKE_CURRENT_LIST_DIR}/../third_party/coralnpu/toolchain/crt/coralnpu_start.S"
  "${CMAKE_CURRENT_LIST_DIR}/../third_party/coralnpu/toolchain/crt/coralnpu_exceptions.cc"
  "${CMAKE_CURRENT_LIST_DIR}/../third_party/coralnpu/toolchain/crt/cxx_guards.cc"
  coralnpu_dma.c
)

add_library(coralnpu_iree STATIC
  coralnpu_main.c
)

target_include_directories(coralnpu_iree
  PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../third_party/iree/runtime/src"
)

foreach(target
    coralnpu_crt
    coralnpu_iree
)
  target_include_directories(${target}
    PRIVATE
      ${CMAKE_CURRENT_LIST_DIR}/..
  )

  target_compile_options(${target}
    PRIVATE
      -march=rv32imf_zicsr_zifencei_zbb
      -mabi=ilp32
      -mcmodel=medany
      -ffreestanding
      -fno-builtin
      -Os
      -ffunction-sections
      -fdata-sections
  )
endforeach()
