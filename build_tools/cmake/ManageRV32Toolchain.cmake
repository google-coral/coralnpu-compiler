# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

function(download_rv32_toolchain toolchain_root)
  if(NOT EXISTS "${toolchain_root}")
    message(STATUS "CoralNPU RV32 Toolchain not found at ${toolchain_root}. Downloading...")

    set(TOOLCHAIN_URL "https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2026.07.15/riscv32-elf-ubuntu-22.04-gcc.tar.xz")
    set(TOOLCHAIN_TARBALL "${CMAKE_BINARY_DIR}/riscv32-elf-ubuntu-22.04-gcc.tar.xz")

    # Download the tarball
    file(DOWNLOAD "${TOOLCHAIN_URL}" "${TOOLCHAIN_TARBALL}"
         SHOW_PROGRESS
         EXPECTED_HASH SHA256=ae36abbec394b29643154c1b4a1322e829937d04e82f41b47f9c27d3bd68e543)

    message(STATUS "Extracting toolchain...")
    # Extract to a temporary directory to avoid cluttering CMAKE_BINARY_DIR
    set(TMP_EXTRACT_DIR "${CMAKE_BINARY_DIR}/tmp_toolchain_extract")
    file(ARCHIVE_EXTRACT
         INPUT "${TOOLCHAIN_TARBALL}"
         DESTINATION "${TMP_EXTRACT_DIR}")

    # Clean up tarball
    file(REMOVE "${TOOLCHAIN_TARBALL}")

    # Move to final destination
    file(RENAME "${TMP_EXTRACT_DIR}/riscv" "${toolchain_root}")

    # Clean up temporary directory
    file(REMOVE_RECURSE "${TMP_EXTRACT_DIR}")

    message(STATUS "Toolchain installed successfully at ${toolchain_root}")
  endif()
endfunction()
