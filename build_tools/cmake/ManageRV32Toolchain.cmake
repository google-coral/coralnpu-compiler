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

    set(TOOLCHAIN_URL "https://storage.googleapis.com/shodan-public-artifacts/toolchain_iree_rv32.tar.gz")
    set(TOOLCHAIN_TARBALL "${CMAKE_BINARY_DIR}/toolchain_iree_rv32.tar.gz")

    # Download the tarball
    file(DOWNLOAD "${TOOLCHAIN_URL}" "${TOOLCHAIN_TARBALL}"
         SHOW_PROGRESS
         EXPECTED_HASH SHA256=01481183814cc66d6a8efb32681e2f74f5a7de321e93c81d563b65e64e3846a8)

    message(STATUS "Extracting toolchain...")
    # Extract to a temporary directory to avoid cluttering CMAKE_BINARY_DIR
    set(TMP_EXTRACT_DIR "${CMAKE_BINARY_DIR}/tmp_toolchain_extract")
    file(ARCHIVE_EXTRACT
         INPUT "${TOOLCHAIN_TARBALL}"
         DESTINATION "${TMP_EXTRACT_DIR}")

    # Clean up tarball
    file(REMOVE "${TOOLCHAIN_TARBALL}")

    # Move to final destination
    file(RENAME "${TMP_EXTRACT_DIR}/toolchain_iree_rv32imf" "${toolchain_root}")

    # Clean up temporary directory
    file(REMOVE_RECURSE "${TMP_EXTRACT_DIR}")

    message(STATUS "Toolchain installed successfully at ${toolchain_root}")
  endif()
endfunction()
