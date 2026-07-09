#!/usr/bin/env bash
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

# Exit immediately on error (including in a pipeline), or when accessing an
# unset variable
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

main() {
  echo "=== Phase 1: Building Targets ==="
  bazel build -c opt \
    @iree_core//lib:libIREECompiler.so \
    //pjrt_plugin:iree_pjrt_coralnpu_dylib \
    //compiler/tools:coralnpu-compile

  # Link CRT and toolchain outputs so libIREECompiler.so in external/iree_core+/lib can resolve relative paths
  mkdir -p "${ROOT_DIR}/bazel-bin/external"
  ln -sf ../crt ../toolchain_rv32 "${ROOT_DIR}/bazel-bin/external/"

  echo
  echo "=== Phase 2: Running JAX Tests (CoralNPU & Multi-Device) ==="

  export PATH_TO_IREE="${ROOT_DIR}"
  export IREE_PJRT_COMPILER_LIB_PATH="${PATH_TO_IREE}/bazel-bin/external/iree_core+/lib/libIREECompiler.so"
  export PJRT_NAMES_AND_LIBRARY_PATHS="coralnpu_plugin:${PATH_TO_IREE}/bazel-bin/pjrt_plugin/libiree_pjrt_coralnpu_dylib.so"
  export IREE_PJRT_LOG_LEVEL=ERROR
  export ENABLE_PJRT_COMPATIBILITY=1

  uv run --no-project --offline --with-requirements "${ROOT_DIR}/requirements_lock.txt" \
    "${SCRIPT_DIR}/basic.py"

  echo
  echo "=== DONE ==="
}

main "$@"
