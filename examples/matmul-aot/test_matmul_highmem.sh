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

# Exit immediately on error, or when accessing an unset variable
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

main() {
  echo "=== Phase 1: Generating StableHLO MLIR ==="
  bazel run --config=dev //examples/matmul-aot:export_matmul -- --output="${TMP_DIR}/matmul.mlir"

  echo
  echo "=== Phase 2: Compiling to VMFB ==="
  bazel build --config=dev //crt:coralnpu_tcm_highmem_ld

  bazel run --config=dev //compiler/tools:coralnpu-compile -- \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    --iree-llvmcpu-target-cpu-features=host \
    --iree-hal-target-device=coralnpu \
    --coralnpu-dump-affinity-profile-format=pretty \
    --coralnpu-dtcm-size-kb=1024 \
    --coralnpu-linker-script-path="${ROOT_DIR}/bazel-bin/crt/coralnpu_tcm_highmem.ld" \
    "${TMP_DIR}/matmul.mlir" \
    -o "${TMP_DIR}/matmul_highmem.vmfb"

  echo
  echo "=== Phase 3: Build run_matmul ==="
  bazel build --config=dev //examples/matmul-aot:run_matmul

  echo
  echo "=== Phase 4: Running matmul ==="
  (
    export LD_LIBRARY_PATH="${ROOT_DIR}/runtime/sim"
    "${ROOT_DIR}/bazel-bin/examples/matmul-aot/run_matmul" --vmfb="${TMP_DIR}/matmul_highmem.vmfb"
  )

  echo
  echo "=== DONE ==="
}

main "$@"
