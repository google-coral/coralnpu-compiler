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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

main() {
  echo "=== Phase 1: Exporting Moonshine Tiny (JAX) to StableHLO MLIR ==="
  bazel run --config=dev //examples/moonshine-jax-aot:export_moonshine -- \
    --output "${SCRIPT_DIR}/moonshine.mlir"

  echo
  echo "=== Phase 2: Compiling to VMFB ==="
  bazel build --config=dev //crt:coralnpu_tcm_highmem_ld

  bazel run --config=dev @iree_core//tools:iree-compile -- \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    --iree-llvmcpu-target-cpu=host \
    --iree-hal-target-device=coralnpu \
    --iree-global-opt-experimental-disable-conv-generalization \
    --coralnpu-dump-affinity-profile-format=pretty \
    --coralnpu-dtcm-size-kb=1024 \
    --coralnpu-linker-script-path="${ROOT_DIR}/bazel-bin/crt/coralnpu_tcm_highmem.ld" \
    "${SCRIPT_DIR}/moonshine.mlir" \
    -o "${SCRIPT_DIR}/moonshine.vmfb"

  echo
  echo "=== Phase 3: Running Moonshine transcription ==="
  bazel run --config=dev //examples/moonshine-jax-aot:transcribe_moonshine -- \
    --vmfb "${SCRIPT_DIR}/moonshine.vmfb" \
    --check "${SCRIPT_DIR}/reference.npy"

  echo
  echo "=== DONE ==="
}

main "$@"
