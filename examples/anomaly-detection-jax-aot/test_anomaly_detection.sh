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
cd "${SCRIPT_DIR}"

main() {
  echo "=== Phase 1: Generating StableHLO MLIR ==="
  bazel run --config=dev //examples/anomaly-detection-jax-aot:export_anomaly_detection -- \
    --output "${SCRIPT_DIR}/anomaly_detection.mlir"

  echo
  echo "=== Phase 2: Compiling to VMFB ==="
  bazel run --config=dev @iree_core//tools:iree-compile -- \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    --iree-llvmcpu-target-cpu=host \
    --iree-hal-target-device=coralnpu \
    --iree-global-opt-experimental-disable-conv-generalization \
    --coralnpu-dump-affinity-profile-format=pretty \
    "${SCRIPT_DIR}/anomaly_detection.mlir" \
    -o "${SCRIPT_DIR}/anomaly_detection.vmfb"

  echo
  echo "=== Phase 3: Running anomaly detection ==="
  bazel run --config=dev //examples/anomaly-detection-jax-aot:detect_anomaly -- \
    --vmfb "${SCRIPT_DIR}/anomaly_detection.vmfb" \
    --check "${SCRIPT_DIR}/reference.npy"

  echo
  echo "=== DONE ==="
}

main "$@"
