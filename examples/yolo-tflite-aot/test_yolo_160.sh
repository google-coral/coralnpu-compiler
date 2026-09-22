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
  echo "=== Phase 1: Converting YOLOv8n (160x160) TFLite to TOSA MLIR ==="
  bazel run --config=dev //examples/yolo-tflite-aot:export_yolo_160 -- \
    --output "${SCRIPT_DIR}/yolo_160.mlir"

  echo
  echo "=== Phase 2: Compiling to VMFB ==="
  bazel run --config=dev @iree_core//tools:iree-compile -- \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    --iree-llvmcpu-target-cpu=host \
    --iree-hal-target-device=coralnpu \
    --iree-global-opt-experimental-disable-conv-generalization \
    --coralnpu-dump-affinity-profile-format=pretty \
    "${SCRIPT_DIR}/yolo_160.mlir" \
    -o "${SCRIPT_DIR}/yolo_160.vmfb"

  echo
  echo "=== Phase 3: Running YOLO (160x160) detection ==="
  bazel run --config=dev //examples/yolo-tflite-aot:detect_yolo_160 -- \
    --vmfb "${SCRIPT_DIR}/yolo_160.vmfb" \
    --check "${SCRIPT_DIR}/reference_160.npy" \
    "${ROOT_DIR}/examples/mobilenetv2-jax-aot/cat.jpg"

  echo
  echo "=== DONE ==="
}

main "$@"
