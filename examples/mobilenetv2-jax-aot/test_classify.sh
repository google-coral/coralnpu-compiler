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
  # bazel run sets BUILD_WORKSPACE_DIRECTORY, so it will write to source tree
  bazel run --config=dev //examples/mobilenetv2-jax-aot:export_mobilenet

  echo
  echo "=== Phase 2: Compiling to VMFB ==="

  local linker_path="$(bazel query --output=location "@rv32_toolchain//:bin/riscv32-unknown-elf-ld" 2>/dev/null | cut -d: -f1)"

  bazel run --config=dev //compiler/tools:coralnpu-compile -- \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    --iree-llvmcpu-target-cpu-features=host \
    --iree-hal-target-device=coralnpu \
    --coralnpu-target-abi=ilp32 \
    --coralnpu-target-cpu-features=+m,+f,+zvl128b,+zve32f \
    --coralnpu-embedded-linker-path="${linker_path}" \
    "${SCRIPT_DIR}/mobilenet_v2.mlir" \
    -o "${SCRIPT_DIR}/mobilenet_v2.vmfb"

  echo
  echo "=== Phase 3: Build classify ==="
  bazel build --config=dev //examples/mobilenetv2-jax-aot:classify

  echo
  echo "=== Phase 4: Running classification ==="
  (
    export LD_LIBRARY_PATH="${ROOT_DIR}/runtime/sim"
    "${ROOT_DIR}/bazel-bin/examples/mobilenetv2-jax-aot/classify" "${SCRIPT_DIR}/cat.jpg"
  )

  echo
  echo "=== DONE ==="
}

main "$@"
