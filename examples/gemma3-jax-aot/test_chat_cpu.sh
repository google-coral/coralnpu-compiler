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
cd "${SCRIPT_DIR}"

main() {
  echo "=== Phase 1: Generating StableHLO MLIR ==="
  echo '| inputs:'
  echo '|   gm.nn.Gemma3_270M()'
  echo '|   gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_270M_IT)'
  echo '| output:'
  echo '|   gemma3_270m.mlir'
  bazel build --config=dev //examples/gemma3-jax-aot:export_gemma
  "${ROOT_DIR}/bazel-bin/examples/gemma3-jax-aot/export_gemma"

  echo
  echo "=== Phase 2: (building IREE and) Compiling to CPU VMFB ==="
  echo '| input:'
  echo '|   gemma3_270m.mlir'
  echo '| output:'
  echo '|   gemma3_270m_cpu.vmfb'
  bazel run --config=dev //compiler/tools:coralnpu-compile -- \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    --iree-llvmcpu-target-cpu-features=host \
    --iree-llvmcpu-loop-unrolling=false \
    --iree-llvmcpu-loop-interleaving=false \
    --iree-llvmcpu-loop-vectorization=false \
    "${PWD}/gemma3_270m.mlir" \
    -o "${PWD}/gemma3_270m_cpu.vmfb"

  echo
  echo "=== Phase 3: Build IREE runtime and python bindings ==="
  bazel build --config=dev \
    @iree_core//tools:iree-run-module \
    @iree_core//runtime/bindings/python:runtime \
    //examples/gemma3-jax-aot:chat_cpu

  echo
  echo "=== Phase 4: Running on CPU ==="
  echo '| input:'
  echo '|   gemma3_270m_cpu.vmfb'

  (
    export LD_LIBRARY_PATH="${ROOT_DIR}/runtime/sim"
    "${ROOT_DIR}/bazel-bin/examples/gemma3-jax-aot/chat_cpu" <<EOF
What is the capital of France?
What is the second largest city?
exit
EOF
  )

  echo
  echo "=== DONE ==="
}

main "$@"
