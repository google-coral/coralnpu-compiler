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
  echo '| outputs:'
  echo '|   gemma3_270m_part1.mlir'
  echo '|   gemma3_270m_part2.mlir'
  echo '|   gemma3_270m_part3.mlir'
  bazel build --config=dev //examples/gemma3-jax-aot:export_gemma_split
  "${ROOT_DIR}/bazel-bin/examples/gemma3-jax-aot/export_gemma_split"

  echo
  echo "=== Phase 2: (building IREE and) Compiling to CPU VMFB ==="
  echo '| inputs:'
  echo '|   gemma3_270m_part1.mlir'
  echo '|   gemma3_270m_part2.mlir'
  echo '|   gemma3_270m_part3.mlir'
  echo '| outputs:'
  echo '|   gemma3_270m_part1.vmfb'
  echo '|   gemma3_270m_part2.vmfb'
  echo '|   gemma3_270m_part3.vmfb'

  bazel build --config=dev //compiler/tools:coralnpu-compile

  local -a iree_options=()
  iree_options+=('--iree-hal-target-device=local')
  iree_options+=('--iree-hal-local-target-device-backends=llvm-cpu')
  iree_options+=('--iree-llvmcpu-target-cpu-features=host')
  # iree_options+=('--iree-llvmcpu-loop-unrolling=false')
  # iree_options+=('--iree-llvmcpu-loop-interleaving=false')
  # iree_options+=('--iree-llvmcpu-loop-vectorization=false')

  echo "Compiling Part 1 (Layers 0..8)..."
  bazel run --config=dev //compiler/tools:coralnpu-compile -- \
    "${iree_options[@]}" \
    "${PWD}/gemma3_270m_part1.mlir" \
    -o "${PWD}/gemma3_270m_part1.vmfb" &

  echo "Compiling Part 2 (Layers 9..17 + Final Norm)..."
  bazel run --config=dev //compiler/tools:coralnpu-compile -- \
    "${iree_options[@]}" \
    "${PWD}/gemma3_270m_part2.mlir" \
    -o "${PWD}/gemma3_270m_part2.vmfb" &

  echo "Compiling Part 3 (Logits Decode)..."
  bazel run --config=dev //compiler/tools:coralnpu-compile -- \
    "${iree_options[@]}" \
    "${PWD}/gemma3_270m_part3.mlir" \
    -o "${PWD}/gemma3_270m_part3.vmfb" &

  echo "Waiting for background compilation jobs to finish..."
  wait

  echo
  echo "=== Phase 3: Build IREE runtime and python bindings ==="
  bazel build --config=dev \
    @iree_core//tools:iree-run-module \
    @iree_core//runtime/bindings/python:runtime \
    //examples/gemma3-jax-aot:chat_cpu_split

  echo
  echo "=== Phase 4: Running on CPU ==="
  echo '| inputs:'
  echo '|   gemma3_270m_part1.vmfb'
  echo '|   gemma3_270m_part2.vmfb'
  echo '|   gemma3_270m_part3.vmfb'

  (
    export LD_LIBRARY_PATH="${ROOT_DIR}/runtime/sim"
    "${ROOT_DIR}/bazel-bin/examples/gemma3-jax-aot/chat_cpu_split" <<EOF
What is the capital of France?
What is the second largest city?
exit
EOF
  )

  echo
  echo "=== DONE ==="
}

main "$@"
