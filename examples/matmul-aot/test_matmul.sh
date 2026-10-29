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

# Matrix dimension (size NxN, default: 32)
N="${N:-32}"
USE_VERILATOR="${USE_VERILATOR:-false}"
# Set by test_matmul_highmem.sh: 1 MB ITCM/DTCM, MPACT only.
HIGHMEM="${HIGHMEM:-false}"
HIGHMEM_ARGS=()

main() {
  EXTRA_ARGS=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --transpose-lhs | --int8 | --bf16)
        EXTRA_ARGS+=("$1")
        shift
        ;;
      --use-verilator)
        USE_VERILATOR=true
        shift
        ;;
      -n | --size)
        N="${2:?missing value for $1}"
        shift 2
        ;;
      -n=* | --size=*)
        N="${1#*=}"
        shift
        ;;
      *)
        echo "Unknown argument: $1" >&2
        echo "Usage: $0 [-n|--size N] [--transpose-lhs] [--int8|--bf16] [--use-verilator]" >&2
        exit 1
        ;;
    esac
  done

  if [[ "${HIGHMEM}" == "true" ]]; then
    if [[ "${USE_VERILATOR}" == "true" ]]; then
      echo "Error: highmem is not supported on Verilator." >&2
      exit 1
    fi
    bazel build --config=dev //crt:coralnpu_tcm_highmem_ld
    HIGHMEM_ARGS=(--coralnpu-dtcm-size-kb=1024
      --coralnpu-linker-script-path="${ROOT_DIR}/bazel-bin/crt/coralnpu_tcm_highmem.ld")
  fi

  echo "=== Phase 1: Generating StableHLO MLIR (N=${N}) ==="
  bazel run --config=dev //examples/matmul-aot:export_matmul -- \
    --output="${TMP_DIR}/matmul.mlir" -n "${N}" "${EXTRA_ARGS[@]}"

  echo
  echo "=== Phase 2: Compiling to VMFB ==="

  compile_vmfb() {
    local out="$1"
    shift
    bazel run --config=dev @iree_core//tools:iree-compile -- \
      --iree-hal-target-device=local \
      --iree-hal-local-target-device-backends=llvm-cpu \
      --iree-llvmcpu-target-cpu-features=host \
      --iree-hal-target-device=coralnpu \
      --coralnpu-dump-affinity-profile-format=pretty \
      --coralnpu-roofline-speedup-threshold=0 \
      "${HIGHMEM_ARGS[@]}" "$@" "${TMP_DIR}/matmul.mlir" -o "${TMP_DIR}/${out}"
  }

  echo "[1/2] Compiling with Zvt (Matrix Extension)..."
  compile_vmfb matmul_zvt.vmfb

  echo "[2/2] Compiling without Zvt (RVV only)..."
  compile_vmfb matmul_rvv.vmfb \
    --coralnpu-target-cpu-features=+m,+f,+zbb,+zvl128b,+zve32f,+zfbfmin,+zvfbfmin,+zvfbfwma

  echo
  echo "=== Phase 3: Build run_matmul ==="
  SIM_NAME="MPACT"
  if [[ "${USE_VERILATOR}" == "true" ]]; then
    bazel build --config=dev @coralnpu_hw//hw_sim:libcoralnpu_simulator_vme.so
    SIM_NAME="Verilator"
  fi
  bazel build --config=dev //examples/matmul-aot:run_matmul

  # Print per-function cycle counts to stderr to compare Zvt with RVV.
  run_target() {
    CORALNPU_SIMULATOR="${SIM_NAME,,}" CORALNPU_STATS_FORMAT=pretty \
      LD_LIBRARY_PATH="${ROOT_DIR}/bazel-bin/external/coralnpu_hw+/hw_sim:${LD_LIBRARY_PATH:-}" \
      "${ROOT_DIR}/bazel-bin/examples/matmul-aot/run_matmul" \
      --vmfb="$1" -n "${N}" "${EXTRA_ARGS[@]}"
  }

  echo
  echo "=== Phase 4: Running matmul with Zvt (Matrix) on ${SIM_NAME} (N=${N}) ==="
  run_target "${TMP_DIR}/matmul_zvt.vmfb"

  echo
  echo "=== Phase 5: Running matmul without Zvt (RVV only) on ${SIM_NAME} (N=${N}) ==="
  run_target "${TMP_DIR}/matmul_rvv.vmfb"

  echo
  echo "=== DONE ==="
}

main "$@"
