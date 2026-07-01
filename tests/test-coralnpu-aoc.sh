#!/usr/bin/env bash
# Exit immediately on error, undefined variables, or pipeline failures
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

usage() {
  echo "Usage: $0 {--bazel|--cmake}"
}

setup-bazel() {
  BUILD_TARGETS=(bazel build --config=dev //compiler/tools:coralnpu-compile @iree_core//tools:iree-run-module)
  IREE_COMPILE=(bazel run --config=dev //compiler/tools:coralnpu-compile --)
  IREE_RUN_MODULE=(bazel run --config=dev @iree_core//tools:iree-run-module --)
}

setup-cmake() {
  local build_dir="../coralnpu-compiler-build"
  if [[ ! -d "${build_dir}" ]]; then
    cmake -G Ninja -B "${build_dir}" -S . \
      -DIREE_TARGET_BACKEND_VMVX=ON \
      -DIREE_HAL_DRIVER_LOCAL_SYNC=ON
  fi

  BUILD_TARGETS=(cmake --build "${build_dir}" --target coralnpu-compile iree-run-module)
  IREE_COMPILE=("${build_dir}"/coralnpu_compiler/tools/coralnpu-compile)
  IREE_RUN_MODULE=("${build_dir}"/third_party/iree/tools/iree-run-module)
}

main() {
  if [[ $# -ne 1 ]]; then
    usage >&2
    return 2
  fi

  if [[ "$1" == "--bazel" ]]; then
    setup-bazel
  elif [[ "$1" == "--cmake" ]]; then
    setup-cmake
  else
    usage >&2
    return 2
  fi

  echo "=== Building Targets ==="
  "${BUILD_TARGETS[@]}"

  echo
  echo "=== Compiling MLIR to VMFB ==="

  local input_mlir="tests/models/stablehlo/matmul_add.mlir"
  output_vmfb="output/stablehlo/matmul_add.vmfb"

  mkdir -p "$(dirname "${output_vmfb}")"

  iree_compile_options=()

  # Input file
  iree_compile_options+=("${PWD}/${input_mlir}")
  # Output file
  iree_compile_options+=(-o "${PWD}/${output_vmfb}")

  # Configure the local device
  iree_compile_options+=(--iree-hal-target-device=local)
  iree_compile_options+=(--iree-hal-local-target-device-backends=vmvx)

  # Configure the CoralNPU device
  iree_compile_options+=(--iree-hal-target-device=coralnpu)
  iree_compile_options+=(--coralnpu-target-abi=ilp32)
  iree_compile_options+=(--coralnpu-target-cpu-features="+m,+f,+zvl128b,+zve32f")

  "${IREE_COMPILE[@]}" "${iree_compile_options[@]}"

  echo
  echo "=== Running IREE Module ==="

  "${IREE_RUN_MODULE[@]}" \
    --device=local-sync \
    --device=coralnpu \
    --module="${PWD}/${output_vmfb}" \
    --function=main \
    --input=4x8xi32="[$(echo {1..32})]" \
    --input=8x4xi32="[$(echo {1..32})]" \
    --input=4x4xi32="[$(echo {1..16})]" \
    --input=4x4xi32=10

  echo
  echo "=== Done! ==="
}

main "$@"
