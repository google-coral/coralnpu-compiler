#!/usr/bin/env bash
# Exit immediately on error, undefined variables, or pipeline failures
set -euo pipefail

# Establish project root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

usage() {
  echo "Usage: $0 {--bazel|--cmake}"
}

build-crt() {
  if [[ "${MODE}" == "bazel" ]]; then
    echo "=== Building CRT with Bazel ==="
    bazel build //crt:coralnpu_tcm_ld //crt:libcoralnpu_crt //crt:libcoralnpu_iree
    return
  fi

  local build_dir="../coralnpu-compiler-build"
  local crt_build_dir="${build_dir}/crt"
  local toolchain_file="${PWD}/crt/riscv32-toolchain.cmake"
  local toolchain_root_abs
  toolchain_root_abs=$(cd "${build_dir}" && pwd)/toolchain_rv32

  if [[ ! -f "${crt_build_dir}/CMakeCache.txt" ]]; then
    cmake -G Ninja \
      -S "${PWD}/crt" \
      -B "${crt_build_dir}" \
      -DCMAKE_TOOLCHAIN_FILE="${toolchain_file}" \
      -DRISCV_TOOLCHAIN_ROOT="${toolchain_root_abs}" \
      -DCMAKE_BUILD_TYPE=Release
  fi

  cmake --build "${crt_build_dir}" \
    --target coralnpu_crt coralnpu_iree
}

setup-bazel() {
  MODE="bazel"
  BUILD_TARGETS=(bazel build --config=dev //compiler/tools:coralnpu-compile @iree_core//tools:iree-run-module)
  IREE_COMPILE=(bazel run --config=dev //compiler/tools:coralnpu-compile --)
  IREE_RUN_MODULE=(bazel run --config=dev @iree_core//tools:iree-run-module --)
}

setup-cmake() {
  MODE="cmake"
  local build_dir="../coralnpu-compiler-build"
  if [[ ! -f "${build_dir}/CMakeCache.txt" ]]; then
    cmake -G Ninja -B "${build_dir}" -S . \
      -DIREE_HAL_DRIVER_LOCAL_SYNC=ON
  fi

  BUILD_TARGETS=(cmake --build "${build_dir}" --target coralnpu-compile iree-run-module)
  IREE_COMPILE=("${build_dir}"/coralnpu_compiler/tools/coralnpu-compile)
  IREE_RUN_MODULE=("${build_dir}"/third_party/iree/tools/iree-run-module)
}

main() {
  if [[ $# -ne 1 ]]; then
    setup-cmake
  elif [[ "$1" == "--bazel" ]]; then
    setup-bazel
  else
    setup-cmake
  fi

  if [[ "${MODE}" == "bazel" ]]; then
    echo "Bazel mode: toolchain download handled by Bazel."
  else
    echo "CMake mode: toolchain download handled by CMake."
  fi
  build-crt

  echo "=== Building Targets ==="
  "${BUILD_TARGETS[@]}"

  echo
  echo "=== Compiling MLIR to VMFB: add.mlir ==="

  local input_mlir="tests/models/stablehlo/add.mlir"
  output_vmfb="output/stablehlo/add.vmfb"

  mkdir -p "$(dirname "${output_vmfb}")"

  iree_compile_options=()

  # Input file
  iree_compile_options+=("${PWD}/${input_mlir}")
  # Output file
  iree_compile_options+=(-o "${PWD}/${output_vmfb}")

  # Configure the CoralNPU device
  iree_compile_options+=(--iree-hal-target-device=coralnpu)
  iree_compile_options+=(--coralnpu-target-abi=ilp32)
  iree_compile_options+=(--coralnpu-target-cpu-features="+m,+f,+zvl128b,+zve32x")

  "${IREE_COMPILE[@]}" "${iree_compile_options[@]}"

  echo
  echo "=== Running IREE Module: CoralNPU only ==="

  "${IREE_RUN_MODULE[@]}" \
    --device=coralnpu \
    --module="${PWD}/${output_vmfb}" \
    --function=main \
    --input=8xi32="[$(echo {1..8})]" \
    --input=8xi32="[$(echo {10..80..10})]"

  echo
  echo "=== Done! ==="

  echo
  echo "=== Compiling MLIR to VMFB: matmul_add.mlir ==="

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
  iree_compile_options+=(--iree-hal-local-target-device-backends=llvm-cpu)
  iree_compile_options+=(--iree-llvmcpu-target-cpu-features=host)

  # Configure the CoralNPU device
  iree_compile_options+=(--iree-hal-target-device=coralnpu)
  iree_compile_options+=(--coralnpu-target-abi=ilp32)
  iree_compile_options+=(--coralnpu-target-cpu-features="+m,+f,+zvl128b,+zve32x")

  "${IREE_COMPILE[@]}" "${iree_compile_options[@]}"

  echo
  echo "=== Running IREE Module: CoralNPU and Local CPU ==="

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
