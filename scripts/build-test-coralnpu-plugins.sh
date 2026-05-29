#!/usr/bin/env bash
# Exit immediately on error, undefined variables, or pipeline failures
set -euo pipefail

# 1. Establish project root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "=== Setting up Repositories ==="

# Helper function to apply a patch only if it hasn't been applied yet
apply_patch() {
  local target_repo="$1"
  local patch_file="${ROOT_DIR}/$2"

  echo "Checking patch: $2 -> $target_repo"
  # --check will fail if the patch is already applied or has conflicts
  if git -C "${target_repo}" apply --check "${patch_file}" &>/dev/null; then
    git -C "${target_repo}" apply "${patch_file}"
    echo "  [SUCCESS] Patch applied."
  else
    echo "  [SKIPPED] Patch already applied or has conflicts."
  fi
}

# Apply all patches
apply_patch "third_party/iree" "iree-v3.10.0-0001-Bazel-fix-bazel-when-used-as-submodule.patch"
apply_patch "third_party/iree" "iree-v3.10.0-0002-Bazel-add-support-for-out-of-tree-plugins.patch"
apply_patch "third_party/iree" "iree-v3.10.0-0003-fix-iree-compile-for-jax-when-used-as-submodule.patch"
apply_patch "third_party/iree" "iree-v3.10.0-0004-Bazel-add-support-for-out-of-tree-drivers.patch"
apply_patch "third_party/llvm-project" "llvm-project-fix-for-jax-when-used-as-submodule.patch"

# 2. Define consistent build directory path
BUILD_DIR="${ROOT_DIR}/../coralnpu-compiler-build"

echo "=== Configuring CMake ==="
# Using the explicitly defined BUILD_DIR
cmake -G Ninja -B "${BUILD_DIR}" -S .

echo
echo "=== Building Targets ==="
cmake --build "${BUILD_DIR}" --target iree-compile iree-run-module

echo
echo "=== Compiling MLIR to VMFB ==="
# Define paths to tools and files explicitly for readability
IREE_COMPILE="${BUILD_DIR}/third_party/iree/tools/iree-compile"
INPUT_MLIR="${ROOT_DIR}/scripts/test.mlir"
OUTPUT_VMFB="${BUILD_DIR}/test_coralnpu.vmfb"

"${IREE_COMPILE}" \
  --iree-hal-target-device=coralnpu \
  --iree-llvmcpu-target-triple=riscv32 \
  --iree-llvmcpu-target-abi=ilp32 \
  --iree-llvmcpu-target-cpu-features=+m,+f,+zvl128b,+zve32f \
  "${INPUT_MLIR}" \
  -o "${OUTPUT_VMFB}"

echo
echo "=== Running IREE Module ==="
IREE_RUN_MODULE="${BUILD_DIR}/third_party/iree/tools/iree-run-module"

"${IREE_RUN_MODULE}" \
  --device=coralnpu \
  --module="${OUTPUT_VMFB}" \
  --function=add \
  --input="8xi32=[1,2,3,4,5,6,7,8]" \
  --input="8xi32=[10,20,30,40,50,60,70,80]"

echo
echo "=== Done! ==="
