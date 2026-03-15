#!/usr/bin/env bash
# Exit immediately on error, undefined variables, or pipeline failures
set -euo pipefail

# 1. Reliably establish the root directory of the project
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
apply_patch "third_party/iree" "iree-v3.10.0-0001-fix-iree-compile-for-jax-when-used-as-submodule.patch"
apply_patch "third_party/llvm-project" "llvm-project-fix-for-jax-when-used-as-submodule.patch"

echo
echo "=== Building Targets ==="
# Combine bazel targets to save Bazel analysis time
bazel build \
  @iree_core//lib:libIREECompiler.so \
  //pjrt_plugin:iree_pjrt_coralnpu_dylib

echo
echo "=== Running JAX Test ==="

# 2. Export environment variables cleanly using the established ROOT_DIR
export PATH_TO_IREE="${ROOT_DIR}"
export IREE_PJRT_COMPILER_LIB_PATH="${PATH_TO_IREE}/bazel-bin/external/iree_core+/lib/libIREECompiler.so"
export PJRT_NAMES_AND_LIBRARY_PATHS="coralnpu_plugin:${PATH_TO_IREE}/bazel-bin/pjrt_plugin/libiree_pjrt_coralnpu_dylib.so"
export ENABLE_PJRT_COMPATIBILITY=1
export TF_CPP_VMODULE="cpu_client=3,pjrt_c_api_wrapper_impl=3"
export TF_CPP_MIN_LOG_LEVEL=0

# 3. Execute the test
${ROOT_DIR}/scripts/jax_test.py

echo "=== Done! ==="
