#!/usr/bin/env bash
# Exit immediately on error, undefined variables, or pipeline failures
set -euo pipefail

# 1. Reliably establish the root directory of the project
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "=== Building Targets ==="
# Combine bazel targets to save Bazel analysis time
bazel build --config=dev \
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
