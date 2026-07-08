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

# Exit immediately on error, unset variable access, or pipeline failure.
set -euo pipefail

# Displays CLI usage information and exits.
function show_usage() {
  cat <<EOF
Usage: bazel run //build_tools/bazel:install -- [options]

Options:
  --prefix=DIR     Installation prefix directory (required)
  --help           Show this help message

Environment Variables:
  DIST_TARBALL     Path to distribution tar.gz archive (set automatically by Bazel sh_binary)
EOF
}

# Main installation entrypoint.
function main() {
  local prefix=""

  # Parse command-line flags
  for arg in "$@"; do
    case "${arg}" in
      --prefix=*)
        prefix="${arg#*=}"
        ;;
      --help)
        show_usage
        exit 0
        ;;
      *)
        echo "Error: Unknown argument: ${arg}" >&2
        show_usage >&2
        exit 2
        ;;
    esac
  done

  # Validate prefix argument
  if [[ -z "${prefix}" ]]; then
    echo "Error: --prefix=DIR is required for installation." >&2
    show_usage >&2
    exit 2
  fi

  # Validate DIST_TARBALL environment variable
  if [[ -z "${DIST_TARBALL:-}" || ! -f "${DIST_TARBALL}" ]]; then
    echo "Error: DIST_TARBALL environment variable is not set or file does not exist: '${DIST_TARBALL:-}'" >&2
    show_usage >&2
    exit 1
  fi

  # Resolve relative prefix against user's working directory when invoked via `bazel run`
  if [[ "${prefix}" != /* && -n "${BUILD_WORKING_DIRECTORY:-}" ]]; then
    prefix="${BUILD_WORKING_DIRECTORY}/${prefix}"
  fi

  echo ">>> Unpacking CoralNPU Compiler distribution to ${prefix}..."
  mkdir -p "${prefix}"
  tar -xzf "${DIST_TARBALL}" -C "${prefix}"
  echo ">>> Installation completed successfully at ${prefix}"
}

main "$@"
