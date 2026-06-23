#!/usr/bin/env bash
# Exit immediately on error, undefined variables, or pipeline failures
set -euo pipefail

# Establish project root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Print usage instructions
usage() {
  cat <<EOF
Usage: $0 [OPTION...] {all|iree|llvm-project}

Apply patches to third-party submodules.

Options:
  -c, --commit         Commit each patch after applying it
  -r, --restore-first  Warning, you will lose unsaved work in the submodule. Restore submodule working tree to HEAD before applying patches
  -h, --help           Show this help message and exit

Targets:
  all                  Patch both iree and llvm-project submodules
  iree                 Patch only the iree submodule
  llvm-project         Patch only the llvm-project submodule
EOF
}

# Helper function to apply a patch only if it hasn't been applied yet
apply_patch() {
  local submodule_dir="$1"
  local patch_file="$2"

  echo "Checking patch: ${patch_file} -> ${submodule_dir}"

  # --check will fail if the patch is already applied or has conflicts
  if git -C "${submodule_dir}" apply --check <"${patch_file}" &>/dev/null; then
    if [[ "${cl_commit}" == "true" ]]; then
      git -C "${submodule_dir}" am <"${patch_file}"
    else
      git -C "${submodule_dir}" apply <"${patch_file}"
    fi
    echo "  [SUCCESS] Patch applied."
  else
    echo "  [SKIPPED] Patch already applied or has conflicts."
  fi
}

# Applies all matching patches to a submodule, optionally restoring it first
patch_submodule() {
  local submodule_dir="$1"
  local patch_glob="$2"

  local -a patches=()
  # NB: do not quote ${patch_glob}! Quoting it will disable the globing.
  readarray -t patches < <(ls -1 ${patch_glob})

  [[ "${#patches[@]}" -gt 0 ]] || return 0

  [[ "${cl_restore_first}" != 'true' ]] || git -C "${submodule_dir}" restore --source=HEAD --worktree -- .

  for patch_file in "${patches[@]}"; do
    apply_patch "${submodule_dir}" "${patch_file}"
  done
}

main() {
  local cl_options
  if ! cl_options="$(getopt -o "chr" -l "commit,help,restore-first," --name "$0" -- "$@")"; then
    usage >&2
    exit 2
  fi
  eval set -- "${cl_options}"

  local cl_commit=''
  local cl_restore_first=''

  while true; do
    case "$1" in
      --)
        shift
        break
        ;;

      -c | --commit)
        cl_commit="true"
        shift
        ;;
      -r | --restore-first)
        cl_restore_first="true"
        shift
        ;;
      -h | --help)
        usage
        exit 0
        ;;

      *)
        echo "Error: $0: internal error." >&2
        exit 1
        ;;
    esac
  done

  if [[ $# -ne 1 ]]; then
    echo "Error: $0: exactly one positional argument is required." >&2
    usage >&2
    exit 2
  fi

  case "$1" in
    all | iree) patch_submodule "third_party/iree" "iree-*.patch" ;;&
    all | llvm-project) patch_submodule "third_party/llvm-project" "llvm-project-*.patch" ;;&
    all | iree | llvm-project) ;;
    *)
      echo "Error: $0: unrecognized option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
