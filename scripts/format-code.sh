#!/usr/bin/env bash
set -euo pipefail

# cd to the root directory
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit

main() {
  # We don't want to format bazel-* directories, or cmake build directories.
  # We also don't want to format the external directory link added by
  # refresh_compile_commands.
  local -a dirs=()
  dirs+=(build_tools)
  dirs+=(compiler)
  dirs+=(examples)
  dirs+=(pjrt_plugin)
  dirs+=(runtime)
  dirs+=(scripts)
  dirs+=(tests)
  dirs+=(tools)

  local exit_code=0

  echo 'Formatting bazel files'
  bazel --quiet run //:buildifier || exit_code=$?

  echo
  echo 'Formatting C/C++ files'
  find "${dirs[@]}" -type f \( -name '*.hpp' -or -name '*.cpp' -or -name '*.cc' -or -name '*.h' -or -name '*.c' \) -print0 \
    | xargs -0 clang-format-19 -i || exit_code=$?

  echo
  echo 'Formating Python files'
  find "${dirs[@]}" -type f -name '*.py' -print0 \
    | xargs -0 bazel --quiet run --run_in_cwd //:yapf -- -i || exit_code=$?

  echo
  echo 'Formatting bash files'
  find "${dirs[@]}" -type f -name '*.sh' -print0 \
    | xargs -0 shfmt -w --language-dialect bash --indent 2 --case-indent --binary-next-line || exit_code=$?

  echo
  echo 'Done!'
  return "${exit_code}"
}

main "$@"
