#!/usr/bin/env bash
set -euo pipefail

# cd to the root directory
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit

main() {
  # We don't want to format bazel-* directories, or cmake build directories.
  # We also don't want to format the external directory link added by
  # refresh_compile_commands.
  local -a dirs=()
  dirs+=(compiler)
  dirs+=(examples)
  dirs+=(pjrt_plugin)
  dirs+=(runtime)
  dirs+=(scripts)
  dirs+=(tests)

  echo 'Formatting bazel files'
  bazel --quiet run //:buildifier

  echo
  echo 'Formatting C/C++ files'
  find "${dirs[@]}" -type f \( -name '*.hpp' -or -name '*.cpp' -or -name '*.h' -or -name '*.c' \) -print0 \
    | xargs -0 clang-format-19 -i

  echo
  echo 'Formating Python files'
  find "${dirs[@]}" -type f -name '*.py' -print0 \
    | xargs -0 bazel --quiet run --run_in_cwd //:yapf -- -i

  echo
  echo 'Formatting bash files'
  find "${dirs[@]}" -type f -name '*.sh' -print0 \
    | xargs -0 shfmt -w --language-dialect bash --indent 2 --case-indent --binary-next-line

  echo
  echo 'Done!'
}

main "$@"
