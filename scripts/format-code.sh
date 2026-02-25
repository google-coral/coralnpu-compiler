#!/usr/bin/env bash
# Exit immediately on error (including in a pipeline), or when accessing an
# unset variable
set -euo pipefail

# cd to the root directory
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit

echo 'Formating bazel files'
bazel --quiet run //:buildifier

echo
echo 'Formating C/C++ files'
find compiler \( -name '*.hpp' -or -name '*.cpp' -or -name '*.h' -or -name '*.c' \) -print0 | xargs -0 clang-format-19 -i

# echo
# echo 'Formating Python files'
# find tests -name '*.py' -print0 | xargs -0 bazel --quiet run --run_in_cwd //:yapf -- -i

echo
echo 'Formating bash files'
find scripts -name '*.sh' -print0 | xargs -0 shfmt -w --language-dialect bash --indent 2 --case-indent --binary-next-line
