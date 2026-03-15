#!/usr/bin/env bash
set -euo pipefail

# cd to the root directory
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit

echo 'Formatting bazel files'
bazel --quiet run //:buildifier -- MODULE.bazel BUILD.bazel extensions.bzl compiler/ runtime/ pjrt_plugin/
echo

echo 'Formatting C/C++ files'
# Added -type f to ensure we only target files
find compiler runtime pjrt_plugin -type f \( -name '*.hpp' -or -name '*.cpp' -or -name '*.h' -or -name '*.c' \) -print0 | xargs -0 clang-format-19 -i

# echo
# echo 'Formating Python files'
# find tests -name '*.py' -print0 | xargs -0 bazel --quiet run --run_in_cwd //:yapf -- -i

echo
echo 'Formatting bash files'
find scripts -name '*.sh' -print0 | xargs -0 shfmt -w --language-dialect bash --indent 2 --case-indent --binary-next-line

echo
echo 'Done!'
