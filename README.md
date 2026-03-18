# CoralNPU Compiler

An IREE compiler plugin for Coral NPU 

## Prerequisites

We try to use bazel as much as possible to manage dependencies. The following
are prerequisites that are not handled by bazel:
- git
- Bash
- Bazel 8.5.1
- clang 19
- shfmt, for bash scripts formatting (https://github.com/mvdan/sh)

In a Debian based linux distro you can get all of the above like this:
```shell
sudo apt install git bash bazel-8.5.1 clang-19 shfmt
```

## Dependencies

In-tree dependencies are located in the [third_party](third_party) directory.

Iree requires a specific commit of llvm-project. We have it checked out in
[third_party/llvm-project](third_party/llvm-project). You can check the
required commit hash by inspecting
[third_party/iree/third_party/llvm-project](third_party/iree/third_party/llvm-project)
(e.g. `git -C third_party/iree/third_party/llvm-project/ rev-parse HEAD`).

## Build

We use bazel. The version has to be backward compatible with IREE's requirements
(i.e. [third_party/iree/.bazelversion](third_party/iree/.bazelversion)).

To build the compiler, run:

```shell
# For a static iree-compile
bazel build @iree_core//tools:iree-compile
# Or, for dynamically linked with libIREECompiler.so
# bazel build --@iree_core//compiler/src/iree/compiler/API:link_shared @iree_core//tools:iree-compile
```

## Run the compiler

```shell
# NB: anything before the -- will be interperted by bazel and not iree-compile
bazel build @iree_core//tools:iree-compile -- [iree-compile options]
```

## Code style

We use Google style, enforced by scripts/format-code.sh.

Before pushing anything, run the following command (NB: commit or stage your
changes before, in case formatting does something horrible, and review the
formatting changes).

```shell
scripts/format-code.sh
```

## Toolchain

We use clang (and lld) 19.

Places that need to be updated when changing version/toolchain:
- [toolchain/cc_toolchain_config.bzl](toolchain/cc_toolchain_config.bzl)
- [scripts/foramt-code.sh](scripts/foramt-code.sh)

## Python

In general, you do not need to manually download or install Python packages (or
Python), everything is managed through bazel. For some editors, you might need
to recreate a similar environment as bazel's. You can do that like this:

```shell
python3.12 -m venv venv
. venv/bin/activate
pip install -r requirements_lock.txt
# if the above failes, try it with requirements.txt
```

### Version

We use Python 3.12

### Dependencies

Put direct dependencies in requirements.txt
Run the following to update requirements_lock.txt

```shell
bazel run //:requirements.update
```

## Shell scripts

Always use bash. Use this header:

```shell
#!/usr/bin/env bash
# Exit immediately on error (including in a pipeline), or when accessing an
# unset variable
set -euo pipefail
```

We assume Bash 4.0 (~2009) or later.

## Tools

If you use an LSP (e.g. clangd), you can run the following command to
generate/refresh `compiler_commands.json`:

```shell
bazel run //:refresh_compile_commands
```
