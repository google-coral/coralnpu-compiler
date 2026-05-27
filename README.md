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
sudo apt install git bash bazel-8.5.1 clang-19 shfmt cmake-format
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

### Development build (initially long; incremental builds fast)

```shell
bazel build --config=dev @iree_core//tools:iree-compile
```

To keep incremental builds as fast as possible, we use `--fission=yes`, which
splits dwarf information out of the .o files (see
https://bazel.build/docs/user-manual). This substantially reduces the input size
to links and reduces link times significantly.

<!--
For dynamiclly linked binary (with libIREECompiler.so):
bazel build --config=dev --@iree_core//compiler/src/iree/compiler/API:link_shared @iree_core//tools:iree-compile
-->

### Release build

```shell
bazel build --config=release @iree_core//tools:iree-compile
```

## Run the compiler

```shell
# NB: anything before the -- will be interperted by bazel and not iree-compile
bazel run @iree_core//tools:iree-compile -- [iree-compile options]
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

We use clang 19, and lld.

Places that need to be updated when changing version/toolchain:
- [.bazelrc](.bazelrc)
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
<br>

# CoralNPU End-to-End IREE-based ML Pipeline

This contains an initial proof-of-concept (PoC) implementation for enabling **CoralNPU** support in the IREE compilation and runtime stack.

The current goal is to support an end-to-end flow for compiling and running JAX models through IREE targeting CoralNPU-style RISCV-32 execution, using host-side simulation instead of physical hardware.

## Overview

This work adds experimental CoralNPU support across the following layers:

1. IREE compiler plugin
2. IREE HAL runtime device and driver
3. CoralNPU simulator integration
4. IREE PJRT plugin for JAX
5. Helper scripts for build, compile, run, and test workflows

The implementation is currently intended as an intermediate PoC checkpoint. It is designed to validate the full software path before finalizing memory-map integration, linker-script plumbing, and broader model support.

## Code Structure

```text
.
├── compiler/
│   └── CoralNPU compiler plugin (skeleton)
├── runtime/
│   ├── driver/
│   │   └── CoralNPU runtime device and driver
│   └── sim/
│       └── CoralNPU simulator integration
├── pjrt_plugin/
│   └── CoralNPU PJRT plugin
└── scripts/
    └── Helper scripts for building, compiling, running, and testing

```

### compiler/

Contains the CoralNPU IREE compiler plugin skeleton.

The coralnpu IREE compiler plugin includes a default llvm-cpu backend. This enables end-to-end simulation using an RV32 target while keeping the compiler path compatible with IREE’s existing lowering and executable generation infrastructure. The compiler plugin is responsible for producing binaries that can be executed by the CoralNPU runtime and simulator flow.

### runtime/

Contains the CoralNPU runtime plugin.

This includes the HAL runtime device, driver, and simulator integration used to execute CoralNPU-targeted binaries on a host machine.

#### runtime/driver/

Contains the standalone CoralNPU IREE HAL runtime device and driver.

The implementation is derived from IREE’s local-sync device, but is kept independent so that CoralNPU-specific behavior can be added more easily. This separation makes it easier to customize future behavior such as:
- Simulator dispatch
- Memory initialization
- Device-specific executable loading
- Runtime ABI adjustments
- Future hardware integration

#### runtime/sim/

Contains CoralNPU simulator integration.

This layer connects the runtime dispatch path to the CoralNPU simulators, including the Mpact-based simulator. This allows CoralNPU RISC-V RV32 targets to run on Linux x86-64 hosts. The current simulator path supports execution without requiring physical Physical CoralNPU development boards or External simulators (e.g., QEMU).

The simulator integration currently supports for simple model execution and validation.

### pjrt_plugin/

Contains the CoralNPU IREE PJRT plugin.

The PJRT plugin invokes the IREE HAL device APIs and builds the dynamic library used by JAX: libiree_pjrt_coralnpu_dylib.so. This library is intended to support compiling and running JAX models through the CoralNPU IREE backend.

### scripts/

Contains helper scripts for building, compiling, running, and testing the current PoC.

## Workflows

Two helper scripts are provided to simplify build and test workflows.

```shell
# Build and test the JAX/PJRT flow
bash ./scripts/build-test-coralnpu-jax.sh
```
This script provides a single-command flow for the JAX/PJRT path (Just-in-Time compilation).

It performs the following steps:

1. Applies required patches
2. Builds the IREE compiler via Bazel
3. Builds the CoralNPU PJRT plugin via Bazel
4. Compiles JAX models for the RV32-based CoralNPU backend
5. Runs the generated binaries

Use this script when validating the JAX-to-IREE-to-CoralNPU flow.

```shell
# Build and test the compiler/runtime plugin flow
bash ./scripts/build-test-coralnpu-plugins.sh
```

This script provides a single-command flow for the compiler/runtime plugin path (Ahead-of-Time compilation).

It performs the following steps:

1. Applies required patches
2. Builds the compiler and runtime via CMake
3. Compiles models for the RV32-based CoralNPU backend
4. Runs the generated binaries

Use this script when validating the lower-level IREE compiler, HAL runtime, and simulator integration.

If one encounters errors while installing Python dependencies, try running:

```shell
gpkg setup
```

## Current Status

The full pipeline currently builds and executes successfully. RISC-V Vector Extension (RVV) is supported and enabled by default.

Simple models are producing correct outputs. For example, array addition has been validated through the PoC flow.

This satisfies the initial proof-of-concept goal: demonstrating that the compiler, runtime, HAL device, PJRT plugin, and simulator can work together in a complete end-to-end path.

## Known Issues and Limitations

### Complex Models Are Not Yet Correct

More complex models, including GEMMA3-270M, do not yet produce correct outputs.

This is expected at the current stage of the PoC.

### CoralNPU Memory Map Is Not Fully Integrated

The CoralNPU memory map is not yet included in the compiler plugin through a linker script.

As a result, the generated binaries are not yet fully aligned with the intended CoralNPU memory layout during compilation.

### Manual Simulator Plumbing Is Required

Because the simulator currently lacks full memory-map integration, manual initialization plumbing is required.

This temporarily limits the range of supported models and may affect correctness for larger or more complex workloads.

## Long-Term Plan

The long-term solution is to fully integrate the CoralNPU linker script into the IREE compilation pipeline. Once this is complete, the compiler should emit binaries that naturally match the CoralNPU memory map, reducing or eliminating manual simulator-side setup.

Planned future work includes:

- Integrating the CoralNPU linker script into the compiler plugin
- Removing temporary manual memory initialization plumbing
- Improving support for larger and more complex models
- Expanding simulator correctness coverage
- Validating additional JAX model workloads
- Preparing the runtime path for future physical CoralNPU hardware support

## Summary

This PoC demonstrates that CoralNPU can be integrated into the IREE ecosystem across compiler, runtime, simulator, and PJRT layers.

At this checkpoint:

- The full flow builds successfully
- The runtime executes through the CoralNPU HAL device
- The simulator can run RV32 targets on Linux x86-64
- Simple model outputs are correct
- Complex model correctness remains future work
- Memory-map and linker-script integration are the main next steps
