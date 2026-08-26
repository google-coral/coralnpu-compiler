# CoralNPU Compiler

An IREE compiler plugin for Coral NPU.

End-to-end flow for compiling and running JAX models (with plans to support
other frontends) through IREE, targeting CoralNPU-style RISCV-32 execution.
Currently, this uses host-side simulation instead of physical hardware.

## Cloning

To clone the project and its required submodules:

```shell
git clone sso://spacebeaker/coralnpu-compiler
cd coralnpu-compiler
```

Initialize the top-level submodules and `third_party/iree` submodules in parallel (`-j $(nproc)`) with shallow depth (`--depth=1`), excluding the redundant duplicate checkout of `llvm-project`:

```shell
git submodule update --init --depth=1 -j $(nproc) -- .
git -C third_party/iree submodule update --init --depth=1 -j $(nproc) -- . ":(exclude)third_party/llvm-project"
```

*Note: We exclude `third_party/llvm-project` inside `third_party/iree` to avoid downloading a redundant ~2.4 GB duplicate copy of LLVM.*

### Patching submodules

*This is a temporary solution; we should have git forks of iree and llvm-project,
that we can patch normally.*

At the root of the project there are patches for `third_party/iree` and `third_party/llvm-project`.
Those can be applied using the script `scripts/patch-third_party.sh`:

```shell
./scripts/patch-third_party.sh --restore-first all
```

See the `--help` option for more details.

*In any case, if you need to revert all the applied patches (uncommitted work will be lost):*

```shell
git submodule foreach 'git clean -fd && git reset --hard HEAD'
git submodule update --init --depth=1 -j $(nproc) --force -- .
git -C third_party/iree submodule update --init --depth=1 -j $(nproc) --force -- . ":(exclude)third_party/llvm-project"
```

## Prerequisites

We try to use bazel/cmake as much as possible to manage dependencies. The following
are prerequisites that are not handled by bazel/cmake:

- git
- Bash >= 4.0
- Bazel 8.6.0
- clang 19
- lld 19
- cmake >= 3.21
- Python 3.13
- shfmt, for bash scripts formatting (https://github.com/mvdan/sh)

In a Debian based linux distro you can get all of the above like this:

```shell
sudo apt install git bash bazel-8.6.0 clang-19 lld-19 cmake python3.13 python3.13-venv shfmt
```

To install Bazel for other distributions, please refer to [the official Bazel documentation](https://bazel.build/install).

### Dependencies

In-tree dependencies are located in the [`third_party`](third_party) directory.

Iree requires a specific commit of llvm-project. We have it checked out in
[`third_party/llvm-project`](third_party/llvm-project).
If you want to use a different revision of IREE, after checking it out in
[`third_party/iree`](third_party/iree), you can check the required llvm-project
commit hash by inspecting
[`third_party/iree/third_party/llvm-project`](third_party/iree/third_party/llvm-project)
(e.g. `git -C third_party/iree/third_party/llvm-project/ rev-parse HEAD`), and
then checking it out in [`third_party/llvm-project`](third_party/llvm-project).

---

## Build - bazel

Bazel's version has to be backward compatible with IREE's requirements
(i.e. [`third_party/iree/.bazelversion`](third_party/iree/.bazelversion)).

### Development build (initially long; incremental builds fast)

```shell
bazel build --config=dev \
    //compiler/tools:coralnpu-compile \
    @iree_core//tools:iree-run-module \
    @iree_core//compiler/bindings/python:compiler \
    @iree_core//runtime/bindings/python:runtime
```

To keep incremental builds as fast as possible, we use `--fission=yes`
(`--config=dev` does it, you don't need to do anything), which splits dwarf
information out of the .o files (see https://bazel.build/docs/user-manual). This
substantially reduces the input size to links and reduces link times
significantly.

<!--
For dynamiclly linked binary (with libIREECompiler.so):
bazel build --config=dev --@iree_core//compiler/src/iree/compiler/API:link_shared //compiler/tools:coralnpu-compile
-->

### Release build

Same as above, but use `--config=release` instead of `--config=dev`.

### Python

In general, you do not need to manually download or install Python packages (or
Python), everything is managed through bazel. For some editors, you might need
to recreate a similar environment as bazel's. You can do that like this:

```shell
python3.13 -m venv venv
. venv/bin/activate
pip install -r requirements_lock.txt
# if the above failes, try it with requirements.txt
```

You can also sandbox your Python 3.13 installation through [conda/miniconda](https://www.anaconda.com/docs/getting-started/working-with-conda/environments):

```shell
conda create --name venv python=3.13
conda activate venv
export PYTHONPATH=$CONDA_PREFIX
pip install -r requirements_lock.txt
```

### Dependencies:

Put direct dependencies in `requirements.txt`, Run the following to update `requirements_lock.txt`

```shell
bazel run //:requirements.update
```

### LSP support

If you use an LSP (e.g. clangd), you can run the following command to
generate/refresh `compiler_commands.json`:

```shell
bazel run --config=dev //:refresh_compile_commands
```

## Build - cmake

Create a Python virtural environment with the required dependencies (See the [Python](#python) section):

```shell
python3.13 -m venv venv
. venv/bin/activate
pip install -r requirements_lock.txt
# if the above failes, try it with requirements.txt
```

Set `BUILD_DIR` to some directory where you want the build results to be.
For example `BUILD_DIR=../coralnpu-compiler-build`.

Run once:

```shell
cmake -G Ninja -B "${BUILD_DIR}" -S .
```

Then, to build the compiler and runtime:

```shell
cmake --build "${BUILD_DIR}" --target coralnpu-compile iree-run-module
```

### Notes on Runtime Simulator in CMake

- **Compiler is standalone**: Building compiler targets (`coralnpu-compile`, IREE compiler plugins, LLVM/MLIR) via CMake is completely standalone and does not invoke or require Bazel.
- **Runtime Simulator Fallback**: The functional simulator library (`libcoralnpu_simulator_mpact.so` or `libcoralnpu_simulator.so`) is only needed when building the runtime HAL driver simulation backend (`coralnpu_runtime::sim`). If a pre-built simulator library path is not explicitly provided via `-DCORALNPU_MPACT_SIMULATOR_LIB=...` or `-DCORALNPU_VERILATOR_SIMULATOR_LIB=...`, CMake automatically invokes Bazel as a fallback to compile the simulator library from source.

---

## Run the standalone compiler (using Bazel)

A normal compilation, without errors or warnings, does not print anything to
stdout or stderr, unless a commandline option that specifically prints
information is used.

```shell
# NB: anything before the -- will be interperted by bazel and not coralnpu-compile
bazel run --config={dev|release} //compiler/tools:coralnpu-compile -- [coralnpu-compile options]
```

For example, to compile model.mlir:

```shell
# Compile for the host machine + CoralNPU (will run in simulation)
bazel run --config=dev //compiler/tools:coralnpu-compile -- \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    --iree-llvmcpu-target-cpu=host \
    --iree-hal-target-device=coralnpu \
    --iree-global-opt-experimental-disable-conv-generalization \
    model.mlir \
    -o model.vmfb
```

See the help message for the complete list of options:

```shell
bazel run --config=dev //compiler/tools:coralnpu-compile -- --help
```

CoralNPU compiler specific options are prefixed with `--coralnpu`.


### Useful options:

**Affinity execution profile report:**

`--coralnpu-dump-affinity-profile-format={pretty|csv|json}` dumps statistics about the compilation (such as the number of dispatches, estimated data size, and estimated work) grouped by the device affinity (e.g., host vs CoralNPU).


**Register allocation report:**

`--coralnpu-dump-register-allocation-report-format={pretty|json}`
`--coralnpu-dump-register-allocation-report-dir=<directory>`
`--coralnpu-dump-register-allocation-report-filter=<pattern>`

Dumps a report containing vector register utilization (unique registers used, vector configuration) and register allocator remarks (spills, reloads, copies) for each loop and at the function level. If the directory is `-`, the report is written to stdout; if empty or omitted, it is written to stderr. The filter option accepts a regex pattern to only report functions matching the name (default: `.*dispatch.*|main`).


**Executable linking:**

`--coralnpu-link-executables={true|false}` controls whether all executable dispatches are linked into a single library or emitted as individual self-contained executables (default: `false`). Emitting separate executables per dispatch avoids overflowing tightly constrained ITCM memory (e.g., 8 KB) on multi-dispatch models.

---

## Packaging and Distribution

This repository supports building Python wheels, standalone binary distribution archives, and local installation trees via native Bazel targets.

### Staging Release Packages to `output/`

To build and stage all release packages (dist tarball + Python wheels) into `bazel-bin/output/`:

```shell
bazel build --config=release //:output
```

This generates:

- `bazel-bin/output/coralnpu-compiler-dist.tar.gz`
- `bazel-bin/output/coralnpu_compiler-0.0.1-py3-none-any.whl`
- `bazel-bin/output/coralnpu_runtime-0.0.1-py3-none-any.whl`


### Specific Packaging Targets

#### Build Standalone Binary Distribution Archive
Build release tarball containing `bin/`, `lib/`, `crt/`, and `toolchain_rv32/` (saved under `bazel-bin/build_tools/bazel/dist_tar.tar.gz`):

```shell
bazel build --config=release //build_tools/bazel:dist_tar
```

#### Native Local Installation
Unpack and install distribution archive directly to a specified directory:

```shell
bazel run --config=release //build_tools/bazel:install -- --prefix=/path/to/install
```

#### Testing the Installation

To verify that the installed compiler package and runtime binaries work end-to-end:

1. **Save the following to `model.mlir`**:
   ```shell
   module {
     func.func @matmul(%arg0: tensor<32x64xf32>, %arg1: tensor<64x128xf32>) -> tensor<32x128xf32> {
       %cst = arith.constant 0.000000e+00 : f32
       %0 = tensor.empty() : tensor<32x128xf32>
       %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32x128xf32>) -> tensor<32x128xf32>
       %2 = linalg.matmul ins(%arg0, %arg1 : tensor<32x64xf32>, tensor<64x128xf32>)
                          outs(%1 : tensor<32x128xf32>) -> tensor<32x128xf32>
       return %2 : tensor<32x128xf32>
     }
   }
   ```

2. **Compile an MLIR model targeting CoralNPU**:
   ```shell
    bazel run --config=dev //compiler/tools:coralnpu-compile -- \
       --iree-hal-target-device=local \
       --iree-hal-local-target-device-backends=llvm-cpu \
       --iree-llvmcpu-target-cpu=host \
       --iree-hal-target-device=coralnpu \
       --iree-global-opt-experimental-disable-conv-generalization \
       $(pwd)/model.mlir \
       -o $(pwd)/model.vmfb
   ```

3. **Execute inference on the simulated CoralNPU device**:
   ```shell
    bazel run --config=dev @iree_core//tools:iree-run-module -- \
       --device=coralnpu \
       --module=$(pwd)/model.vmfb \
       --function=matmul \
       --input=32x64xf32=1.0 \
       --input=64x128xf32=2.0
   ```

### Build Python Wheels (`coralnpu_compiler` and `coralnpu_runtime`)
To build Python wheels for the local host platform (saved under `bazel-bin/build_tools/bazel/python_packages/...`):

```shell
bazel build --config=release \
    //build_tools/bazel/python_packages/coralnpu_compiler:wheel \
    //build_tools/bazel/python_packages/coralnpu_runtime:wheel
```

#### Testing the Python Packages

To test the Python compiler (`coralnpu_compiler`) and runtime (`coralnpu_runtime`) wheel packages,

1. **Create and activate a virtual environment**:
   ```shell
    python3.13 -m venv .venv
    source .venv/bin/activate
   ```
    The Python version has to be 3.13, if this is not available see the [Python section](#python).

2. **Build and install the Python wheels**:
   ```shell
    bazel build --config=release \
        //build_tools/bazel/python_packages/coralnpu_compiler:wheel \
        //build_tools/bazel/python_packages/coralnpu_runtime:wheel
   ```

   ```shell
    pip install \
        bazel-bin/build_tools/bazel/python_packages/coralnpu_compiler/coralnpu_compiler-0.0.1-py3-none-any.whl \
        bazel-bin/build_tools/bazel/python_packages/coralnpu_runtime/coralnpu_runtime-0.0.1-py3-none-any.whl
   ```

3. **Test if you can import and use the Python wheels**:
   ```shell
    python -c "import coralnpu.compiler as cnpuc; print(cnpuc)"
    <module 'coralnpu.compiler' from '/path/to/python3.13/site-packages/coralnpu/compiler/__init__.py'>
   ```

   ```shell
    python -c "import coralnpu.runtime as cnpurt; print(cnpurt)"
    <module 'coralnpu.runtime' from '/path/to/python3.13/site-packages/coralnpu/runtime/__init__.py'>
   ```

3. **Run end-to-end Python compilation and inference**:
   ```python
   import numpy as np
   import coralnpu.compiler as coralnpu_compiler
   import coralnpu.runtime as coralnpu_runtime

   mlir_code = """
   module {
     func.func @matmul(%arg0: tensor<32x64xf32>, %arg1: tensor<64x128xf32>) -> tensor<32x128xf32> {
       %cst = arith.constant 0.000000e+00 : f32
       %0 = tensor.empty() : tensor<32x128xf32>
       %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32x128xf32>) -> tensor<32x128xf32>
       %2 = linalg.matmul ins(%arg0, %arg1 : tensor<32x64xf32>, tensor<64x128xf32>)
                          outs(%1 : tensor<32x128xf32>) -> tensor<32x128xf32>
       return %2 : tensor<32x128xf32>
     }
   }
   """

   # Compile MLIR to VMFB bytes
   vmfb_bytes = coralnpu_compiler.compile_str(
       mlir_code,
       target_backends=["llvm-cpu", "coralnpu"],
       extra_args=[
           "--iree-hal-target-device=local",
           "--iree-hal-local-target-device-backends=llvm-cpu",
           "--iree-llvmcpu-target-cpu=host",
           "--iree-hal-target-device=coralnpu",
           "--iree-global-opt-experimental-disable-conv-generalization",
       ],
   )

   # Run inference on simulated CoralNPU
   config = coralnpu_runtime.Config("coralnpu")
   context = coralnpu_runtime.SystemContext(config=config)
   vm_module = coralnpu_runtime.VmModule.from_flatbuffer(context.instance, vmfb_bytes)
   context.add_vm_module(vm_module)

   arg0 = np.ones((32, 64), dtype=np.float32)
   arg1 = np.full((64, 128), 2.0, dtype=np.float32)
   result = context.modules.module.matmul(arg0, arg1)
   print("Output shape:", result.shape, "Output sample:", result[0, 0])
   ```

### Multi-Platform Build (All Target Platforms)
Build packages for all target platforms

```shell
bazel build --config=release //build_tools/bazel:all_platform_packages
```

### Individual Cross-Compilation

To cross-compile a single package for a specific target platform, pass `--platforms=//build_tools/bazel/platforms:<platform>`:

For example:

```shell
# Build distribution tarball for Linux AArch64 (ARM64)
bazel build --config=release --platforms=//build_tools/bazel/platforms:linux_aarch64 //build_tools/bazel:dist_tar
```

#### Available Platform Labels (`//build_tools/bazel/platforms:...`)
- `//build_tools/bazel/platforms:linux_x86_64` (Linux x86_64)
- `//build_tools/bazel/platforms:linux_aarch64` (Linux ARM64)
- `//build_tools/bazel/platforms:macosx_x86_64` (macOS Intel)
- `//build_tools/bazel/platforms:macosx_arm64` (macOS Apple Silicon)
- `//build_tools/bazel/platforms:windows_x86_64` (Windows x86_64)

> [!NOTE]
> **TODO (Cross-Compilation C++ Toolchains)**: The build system infrastructure (`platform()` targets, Starlark transitions, and wheel tagging) is in place for multi-platform builds. However, actually compiling C++ binaries for non-host platforms (e.g., `linux_aarch64`, `macosx_arm64`, `windows_x86_64`) requires registering corresponding C++ cross-compiler toolchains / sysroots (e.g. `aarch64-linux-gnu`, `osxcross`, `mingw-w64`) in `MODULE.bazel`. Currently, only the host C++ toolchain is registered.

---

## Testing

### Running Tests with Bazel

Run all tests in the repository:

```shell
bazel test --config=dev //tests/...
```

Run the CI test suite:

```shell
bazel test --config=dev //tests:ci
```

We have some StableHLO tests. To run just those:

```shell
bazel test --config=dev //tests/models/stablehlo/...
```

We also have Linalg op tests. To run just those:

```shell
bazel test --config=dev //tests/models/linalg/...
```

You can filter compiler tests by data-type tag (e.g., `i8`, `i16`, `i32`, `f32`):

```shell
# Run only f32 Linalg check tests
bazel test --config=dev //tests/models/linalg/... --test_tag_filters=f32

# Run only i8 StableHLO check tests
bazel test --config=dev //tests/models/stablehlo/... --test_tag_filters=i8
```

> [!NOTE]
> Running `bazel test` against `//tests/models/linalg/...` or `//tests/models/stablehlo/...` executes tests against static `.mlir` files checked into the repository under `generated_<type>/` directories. **Running tests will never trigger test regeneration.**

### Generating and Synchronizing Check Tests with Bazel

The Linalg and StableHLO compiler check tests are generated from dynamic-shape templates using the `check_gen` tool and checked into the repository under `generated_<type>/` directories (e.g., `tests/models/linalg/generated_i8/`).

#### 1. Generate only missing tests (Fast / Incremental):
When adding new test shapes or operations to `op_tests_*.bzl`, build and copy only the newly added tests:

```shell
bazel run --config=dev //tests:copy_missing
# or per-package:
# bazel run --config=dev //tests/models/linalg:copy_missing
# bazel run --config=dev //tests/models/stablehlo:copy_missing
```

#### 2. Remove stale / unreferenced tests:
When removing or renaming test templates or instances:

```shell
bazel run --config=dev //tests:clean_stale
# or per-package:
# bazel run --config=dev //tests/models/linalg:clean_stale
# bazel run --config=dev //tests/models/stablehlo:clean_stale
```

#### 3. One-step synchronization (Clean stale + copy missing):

```shell
bazel run --config=dev //tests:sync_generated
```

#### 4. Verify test synchronization:
Verify that all generated tests on disk match the declarations in `op_tests_*.bzl`:

```shell
bazel test --config=dev //tests:check_generated
```

#### 5. Force regenerate all check tests:
Force reference evaluation and regenerate all tests across the repository. Delete all existing generated test files first to ensure no stale or obsolete tests remain:

```shell
rm -f tests/models/*/generated_*/*.mlir
bazel run --config=dev //tests:copy_generated
```

### Manual Tests

Tests can be marked as `manual` (either on the op macro via `tags = ["manual"]` or on individual instance shapes via `[("<shape>", ["manual"])]`) when an operation or shape instance is work-in-progress or known to fail on CoralNPU:

- **Always Generated**: All declared tests in `op_tests_*.bzl` (including manual tests) are generated into `generated_<type>/` by `copy_missing` / `copy_generated`.
- **Excluded from CI**: Tests tagged `manual` are automatically skipped by `//tests:ci` and wildcard test runs (`//tests/...`).
- **Running a Manual Test**: You can execute any individual manual test directly by its target label:
  ```shell
  bazel test --config=dev //tests/models/linalg:<test_name>_<suffix>_check_test
  ```

### Running Tests with CMake/CTest

To run the tests with CMake, you need to configure CMake with testing enabled (`-DIREE_BUILD_TESTS=ON`):

```shell
cmake -G Ninja -B "${BUILD_DIR}" -S . -DIREE_BUILD_TESTS=ON
```

Then build the compiler, runtime, test runner, and test dependencies (including generated test bytecode modules):

```shell
# Build all test dependencies and test bytecode modules
cmake --build "${BUILD_DIR}" --target iree-test-deps -j $(nproc)

# (Optional) Or build only model test dependencies
cmake --build "${BUILD_DIR}" --target tests/models/all -j $(nproc)
```

Finally, run the tests using `ctest`. Always pass `-L "ci"` to run the passing CI test suite and exclude manual/failing test instances:

```shell
# Run all CI tests
ctest --test-dir "${BUILD_DIR}" -L "ci" -j $(nproc)

# Run only StableHLO CI tests
ctest --test-dir "${BUILD_DIR}" -R "tests/models/stablehlo/.*" -L "ci" -j $(nproc)

# Run only Linalg CI tests
ctest --test-dir "${BUILD_DIR}" -R "tests/models/linalg/.*" -L "ci" -j $(nproc)
```

You can also filter compiler check tests by data-type label (`f32`, `i16`, `i32`, `i8`):

```shell
# Run only f32 Linalg CI tests
ctest --test-dir "${BUILD_DIR}" -R "tests/models/linalg/.*" -L "f32" -LE "manual" -j $(nproc)

# Run only i8 StableHLO CI tests
ctest --test-dir "${BUILD_DIR}" -R "tests/models/stablehlo/.*" -L "i8" -LE "manual" -j $(nproc)
```

> [!NOTE]
> Like Bazel, running CMake check tests in `tests/models/linalg/` or `tests/models/stablehlo/` executes tests against static `.mlir` files checked into the repository under `generated_<type>/` directories. **Running tests with CMake will never trigger test regeneration.** To regenerate check tests, use the Bazel `copy_generated` targets described above.

---

## Examples:

### MobileNet V2 (JAX) - ahead-of-time compilation

The compiler can be used to compile a JAX/StableHLO MLIR model to a vmfb binary that can be loaded by the IREE runtime Python bindings.

```shell
./examples/mobilenetv2-jax-aot/test_classify.sh
```

The script first exports the model to mlir, using the StableHLO dialect. It then
compiles the model to a vmfb, targeting the local host + CoralNPU. And finally
runs an inference using the compiled model (the CoralNPU payload runs in
simulation).

### MobileNet V2 (TFLite / TOSA) - ahead-of-time compilation

The toolchain supports compiling TFLite models via TOSA MLIR targeting CoralNPU:

```shell
./examples/mobilenetv2-tflite-aot/test_classify.sh
```

The script:
1. Downloads the official MobileNetV2 TFLite model.
2. Legalizes the TFLite FlatBuffer to TOSA MLIR via `tosa-converter-for-tflite`.
3. Compiles the TOSA model to a `.vmfb` module targeting CoralNPU.
4. Executes image classification inference against the CoralNPU simulator.


### The `pjrt_plugin`:

The PJRT plugin invokes the IREE HAL device APIs and builds the dynamic library used by JAX for Just-in-Time (JIT) compilation and execution:

- **`libiree_pjrt_coralnpu_dylib.so`**: Unified multi-device PJRT plugin supporting both CPU (device 0) and CoralNPU (device 1) targets.

#### Running Multi-Device JAX Tests (CPU + CoralNPU):

```shell
./examples/gemma3-jax-pjrt/test_basic.sh
```

This script:
1. Builds `libIREECompiler.so`, `libiree_pjrt_coralnpu_dylib.so`, and `coralnpu-compile` via Bazel.
2. Runs [`examples/gemma3-jax-pjrt/basic.py`](examples/gemma3-jax-pjrt/basic.py) to test single-device CPU execution, single-device CoralNPU execution, and concurrent multi-device execution (CPU + CoralNPU) in JAX.

#### Running Gemma3-270M JAX Chat (CPU):

```shell
./examples/gemma3-jax-pjrt/test_chat.sh
```

This script builds `libiree_pjrt_coralnpu_dylib.so` and runs end-to-end interactive multi-turn chat with Gemma3-270M on CPU via JAX JIT compilation.
---

## Developer Tools

### MLIR Op Lister

A tool to list all registered MLIR operations.

Using Bazel:
```shell
bazel run //tools/list-mlir-ops -- [dialect_namespace]
```

Using CMake:
```shell
./$BUILD_DIR/tools/list-mlir-ops/list-mlir-ops [dialect_namespace]
```

---

## Code style

We use Google style, enforced by scripts/format-code.sh.

Before pushing anything, run the following command (NB: commit or stage your
changes before, in case formatting does something horrible, and review the
formatting changes).

```shell
scripts/format-code.sh
```

## Toolchain

We use clang 19, and lld (to build the compiler).

Places that need to be updated when changing version/toolchain:
- [.bazelrc](.bazelrc)
- [CMakeLists.txt](CMakeLists.txt)
- [scripts/format-code.sh](scripts/format-code.sh)

## Shell scripts

Always use bash. Use this header:

```shell
#!/usr/bin/env bash
# Exit immediately on error (including in a pipeline), or when accessing an
# unset variable
set -euo pipefail
```
