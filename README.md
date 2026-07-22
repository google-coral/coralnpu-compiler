# CoralNPU Compiler

An IREE compiler plugin for Coral NPU.

End-to-end flow for compiling and running JAX models (with plans to support
other frontends) through IREE, targeting CoralNPU-style RISCV-32 execution.
Currently, this uses host-side simulation instead of physical hardware.

## Cloning

The project includes a few submodules which need to be cloned as well.
To clone the project and its submodules (**very big**, read ahead):

```shell
git clone --recurse-submodules sso://spacebeaker/coralnpu-compiler
```

If you already cloned without `--recurse-submodules` (**very big**, read ahead):

```shell
git submodule update --init --recursive sso://spacebeaker/coralnpu-compiler
```

To reduce the size of submodules (especially `third_party/llvm-project`) you can
add `--shallow-submodules` to `git clone`, or `--depth=1` to `git submodule update`.
These will create a shallow clone of the submodules, with a history truncated to 1 revision.

### Patching submodules

*This is a temporary solution; we should have git forks of iree and llvm-project,
that we can patch normally.*

At the root of the project there are patches for `third_part/iree` and
`third_part/llvm-project`.
Those can be applied using the script `scripts/patch-third_party.sh`:

```shell
./scripts/patch-third_party.sh --restore-first all
```

See the `--help` option for more details.

## Prerequisites

We try to use bazel/cmake as much as possible to manage dependencies. The following
are prerequisites that are not handled by bazel/cmake:
- git
- Bash >= 4.0
- Bazel 8.6.0
- clang 19
- lld 19
- cmake >= 3.21
- shfmt, for bash scripts formatting (https://github.com/mvdan/sh)

In a Debian based linux distro you can get all of the above like this:
```shell
sudo apt install git bash bazel-8.6.0 clang-19 lld-19 shfmt cmake
```

## Dependencies

In-tree dependencies are located in the [third_party](third_party) directory.

Iree requires a specific commit of llvm-project. We have it checked out in
[third_party/llvm-project](third_party/llvm-project).
If you want to use a different revision of IREE, after checking it out in
[third_party/iree](third_party/iree), you can check the required llvm-project
commit hash by inspecting
[third_party/iree/third_party/llvm-project](third_party/iree/third_party/llvm-project)
(e.g. `git -C third_party/iree/third_party/llvm-project/ rev-parse HEAD`), and
then checking it out in [third_party/llvm-project](third_party/llvm-project).

## Build - bazel

Bazel's version has to be backward compatible with IREE's requirements
(i.e. [third_party/iree/.bazelversion](third_party/iree/.bazelversion)).

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
python3.12 -m venv venv
. venv/bin/activate
pip install -r requirements_lock.txt
# if the above failes, try it with requirements.txt
```

Dependencies:
Put direct dependencies in requirements.txt
Run the following to update requirements_lock.txt

```shell
bazel run //:requirements.update
```

### LSP support

If you use an LSP (e.g. clangd), you can run the following command to
generate/refresh `compiler_commands.json`:

```shell
bazel run //:refresh_compile_commands
```
<br>

## Build - cmake

Create a python venv with the required dependencies:

```shell
python3.12 -m venv venv
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

## Packaging and Distribution

This repository supports building Python wheels, standalone binary distribution archives, and local installation trees via native Bazel targets.

### 1. Staging Release Packages to `output/`

To build and stage all release packages (dist tarball + Python wheels) into `bazel-bin/output/`:

```shell
bazel build --config=release //:output
```

This generates:
- `bazel-bin/output/coralnpu-compiler-dist.tar.gz`
- `bazel-bin/output/coralnpu_compiler-0.0.1-py3-none-any.whl`
- `bazel-bin/output/coralnpu_runtime-0.0.1-py3-none-any.whl`

---

### 2. Specific Packaging Targets

#### Build Python Wheels (`coralnpu_compiler` and `coralnpu_runtime`)
```shell
# Build Python wheels for the local host platform (saved under bazel-bin/build_tools/bazel/python_packages/...)
bazel build --config=release //build_tools/bazel/python_packages/coralnpu_compiler:wheel //build_tools/bazel/python_packages/coralnpu_runtime:wheel
```

#### Build Standalone Binary Distribution Archive
```shell
# Build release tarball containing bin/, lib/, crt/, and toolchain_rv32/ (saved under bazel-bin/build_tools/bazel/dist_tar.tar.gz)
bazel build --config=release //build_tools/bazel:dist_tar
```

#### Native Local Installation
```shell
# Unpack and install distribution archive directly to a specified directory
bazel run --config=release //build_tools/bazel:install -- --prefix=/path/to/install
```

#### Testing the Installation

To verify that the installed compiler package and runtime binaries work end-to-end:

1. Save the following to model.mlir:
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
   /path/to/install/bin/coralnpu-compile \
       --iree-hal-target-device=local \
       --iree-hal-local-target-device-backends=llvm-cpu \
       --iree-llvmcpu-target-cpu=host \
       --iree-hal-target-device=coralnpu \
       --coralnpu-target-abi=ilp32 \
       --coralnpu-target-cpu-features=+m,+f,+zvl128b,+zve32f \
       model.mlir \
       -o model.vmfb
   ```

3. **Execute inference on the simulated CoralNPU device**:
   ```shell
   LD_LIBRARY_PATH=/path/to/install/lib /path/to/install/bin/iree-run-module \
       --device=coralnpu \
       --module=model.vmfb \
       --function=matmul \
       --input=32x64xf32=1.0 \
       --input=64x128xf32=2.0
   ```

#### Testing the Python Packages

To test the Python compiler (`coralnpu_compiler`) and runtime (`coralnpu_runtime`) wheel packages:

1. **Create and activate a virtual environment**:
   ```shell
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Build and install the Python wheels**:
   ```shell
   bazel build --config=release //build_tools/bazel/python_packages/coralnpu_compiler:wheel //build_tools/bazel/python_packages/coralnpu_runtime:wheel
   pip install bazel-bin/build_tools/bazel/python_packages/coralnpu_compiler/coralnpu_compiler-0.0.1-py3-none-any.whl \
               bazel-bin/build_tools/bazel/python_packages/coralnpu_runtime/coralnpu_runtime-0.0.1-py3-none-any.whl
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
           "--coralnpu-target-abi=ilp32",
           "--coralnpu-target-cpu-features=+m,+f,+zvl128b,+zve32f",
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

#### Multi-Platform Build (All Target Platforms)
```shell
# Build packages for all target platforms
bazel build --config=release //build_tools/bazel:all_platform_packages
```

---

### 3. Individual Cross-Compilation

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

### Running Tests with CMake

To run the tests with CMake, you need to configure CMake with testing enabled (`-DIREE_BUILD_TESTS=ON`):

```shell
cmake -G Ninja -B "${BUILD_DIR}" -S . -DIREE_BUILD_TESTS=ON
```

Then build the compiler, runtime, and test dependencies. Note that `iree-check-module` (required for running tests) is not built by default and must be built explicitly:

```shell
# Build default targets (compiler, runtime, generated tests)
cmake --build "${BUILD_DIR}" -j $(nproc)

# Build test runner dependency
cmake --build "${BUILD_DIR}" --target iree-check-module -j $(nproc)
```

Finally, run the tests using `ctest`. It is recommended to run tests in parallel:

```shell
# Run all tests
ctest --test-dir "${BUILD_DIR}" -j $(nproc)

# Run only CI tests
ctest --test-dir "${BUILD_DIR}" -L "ci" -j $(nproc)

# Run only StableHLO tests
ctest --test-dir "${BUILD_DIR}" -R "tests/models/stablehlo/.*" -j $(nproc)

# Run only Linalg op tests
ctest --test-dir "${BUILD_DIR}" -R "tests/models/linalg/.*" -j $(nproc)
```

## Run the compiler (using Bazel)
<!-- TODO: bazel and cmake should have an install target, and this section
should assume the compiler was installed using those -->

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
    --coralnpu-target-abi=ilp32 \
    --coralnpu-target-cpu-features=+m,+f,+zvl128b,+zve32f \
    model.mlir \
    -o model.vmfb
```

See the help message for the complete list of options:

```shell
bazel run --config=dev //compiler/tools:coralnpu-compile -- --help
```

CoralNPU compiler specific options are prefixed with `--coralnpu`.

### Useful options:

** Affinity execution profile report **
`--coralnpu-dump-affinity-profile-format={pretty|csv|json}`
Dumps statistics about the compilation (such as the number of dispatches, estimated data size, and estimated work) grouped by the device affinity (e.g., host vs CoralNPU).

## Examples:

### MobileNet V2 - ahead-of-time compilation

The compiler can be used to compile an mlir model to a vmfb binary, that can be loaded by the IREE runtime python bindings.

```shell
./examples/mobilenetv2-jax-aot/test_classify.sh
```

The script first exports the model to mlir, using the StableHLO dialect. It then
compiles the model to a vmfb, targeting the local host + CoralNPU. And finally
runs an inference using the compiled model (the CoralNPU payload runs in
simulation).

### pjrt_plugin

The PJRT plugin invokes the IREE HAL device APIs and builds the dynamic library used by JAX: libiree_pjrt_coralnpu_dylib.so. This library is intended to support compiling and running JAX models through the CoralNPU IREE backend.

```shell
# Build and test the JAX/PJRT flow
./scripts/build-test-coralnpu-jax.sh
```

This script provides a single-command flow for the JAX/PJRT path (Just-in-Time compilation).

It performs the following steps:

1. Applies required patches
2. Builds the IREE compiler via Bazel
3. Builds the CoralNPU PJRT plugin via Bazel
4. Compiles JAX models for the RV32-based CoralNPU backend
5. Runs the generated binaries

## Developer Tools

### MLIR Op Lister

A tool to list all registered MLIR operations.

Using Bazel:
```shell
bazel run //tools/list-mlir-ops -- [dialect_namespace]
```

Using CMake:
```shell
./build-cmake/tools/list-mlir-ops/list-mlir-ops [dialect_namespace]
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

