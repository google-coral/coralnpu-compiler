# `check_gen`

`check_gen` is a utility tool used to generate concrete test cases (check tests) from MLIR functions with dynamic shapes. It uses IREE's compiler and runtime to evaluate input generators at build time, yielding concrete test inputs and outputs which are then baked into a self-checking MLIR test file.
This allows writing a single test file with dynamic shapes, and automatically generating tests for many different concrete configurations.

## What it does

Given:
1.  A MLIR file containing a function under test with dynamic shapes (e.g., `func.func @main(%arg0: tensor<?xi32>)`).
2.  One or more **generator** files. These are typically precompiled `.vmfb` files (recommended), but can also be `.mlir` files (which will be JIT-compiled on the fly). Each generator file must contain a module with one or more generator functions. Each function must return exactly one value, and its result type must be different from all other functions in the same module. Generator functions can accept arbitrary scalar arguments (e.g. fill values, seeds, sizes). The number of generator files must match the number of arguments of the function under test, and for each argument, the corresponding generator file must contain a function with a compatible result type.
3.  A list of concrete **instances** (arguments to be passed to the generator functions, e.g., `(8)`, `(10,256)`).

`check_gen` does the following for each instance:
1.  **Evaluates the generators**: It compiles and runs the generator functions using IREE's ConstEval JIT to produce concrete input tensors.
2.  **Evaluates the test function**: It compiles and runs the test function with the generated concrete inputs to get the expected output tensors.
3.  **Generates a check test**: It creates a new MLIR file that:
    *   Defines the input and expected output tensors as `util.unfoldable_constant` ops.
    *   Calls the test function with the input constants.
    *   Uses the `check.expect_eq` operation (or `check.expect_almost_eq` for floating-point tensors) to compare the test function's output with the expected output constants.

## Direct Usage

You can run the `check_gen` binary directly from the command line.

### Syntax

```bash
check_gen \
  -o <output_directory> \
  [--default-gen <default_gen.vmfb/mlir>] \
  --instance "<instance_1>" [--instance "<instance_2>" ...] \
  <test_dynamic.mlir> \
  [<generator_1.vmfb/mlir> [<generator_2.vmfb/mlir> ...]]
```

*   `-o <output_directory>`: The directory where the generated `_check.mlir` files will be written.
*   `--default-gen <default_gen.vmfb/mlir>`: (Optional) The path to a VMFB or MLIR file containing default generators.
*   `--instance "<instance>"`: The arguments for the generator functions. The format is a sequence of parenthesized arguments for each generator.
    *   For a single generator: `--instance "(4,8)"`
    *   For two generators: `--instance "(4,2)(5)"`
*   `<test_dynamic.mlir>`: The path to the MLIR file containing the function under test (must be named `@main`).
*   `<generator_N.vmfb/mlir>`: The paths to VMFB (or MLIR) files containing the generator modules. If `--default-gen` is used, some or all of these can be omitted or replaced with the keyword `default`.

### Naming Scheme of Generated Tests

The generated check test files are named using the following pattern:
`<output_directory>/<test_file_base_name>_<instance_suffix>_check.mlir`

*   `<test_file_base_name>`: The base name of the input test file (e.g., `add_rank2_i32` for `add_rank2_i32.mlir`).
*   `<instance_suffix>`: A representation of the concrete instance arguments, where values within a group are separated by underscores (`_`) and groups are separated by dashes (`-`). For example, if the instance is `(4,8)(4,8)`, the suffix will be `4_8-4_8`, resulting in a file named `add_rank2_i32_4_8-4_8_check.mlir`.

### Example

Suppose we have a test file `add_rank2_i32.mlir`.

**`add_rank2_i32.mlir`**:
```mlir
func.func @main(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}
```

We will use the `sequential.mlir` generator from the `generators` directory:
**`sequential.mlir`** (generator module for sequential values):
```mlir
[...]
module {
  [...]
  func.func @sequential_rank_2_i32(%dim0: index, %dim1: index) -> tensor<?x?xi32> {
    %alloc = tensor.empty(%dim0, %dim1) : tensor<?x?xi32>
    %res = linalg.generic {
      indexing_maps = [#map2],
      iterator_types = ["parallel", "parallel"]
    } outs(%alloc : tensor<?x?xi32>) {
    ^bb0(%out: i32):
      %idx0 = linalg.index 0 : index
      %idx0_i32 = arith.index_cast %idx0 : index to i32
      %idx1 = linalg.index 1 : index
      %idx1_i32 = arith.index_cast %idx1 : index to i32
      %dim1_i32 = arith.index_cast %dim1 : index to i32
      %tmp1 = arith.muli %idx0_i32, %dim1_i32 : i32
      %val1 = arith.addi %idx1_i32, %tmp1 : i32
      linalg.yield %val1 : i32
    } -> tensor<?x?xi32>
    return %res : tensor<?x?xi32>
  }
  [...]
}
```

To generate check tests, you first need to compile the generators to `.vmfb` files (recommended), or you can pass `.mlir` files directly (they will be JIT-compiled on the fly by `check_gen`).

To compile a generator manually:
```bash
iree-compile --iree-hal-target-backends=vmvx \
  tools/check_gen/generators/sequential.mlir \
  -o tools/check_gen/generators/sequential.vmfb
```

First build the `check_gen` tool:
```bash
bazel build --config=dev //tools/check_gen:check_gen
```

Then, run the binary directly with the `.vmfb` files:
```bash
./bazel-bin/tools/check_gen/check_gen \
  -o tools/check_gen/examples/output \
  --instance "(4,8)(4,8)" \
  --instance "(120,256)(120,256)" \
  tools/check_gen/examples/add_rank2_i32.mlir \
  tools/check_gen/generators/sequential.vmfb \
  tools/check_gen/generators/sequential.vmfb
```

This will generate:
- `tools/check_gen/examples/output/add_rank2_i32_4_8-4_8_check.mlir`
- `tools/check_gen/examples/output/add_rank2_i32_120_256-120_256_check.mlir`

For example, `add_rank2_i32_4_8-4_8_check.mlir` will look like this:

```mlir
module {
  func.func @"add_rank2_i32_4_8-4_8"() {
    %0 = util.unfoldable_constant dense<[
      [0, 1, 2, 3, 4, 5, 6, 7],
      [8, 9, 10, 11, 12, 13, 14, 15],
      [16, 17, 18, 19, 20, 21, 22, 23],
      [24, 25, 26, 27, 28, 29, 30, 31]
    ]> : tensor<4x8xi32>
    %1 = util.unfoldable_constant dense<[
      [0, 1, 2, 3, 4, 5, 6, 7],
      [8, 9, 10, 11, 12, 13, 14, 15],
      [16, 17, 18, 19, 20, 21, 22, 23],
      [24, 25, 26, 27, 28, 29, 30, 31]
    ]> : tensor<4x8xi32>
    %2 = stablehlo.add %0, %1 : tensor<4x8xi32>
    %3 = util.unfoldable_constant dense<[
      [0, 2, 4, 6, 8, 10, 12, 14],
      [16, 18, 20, 22, 24, 26, 28, 30],
      [32, 34, 36, 38, 40, 42, 44, 46],
      [48, 50, 52, 54, 56, 58, 60, 62]
    ]> : tensor<4x8xi32>
    "check.expect_eq"(%2, %3) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
    return
  }
}
```

## Default Generators

To simplify CLI commands and build targets, you can provide a **default generator file** containing generators for multiple MLIR types. This allows you to:
1.  Use the keyword `"default"` as a placeholder in the generator list.
2.  Omit generators entirely for arguments that can be satisfied by a generator in the default generator file.

Use the `--default-gen=<path>` flag to register a default generator file:

```bash
./bazel-bin/tools/check_gen/check_gen \
  -o tools/check_gen/examples/output \
  --default-gen=tools/check_gen/generators/sequential.mlir \
  --instance "(4,8)(4,8)" \
  tools/check_gen/examples/add_rank2_i32.mlir \
  default default
```

Or omit them entirely (since both arguments of `add_rank2_i32.mlir` are `tensor<?x?xi32>`, they will both be completed by the default generator):

```bash
./bazel-bin/tools/check_gen/check_gen \
  -o tools/check_gen/examples/output \
  --default-gen=tools/check_gen/generators/sequential.mlir \
  --instance "(4,8)(4,8)" \
  tools/check_gen/examples/add_rank2_i32.mlir
```

## Bazel Rules Integration

Typically, you don't run `check_gen` directly. Instead, you use the Bazel macros provided in `tests/defs.bzl` or `tools/check_gen/def.bzl`.

### `check_gen_tests` Macro (Recommended)

The easiest way to integrate check_gen tests into your Bazel builds is using the `check_gen_tests` macro. It automatically handles running the generator, compiling the generated MLIR files, and defining the test targets.

Here is the recommended setup using precompiled VMFB generators:

```bazel
load("//tools/check_gen:def.bzl", "check_gen_tests")

# Define the generator, compilation, and test targets in one go
check_gen_tests(
    name = "add_rank2_i32_macro",
    test = "add_rank2_i32.mlir",
    default_gen = "//tools/check_gen/generators:sequential_vmfb",
    # Or:
    # arg_gens = [
    #     "//tools/check_gen/generators:sequential_vmfb",
    #     "//tools/check_gen/generators:sequential_vmfb",
    # ],
    instances = ["(2,16)(2,16)"],
    compiler_flags = [
        "--iree-hal-target-backends=vmvx",
    ],
    runner_args = [
        "--device=local-task",
    ],
)
```

This will define:
- A generator target `generate_add_rank2_i32_macro` which runs `check_gen` once.
- For each instance (in this case, only one), it defines:
  - A compilation target: `add_rank2_i32_macro_2_16-2_16_bytecode`
  - A test target: `add_rank2_i32_macro_2_16-2_16_check_test`

You can run the tests with:
```bash
bazel test //path/to:add_rank2_i32_macro_2_16-2_16_check_test
```

---

### Lower-level Bazel Rules (Advanced)

If you need to customize the build or test steps (e.g., use a custom test runner, or run custom passes on the generated MLIR), you can use the lower-level rules.

#### `check_gen_test_generator` Rule

This rule runs the generator for specific instances and outputs the generated MLIR file as an artifact.

```bazel
load("//tools/check_gen:def.bzl", "check_gen_test_generator")

check_gen_test_generator(
    name = "add_rank2_i32_mlir",
    test = "add_rank2_i32.mlir",
    default_gen = "//tools/check_gen/generators:sequential_vmfb",
    arg_gens = [
        "default",
        "//tools/check_gen/generators:fill_vmfb",
    ],
    instances = ["(8,4)(1,8,4)"],
)
```

The generated MLIR files are exposed as individual targets. You can reference them in other rules using the label: `:<generator_target_name>_<instance_suffix>` (e.g., `:add_rank2_i32_mlir_8_4-1_8_4`).

#### Declaring compilation and test targets manually

You can manually link `check_gen_test_generator` output to `iree_bytecode_module` and `native_test`:

```bazel
load("@iree_core//build_tools/bazel:iree_bytecode_module.bzl", "iree_bytecode_module")
load("@iree_core//build_tools/bazel:native_binary.bzl", "native_test")

# 1. Compile the generated MLIR file to an IREE bytecode module
# We refer to the auto-selected target defined by check_gen_test_generator
iree_bytecode_module(
    name = "add_rank2_i32_8_4-1_8_4_vmfb_bytecode",
    src = ":add_rank2_i32_mlir_8_4-1_8_4",
    compile_tool = "@iree_core//tools:iree-compile",
    flags = ["--iree-hal-target-backends=vmvx"],
)

# 2. Run the compiled module using iree-check-module
native_test(
    name = "add_rank2_i32_8_4-1_8_4_vmfb_test",
    src = "@iree_core//tools:iree-check-module",
    args = ["--module=$(location :add_rank2_i32_8_4-1_8_4_vmfb_bytecode)"],
    data = [":add_rank2_i32_8_4-1_8_4_vmfb_bytecode"],
)
```

## CMake Integration

Similar to Bazel, you can use CMake macros to integrate check_gen tests. The `check_gen_tests` function is provided in `def.cmake`.

### `check_gen_tests` Function

This function automatically handles running the generator, compiling the generated MLIR files, and defining the ctest targets.

Here is the recommended setup:

```cmake
include(tools/check_gen/def.cmake)

check_gen_tests(
    NAME
      "add_rank2_i32_macro"
    TEST
      "add_rank2_i32.mlir"
    DEFAULT_GEN
      "${CMAKE_BINARY_DIR}/path/to/sequential_vmfb.vmfb"
    DEFAULT_GEN_TARGET
      "sequential_vmfb_target"
    INSTANCES
      "(2,16)(2,16)"
)
```

This will define:
- A custom target `generate_add_rank2_i32_macro` which runs `check_gen`.
- Compilation targets for each instance.
- CTest targets named `<NAME>_<SUFFIX>_check_test` (e.g., `add_rank2_i32_macro_2_16-2_16_check_test`).

You can run the tests with:
```bash
ctest -R "add_rank2_i32_macro.*_check_test"
```
