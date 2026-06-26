"""Rules and macros for generating IREE check tests from templates."""

load("@iree_core//build_tools/bazel:native_binary.bzl", "native_test")
load("//build_tools/bazel:bytecode_module.bzl", "coralnpu_bytecode_module")

def parse_instance_to_suffix(inst_str):
    """Parses an instance shape string into a filename suffix.

    Args:
        inst_str: The instance shape string, e.g., "(4,8)(8,4)".

    Returns:
        A suffix string, e.g., "4_8-8_4".
    """
    groups = inst_str.split(")(")
    parsed_groups = []
    for g in groups:
        if g.startswith("("):
            g = g[1:]
        if g.endswith(")"):
            g = g[:-1]
        dims = g.split(",")
        dims = [d.strip() for d in dims if d.strip()]
        parsed_groups.append("_".join(dims))
    return "-".join(parsed_groups)

def _check_gen_impl(ctx):
    check_gen_bin = ctx.executable.check_gen_tool

    test_file_base = ctx.file.test.basename
    if "." in test_file_base:
        test_file_base = test_file_base.split(".")[0]

    # Declare outputs in bazel-out
    outputs = []
    for inst in ctx.attr.instances:
        suffix = parse_instance_to_suffix(inst)
        out_file = ctx.actions.declare_file(test_file_base + "_" + suffix + "_check.mlir")
        outputs.append(out_file)

    # Run check_gen tool
    args = ctx.actions.args()
    args.add("-o", outputs[0].dirname)

    # Add default generators if present
    if ctx.file.default_gen:
        args.add("--default-gen", ctx.file.default_gen.path)

    for inst in ctx.attr.instances:
        args.add("--instance", inst)
    args.add(ctx.file.test)

    # Add generators (resolve mapping, handle -1 for "default")
    for idx in ctx.attr.arg_gens_mapping:
        if idx == -1:
            args.add("default")
        else:
            if idx < 0 or idx >= len(ctx.files.arg_gens):
                fail("arg_gens_mapping index {} out of range (0-{})".format(idx, len(ctx.files.arg_gens) - 1))
            args.add(ctx.files.arg_gens[idx])

    inputs = [ctx.file.test] + ctx.files.arg_gens
    if ctx.file.default_gen:
        inputs.append(ctx.file.default_gen)

    ctx.actions.run(
        outputs = outputs,
        inputs = inputs,
        executable = check_gen_bin,
        arguments = [args],
        mnemonic = "CheckGen",
        progress_message = "Generating check tests from template {}".format(ctx.file.test.path),
    )

    return [DefaultInfo(
        files = depset(outputs),
    )]

_check_gen = rule(
    implementation = _check_gen_impl,
    attrs = {
        "check_gen_tool": attr.label(
            default = Label("//tools/check_gen:check_gen"),
            executable = True,
            cfg = "exec",
        ),
        "test": attr.label(
            allow_single_file = [".mlir"],
            mandatory = True,
        ),
        "arg_gens": attr.label_list(
            allow_files = [".mlir", ".vmfb"],
            mandatory = True,
        ),
        "arg_gens_mapping": attr.int_list(
            mandatory = True,
        ),
        "default_gen": attr.label(
            allow_single_file = [".mlir", ".vmfb"],
        ),
        "instances": attr.string_list(
            mandatory = True,
        ),
    },
)

def check_gen_test_generator(name, test, arg_gens = [], instances = [], default_gen = None, **kwargs):
    """Generates test cases from a check template.

    Args:
        name: The name of the target.
        test: The test function MLIR file.
        arg_gens: The list of argument generator MLIR files (can contain duplicates).
        instances: The list of concrete instances to generate.
        default_gen: The default generators file target (vmfb or mlir).
        **kwargs: Other arguments passed to the underlying rule.
    """

    # Deduplicate arg_gens (skip "default" placeholder)
    unique_arg_gens = []
    for g in arg_gens:
        if g != "default" and g not in unique_arg_gens:
            unique_arg_gens.append(g)

    # Compute mapping (-1 for "default" placeholder)
    mapping = []
    for g in arg_gens:
        if g == "default":
            mapping.append(-1)
        else:
            idx = -1
            for i, u in enumerate(unique_arg_gens):
                if u == g:
                    idx = i
                    break
            if idx == -1:
                fail("Internal error: target not found in unique_arg_gens")
            mapping.append(idx)

    _check_gen(
        name = name,
        test = test,
        arg_gens = unique_arg_gens,
        arg_gens_mapping = mapping,
        default_gen = default_gen,
        instances = instances,
        **kwargs
    )

    # Resolve test file base name
    test_file_base = test
    if "/" in test_file_base:
        test_file_base = test_file_base.split("/")[-1]
    if "." in test_file_base:
        test_file_base = test_file_base.split(".")[0]

    # Define select_file targets for each instance
    for inst in instances:
        suffix = parse_instance_to_suffix(inst)
        gen_file_name = test_file_base + "_" + suffix + "_check.mlir"
        _select_file(
            name = name + "_" + suffix,
            srcs = [":" + name],
            filename = gen_file_name,
            visibility = kwargs.get("visibility"),
            tags = kwargs.get("tags"),
            testonly = kwargs.get("testonly"),
        )

def check_gen_tests(
        name,
        test,
        arg_gens = [],
        instances = [],
        default_gen = None,
        compiler_flags = [],
        runner_args = [],
        tags = [],
        timeout = None,
        deps = [],
        bytecode_tags = [],
        **kwargs):
    """Defines a test generator and test targets for templated tests.

    Args:
        name: Base name for the targets.
        test: The test function MLIR file.
        arg_gens: List of generator MLIR/VMFB files (can contain duplicates).
        instances: List of instance shape strings (or tuples with tags).
        default_gen: The default generators file target (vmfb or mlir).
        compiler_flags: Flags for the compiler (iree-compile).
        runner_args: Args for the runner (iree-check-module).
        tags: Tags for the test targets.
        timeout: Timeout for the test targets.
        deps: Dependencies for the test targets.
        bytecode_tags: Tags for the bytecode compilation targets.
        **kwargs: Passed to check_gen_test_generator.
    """

    # Resolve base name of test file
    test_file_base = test
    if "/" in test_file_base:
        test_file_base = test_file_base.split("/")[-1]
    if "." in test_file_base:
        test_file_base = test_file_base.split(".")[0]

    # Extract flat list of instances
    flat_instances = []
    for inst_entry in instances:
        if type(inst_entry) == "string":
            flat_instances.append(inst_entry)
        elif type(inst_entry) == "tuple" or type(inst_entry) == "list":
            flat_instances.append(inst_entry[0])

    # 1. Run check_gen once to generate all MLIR files
    check_gen_test_generator(
        name = "generate_" + name,
        test = test,
        arg_gens = arg_gens,
        default_gen = default_gen,
        instances = flat_instances,
        tags = tags,
        **kwargs
    )

    # 2. For each instance, define the compilation and test targets
    for inst_entry in instances:
        inst = ""
        extra_tags = []
        if type(inst_entry) == "string":
            inst = inst_entry
            extra_tags = []
        elif type(inst_entry) == "tuple" or type(inst_entry) == "list":
            inst = inst_entry[0]
            extra_tags = inst_entry[1]

        suffix = parse_instance_to_suffix(inst)
        combined_tags = tags + extra_tags

        bytecode_module_name = name + "_" + suffix + "_bytecode"

        # The select_file target name generated by check_gen_test_generator
        select_file_target_name = "generate_" + name + "_" + suffix

        # Compile the selected MLIR file
        coralnpu_bytecode_module(
            name = bytecode_module_name,
            src = ":" + select_file_target_name,  # Refer to the auto-generated target
            compile_tool = "@iree_core//tools:iree-compile",
            flags = list(compiler_flags),
            # if the test is taged manual, we have to pass the tag so we don't try to generate the test (which fails)
            tags = bytecode_tags + (["manual"] if "manual" in extra_tags else []),
            deps = deps,
            visibility = ["//visibility:private"],
        )

        # Run the test
        native_test(
            name = name + "_" + suffix + "_check_test",
            args = [
                "--module=$(location :%s)" % bytecode_module_name,
            ] + runner_args,
            data = [":%s" % bytecode_module_name],
            src = "@iree_core//tools:iree-check-module",
            tags = combined_tags,
            timeout = timeout,
        )

def _select_file_impl(ctx):
    for f in ctx.files.srcs:
        if f.basename == ctx.attr.filename:
            return [DefaultInfo(files = depset([f]))]
    fail("File %s not found in srcs of %s" % (ctx.attr.filename, ctx.label))

_select_file = rule(
    implementation = _select_file_impl,
    attrs = {
        "srcs": attr.label_list(mandatory = True),
        "filename": attr.string(mandatory = True),
    },
)
