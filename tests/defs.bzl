"""Custom Bazel macros for CoralNPU compiler tests."""

load("@iree_core//build_tools/bazel:native_binary.bzl", "native_test")
load("//build_tools/bazel:bytecode_module.bzl", "coralnpu_bytecode_module")
load("//tools/check_gen:def.bzl", "check_gen_test_generator", "parse_instance_to_suffix")

def coralnpu_check_test(
        name,
        src,
        compiler_flags = [],
        runner_args = [],
        tags = [],
        timeout = None,
        deps = [],
        env = {},
        **kwargs):
    bytecode_module_name = name + "_bytecode_module"

    coralnpu_bytecode_module(
        name = bytecode_module_name,
        src = src,
        flags = compiler_flags,
        tags = tags,
        deps = deps,
        visibility = ["//visibility:private"],
    )

    native_test(
        name = name,
        args = [
            "--module=$(location :%s.vmfb)" % bytecode_module_name,
        ] + runner_args,
        data = [":%s.vmfb" % bytecode_module_name],
        src = "@iree_core//tools:iree-check-module",  # Use absolute label to be safe
        tags = tags + ["driver=coralnpu", "target=coralnpu"],
        timeout = timeout,
        env = env,
        **kwargs
    )

STANDARD_DEFAULT_GEN = "//tools/check_gen/generators:sequential_vmfb"

def _copy_generated_check_tests_impl(ctx):
    script_file = ctx.actions.declare_file(ctx.label.name + ".sh")
    lines = [
        "#!/bin/bash",
        "set -e",
        'DEST="$BUILD_WORKSPACE_DIRECTORY/' + ctx.attr.target_dir + '"',
        'mkdir -p "$DEST"',
        'echo "Copying check test files to $DEST..."',
    ]
    for f in ctx.files.filegroups:
        lines.append('cp -f "%s" "$DEST/"' % f.short_path)
    lines.append('echo "Successfully copied %d check test files to %s!"' % (len(ctx.files.filegroups), ctx.attr.target_dir))

    ctx.actions.write(
        output = script_file,
        content = "\n".join(lines) + "\n",
        is_executable = True,
    )

    runfiles = ctx.runfiles(files = ctx.files.filegroups)
    return [DefaultInfo(
        executable = script_file,
        runfiles = runfiles,
    )]

copy_generated_check_tests = rule(
    implementation = _copy_generated_check_tests_impl,
    attrs = {
        "target_dir": attr.string(mandatory = True),
        "filegroups": attr.label_list(mandatory = True, allow_files = True),
    },
    executable = True,
)

def _copy_generated_check_tests_suite_impl(ctx):
    script_file = ctx.actions.declare_file(ctx.label.name + ".sh")
    lines = [
        "#!/bin/bash",
        "set -e",
        'echo "Running check test generation copy suite: %s..."' % ctx.label.name,
    ]
    for dep in ctx.attr.deps:
        lines.append('"%s"' % dep[DefaultInfo].files_to_run.executable.short_path)
    lines.append('echo "Successfully generated and copied check test suite: %s!"' % ctx.label.name)

    ctx.actions.write(
        output = script_file,
        content = "\n".join(lines) + "\n",
        is_executable = True,
    )

    runfiles = ctx.runfiles(
        files = [dep[DefaultInfo].files_to_run.executable for dep in ctx.attr.deps],
        transitive_files = depset(transitive = [dep[DefaultInfo].default_runfiles.files for dep in ctx.attr.deps]),
    )
    return [DefaultInfo(
        executable = script_file,
        runfiles = runfiles,
    )]

copy_generated_check_tests_suite = rule(
    implementation = _copy_generated_check_tests_suite_impl,
    attrs = {
        "deps": attr.label_list(mandatory = True, cfg = "exec"),
    },
    executable = True,
)

def coralnpu_check_gen_tests(
        name,
        test,
        instances,
        arg_gens = [],
        default_gen = None,
        compiler_flags = [],
        runner_args = [],
        tags = [],
        timeout = None,
        deps = [],
        check_mlir_files = None,
        generation_targets = None,
        target_dir = "generated",
        **kwargs):
    """Defines check_gen generation targets and independent static check test targets.

    Args:
      name: Base name for the targets.
      test: The test function MLIR file.
      instances: List of instance shape strings.
      arg_gens: List of generator MLIR or VMFB files/targets.
      default_gen: Default generators.
      compiler_flags: Flags for the compiler.
      runner_args: Args for the runner.
      tags: Tags for the test targets.
      timeout: Timeout for the test targets.
      deps: Dependencies for the test targets.
      check_mlir_files: Optional list to collect static check MLIR file paths.
      generation_targets: Optional list to collect check_gen filegroup labels.
      target_dir: Subdirectory where static check test files are stored.
      **kwargs: Passed to all targets.
    """
    if default_gen == None:
        default_gen = STANDARD_DEFAULT_GEN

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

    # Generation target (produces MLIR files in bazel-out/)
    gen_target_name = "gen_" + name
    check_gen_test_generator(
        name = gen_target_name,
        test = test,
        arg_gens = arg_gens,
        default_gen = default_gen,
        instances = flat_instances,
        # Run only when explicitly called
        tags = tags + ["manual"],
    )
    if generation_targets != None and "manual" not in tags:
        generation_targets.append(":" + gen_target_name + "_mlir_files")

    # Individual copy target for this template line
    pkg = native.package_name()
    if target_dir:
        dest_dir = target_dir if target_dir.startswith(pkg) else (pkg + "/" + target_dir)
    else:
        dest_dir = pkg

    copy_generated_check_tests(
        name = "copy_" + name,
        filegroups = [":" + gen_target_name + "_mlir_files"],
        target_dir = dest_dir,
        tags = tags + (["manual"] if "manual" not in tags else []),
    )

    # Independent check test targets consuming static files in target_dir/
    generated_check_files = []
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

        check_mlir_file = (target_dir + "/" if target_dir != "" else "") + test_file_base + "_" + suffix + "_check.mlir"
        test_target_name = name + "_" + suffix + "_check_test"

        if "manual" not in combined_tags:
            coralnpu_check_test(
                name = test_target_name,
                src = check_mlir_file,
                compiler_flags = compiler_flags,
                runner_args = runner_args,
                tags = combined_tags,
                timeout = timeout,
                deps = deps,
                **kwargs
            )
            generated_check_files.append(check_mlir_file)

    native.filegroup(
        name = name + "_check_mlir_files",
        srcs = generated_check_files,
        visibility = ["//visibility:public"],
    )

    if check_mlir_files != None:
        check_mlir_files.extend(generated_check_files)
