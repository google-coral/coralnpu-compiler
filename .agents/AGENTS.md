# Project Rules

## Project Overview & Repository Layout
- **CoralNPU Compiler**: An IREE compiler plugin and HAL runtime driver for
  compiling and running ML models (JAX/StableHLO, TFLite/TOSA, Linalg) targeting
  CoralNPU RISC-V32 (`rv32` + RVV / `zvt` matrix extension).
- **Key Directories**:
  - `compiler/`: IREE compiler target backend (`CoralNPUTarget`), MLIR dialects,
    passes, tiling, and codegen.
  - `runtime/`: IREE HAL driver and simulator backends (`mpact`, `spike`,
    `verilator`, `fpga`).
  - `crt/`: Bare-metal C runtime and linker scripts for RISC-V32 dispatches.
  - `pjrt_plugin/`: Multi-device (CPU + CoralNPU) PJRT plugin for JAX JIT
    execution.
  - `tests/`: Compiler, runtime, and model check tests (`tests/models/linalg/`,
    `tests/models/stablehlo/`).
  - `tools/` & `scripts/`: Developer utilities (`check_gen`, `list-mlir-ops`,
    `format-code.sh`, `patch-third_party.sh`).
  - `build_tools/`: Bazel and CMake build rules, packaging, and Python wheel
    definitions.
  - `third_party/`: Submodules (`iree`, `llvm-project`, `coralnpu`), patched via
    root-level `*.patch` files.
- Refer to `README.md` for full build, simulator, packaging, and test-generation
  workflows.

## Agent Governance & Test Integrity
- The agent must never change agent rules without explicit authorization.
- Never bypass failing tests by tagging them `manual`, disabling test cases, or
  weakening existing `CHECK` assertions/tolerances without explicit
  authorization; tests must genuinely exercise the target code path.
- Operate directly on the current Git worktree; never create or switch Git
  branches, create commits, or push to Gerrit unless explicitly requested.
- Default to stress-testing assumptions: lead with risks, unhandled edge cases,
  and simpler codebase/upstream alternatives, and avoid validating flattery. If
  a request contradicts an earlier goal or project invariant, point out the
  contradiction before editing files.

## Temporary Files
- All temporary files (scratch files, scripts, temporary outputs, logs, output
  redirections, etc.) created by the agent MUST be placed in the `./agent-temp/`
  directory.
- Do not clutter the project root or other directories with temporary files.
- If the `./agent-temp/` directory does not exist, create it before writing any
  temporary files.

## Design Documents & Two-CL Workflow
- For non-trivial changes (new passes, operation enablements, third-party
  patches, or changes >150 LOC), suggest following the two-CL Spec-Driven
  Development workflow using the `/design-doc` and `/implement-design` skills
  (see `./docs/design/README.md`,
  `./.agents/skills/design-doc/references/standards.md`, and
  `./.agents/skills/implement-design/references/standards.md` for full details):
  1. **CL 1 (Design CL — `/design-doc`)**:
     - `/design-doc draft [<change>]`: Verify codebase context, challenge
       assumptions, right-size scope (tightening the spec and moving deferred
       slices to **Part 3: Follow-Up Items**, or splitting tasks into staged
       `CL 2a`, `CL 2b`, ...), author `./docs/design/changes/<change>.md` (based
       on `./docs/design/changes/TEMPLATE.md`), and stop for developer review.
     - `/design-doc audit [<change>]`: After developer review, spawn a
       repeatable isolated subagent (with no chat history and a narrow view of
       `<change>.md` and its referenced files) for blind comprehension and
       specification verification without overwriting the doc automatically.
     - `/design-doc revise [<change>]`: Iterate on inline `A:` notes or Gerrit
       review comments until CL 1 receives `+2`.
  2. **CL 2 (Implementation CL — `/implement-design`)**:
     - `/implement-design code [<change>]`: Implement the tasks in
       `<change>.md` (working in Bazel `--config=dev` first and syncing
       `CMakeLists.txt`, `refresh_compile_commands` in `BUILD.bazel`, and
       `README.md` once approved). Adapt flexibly to the codebase or propose
       better solutions as needed, provided you inform the developer and keep
       `./docs/design/changes/<change>.md` updated. Record out-of-scope
       follow-ups in **Part 3: Follow-Up Items** of `<change>.md` for Gerrit
       review.
     - `/implement-design audit [<change>]`: Spawn a repeatable isolated
       subagent (with no chat history) to audit `git diff` against
       `<change>.md` for spec drift, edge cases, test integrity, and code style
       / comment density before or during local review.
     - `/implement-design archive [<change>]`: Fold lasting architectural
       changes into `./docs/design/<component>.md`, archive the change doc via
       `git mv ./docs/design/changes/<change>.md
       ./docs/design/archive/<change>.md`, and run `./scripts/format-code.sh`.
     - `/implement-design file-followups [<change>]`: After CL 2 receives `+2`
       in Gerrit, file reviewer-approved Part 3 follow-ups into Buganizer under
       component `564546727` (with explicit user authorization).
- **Buganizer Component `564546727` (`issues` CLI)**:
  - When `/google/bin/releases/issues-cli/issues` is available, run it with
    `BypassSandbox: true`: use `issues readonly render <id>`,
    `issues readonly list-updates <id>`, and
    `issues readonly search "componentid:564546727 status:open <keywords>" --limit 5`
    for read-only lookups, and require explicit user authorization before
    creating tickets via
    `issues mutate create --component_id=564546727 --title="<title>" --priority=<P1|P2|P3> --type=BUG --description_file=./agent-temp/.bug_desc.md`.
  - If `/google/bin/releases/issues-cli/issues` does not exist on the machine,
    skip the CLI commands and either note in `<change>.md` that Buganizer was
    not searched (`draft`) or present the suggested tickets for manual filing
    (`file-followups`).
- **Documentation Tiers in `./docs/design/`**:
  - `./docs/design/INDEX.md`: Map of living component docs to source paths and
    passes.
  - `./docs/design/<component>.md`: Living component architecture at `HEAD`.
  - `./docs/design/changes/<change>.md`: Active, unmerged change designs/plans.
  - `./docs/design/archive/<change>.md`: Historical completed change plans
    (reference for past alternatives and trade-offs only; do not treat as
    current architecture).

## Third-party code
- Do NOT modify `third_party/iree`, `third_party/llvm-project`, or
  `third_party/iree/third_party/stablehlo` without explicit user authorization
  first.
- Try to avoid changes to `third_party/iree` and `third_party/llvm-project`. If
  you find an obvious bug in those directories, inform the user before fixing
  it.
- When changes to third-party code are necessary for new functionality, restrict
  them to minimal extension points or hooks, and implement the actual logic
  within this project.
- Root-level patch files (`iree-*.patch`, `llvm-project-*.patch`,
  `stablehlo-*.patch`) are applied with `./scripts/patch-third_party.sh`, so
  these submodules are expected to appear dirty in `git status` (do not revert
  or stage them). Keep `.patch` files small and scoped to a single upstreamable
  logical concern; split unrelated third-party changes into separate numbered
  patch files so they are easy to rebase or drop when upgrading upstream
  submodules.

## Build systems & Testing
- Unless you are explicitly asked to, use bazel.
- Always build, run, and test using `--config=dev`, unless explicitly asked to
  use a different config.
- Building from scratch can take a very long time; be careful not to invalidate
  bazel's cache (e.g. by omitting `--config=dev`), and NEVER run `bazel clean`.
- When developing a new feature, or fixing a bug, if changes to the build system
  are needed, initially work on bazel; only after we reach the final solution,
  approved by the user, do a similar change to cmake.
- Common Bazel commands:
  - Build compiler and runtime:
    `bazel build --config=dev @iree_core//tools:iree-compile @iree_core//tools:iree-run-module`
  - Run CI test suite: `bazel test --config=dev //tests:ci`
  - Run scoped model tests: `bazel test --config=dev //tests/models/linalg/...`
    or `//tests/models/stablehlo/...` (optionally filter by data type with
    `--test_tag_filters=f32,bf16,i8,i16,i32`).
  - Check / sync generated tests: `//tests/models/...` runs against static
    `.mlir` files under `generated_<type>/` and never regenerates them
    automatically. Verify sync with `bazel test --config=dev //tests:check_generated`
    and update with `bazel run --config=dev //tests:sync_generated` (or
    `//tests:copy_missing`).

## C/C++ code style
- Keep files, classes, and functions short, focused, and dedicated to a single,
  well-defined task. Do not mix unrelated responsibilities; decouple into
  separate components as needed.
- Organize `#include` directives into labeled groups separated by blank lines:
  the corresponding header first (in `.cpp` files), project headers,
  `// IREE:`, `// MLIR:`, `// LLVM:`, and standard library headers.
- Flatten control flow with early guard clauses (`if (!x) return failure();`,
  `if (!x) continue;`); if an `if` block ends with `return`, `break`, or
  `continue`, omit the `else` and avoid deeply nested conditionals.
- Lambda expressions should be kept short and used to write simple, inline
  functors with less boilerplate; prefer explicit captures if the lambda escapes
  the current scope, and default capture by reference only when its lifetime is
  obviously shorter than the captures.
- If a local function is inside an anonymous namespace, don't make it also static.
- Annotate non-obvious literal arguments (booleans, magic numbers, `nullptr`) at
  call sites with `/*paramName=*/` comments (e.g.,
  `setLoopUnrollAttr(forOp, /*fullUnroll=*/false)`).
- Prefer LLVM/MLIR ADT and error utilities (`FailureOr<T>`, `LogicalResult`,
  `ArrayRef`, `MutableArrayRef`, `StringRef`, `SmallVector`, `llvm::all_of`,
  `llvm::any_of`, `llvm::count_if`, `llvm::TypeSwitch`) over raw loops,
  out-parameters, or `std::vector` where appropriate.
- Document every function you add or modify. If a function is declared in a `.h`
  header file, place its documentation comment on the header declaration rather
  than the `.cc`/`.cpp` definition.
- In code you add or modify, include a brief comment for non-obvious conditions
  (`if`) and loops (`for`, `while`) explaining their rationale and intent (why,
  not what); omit comments when the code is self-explanatory.
- Always review and update existing comments in or near code you touch so they
  remain accurate and consistent with the updated implementation.

## MLIR & LLVM Development
- **Follow Existing IREE Patterns:** Before implementing new passes, rewrites, or lowerings, look for analogous cases in `third_party/iree` and follow their established conventions, idioms, and abstractions.
- **Prefer Interfaces & Inference Utilities Over Op Lists:** Use MLIR/IREE
  interfaces and analysis helpers (e.g., `TilingInterface`,
  `linalg::isaContractionOpInterface`, `linalg::inferContractionDims`,
  `iree_compiler::isComputeOp`, `TargetBackend::SupportedTypes`) instead of
  maintaining long hardcoded lists of concrete operation types.
- **Use Native Type Checking:** Prefer `llvm::isa<>` and `llvm::dyn_cast<>` over string comparisons for identifying operations, types, or attributes.
- **Utilize API Features:** Trust the framework's APIs before writing defensive
  boilerplate. For example, MLIR's `walk()` safely supports erasing the
  currently visited operation in-place (in post-order, or in pre-order when
  returning `WalkResult::skip()`) without needing a separate collection vector.
- **Diagnostics, Fallbacks, and Logging:** Keep normal compilation silent on
  `stdout` and `stderr` (see `README.md`):
  - **Errors (`emitError` / `emitOpError` + `signalPassFailure()`):** Use only
    when compilation cannot proceed (invalid IR, invalid pass/CLI options, or
    unsupported cases with no valid fallback).
  - **Warnings (`emitWarning`):** Reserve strictly for user-actionable issues.
    Never emit warnings (or `op->emitRemark(...)`, which also prints to `stderr`
    by default) for internal compiler fallbacks or heuristic choices.
  - **Graceful Fallbacks & Performance Degradation:** When an optimization,
    tiling strategy, or specialized lowering cannot be applied and a valid
    fallback path exists, recover gracefully and report high-level fallbacks via
    MLIR's opt-in remark engine (`mlir::remark::missed` / `mlir::remark::failed`
    from `mlir/IR/Remarks.h`), optionally paired with `LDBG()`
    (`llvm/Support/DebugLog.h`).
  - **Pattern Matching & Tracing:** Use `rewriter.notifyMatchFailure` for
    unmatched rewrite patterns and `LDBG()` / `LLVM_DEBUG` for detailed
    step-by-step pass tracing.
- **Handle Sub-Byte Types Carefully:** When computing byte sizes from element
  types, accumulate total bits first (`numElements * bitWidth`) and divide by 8
  only at the end using `llvm::divideCeil(bits, 8)` so sub-byte types (`i1`,
  `i2`, `i4`) are not truncated to zero.
- **Keep Cross-Component Predicates Synchronized:** When a capability check
  spans multiple stages (e.g., tile-size selection, IREE patch hooks, and
  codegen lowering), share a common utility or explicitly document the peer
  functions that must be kept in sync.
- **Preserve Semantic Guardrails:** Do not prematurely remove compiler directives (like unfoldable constants or optimization barriers) that protect intermediate passes from unwanted behavior like over-folding. If they need to be removed from the final output, do so in a dedicated final cleanup pass.

## Code Formatting & Repository Hygiene
- Update `README.md` whenever you add, rename, or remove user-facing
  `--coralnpu-*` compiler flags or runtime options.
- When adding new top-level C/C++ targets in Bazel, add them to
  `refresh_compile_commands` in `BUILD.bazel`.
- Format modified code using `./scripts/format-code.sh`.
- Prefix commit titles with a lowercase component tag in brackets, e.g.,
  `[compiler]`, `[runtime]`, `[tiling]`, `[tests]`, `[tools]`, `[bazel]`,
  `[cmake]`, or `[patches]`.
