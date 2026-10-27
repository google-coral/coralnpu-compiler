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
  patches, or changes >150 LOC), suggest starting the two-CL workflow with
  `/design-doc draft <change>` (or `/design-doc draft-from-commits [<commits>]
  <change>` for existing commits; see `./docs/design/README.md`).
- **Documentation Tiers in `./docs/design/`**:
  - `./docs/design/INDEX.md`: Map of living component docs to source paths and
    passes (check this first when exploring a compiler subsystem).
  - `./docs/design/<component>.md`: Living component architecture at `HEAD`.
  - `./docs/design/changes/<change>.md`: Active, unmerged change design briefs.
  - `./docs/design/archive/<change>.md`: Historical completed change briefs
    (reference for past trade-offs only; do not treat as current architecture).
- **Buganizer Component `564546727` (`issues` CLI)**:
  - When `/google/bin/releases/issues-cli/issues` is available, run it (passing
    `BypassSandbox: true` if present in the tool schema) for read-only lookups
    (`issues readonly render|list-updates|search`), and require explicit user
    authorization before creating tickets
    (`issues mutate create --component_id=564546727 ...`). If `issues` is
    missing or fails with a LOAS/ACL restriction (`RPC_RESTRICTIONS_VIOLATION`
    or `LabelACLRejected`), treat Buganizer as unavailable and proceed without
    it.

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
- Do NOT add new third-party dependencies (`http_archive`, submodules, or CMake
  packages) under `restricted` (e.g., GPL, LGPL, AGPL) or `reciprocal` (e.g.,
  MPL, EPL) licenses, or link existing `restricted` targets (`@libftdi1`,
  `@libusb`) into release packages (`//build_tools/bazel:dist_tar` or Python
  wheels), without explicit user authorization and flagging the license/linking
  implications. Set `package(licenses = [...])` in `*.BUILD` files to match the
  upstream license category (`"notice"` for Apache-2.0/MIT/BSD, `"restricted"`
  for GPL/LGPL, `"reciprocal"` for MPL/EPL).

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
    automatically. Verify sync with
    `bazel test --config=dev //tests:check_generated` and update with
    `bazel run --config=dev //tests:sync_generated` (or `//tests:copy_missing`).

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
- If a local function is inside an anonymous namespace, don't make it also
  static.
- Annotate non-obvious literal arguments (booleans, magic numbers, `nullptr`) at
  call sites with `/*paramName=*/` comments (e.g.,
  `setLoopUnrollAttr(forOp, /*fullUnroll=*/false)`).
- Prefer LLVM/MLIR ADT and error utilities (`FailureOr<T>`, `LogicalResult`,
  `ArrayRef`, `MutableArrayRef`, `StringRef`, `SmallVector`, `llvm::all_of`,
  `llvm::any_of`, `llvm::count_if`, `llvm::TypeSwitch`) over raw loops,
  out-parameters, or `std::vector` where appropriate.
- Document every function you add or modify. If a function is declared in a `.h`
  header file, place its documentation comment on the header declaration rather
  than the `.cc`/`.cpp` definition; if it is already documented in a TableGen
  (`.td`) definition (such as pass constructors in `Passes.h` mirroring
  `Passes.td`), keep the documentation in the `.td` file and do not duplicate it
  in `.h` or `.cpp`.
- In code you add or modify, include a brief comment for non-obvious conditions
  (`if`) and loops (`for`, `while`) explaining their rationale and intent (why,
  not what); omit comments when the code is self-explanatory.
- Always review and update existing comments in or near code you touch so they
  remain accurate and consistent with the updated implementation.

## MLIR & LLVM Development

- **Follow Existing IREE Patterns:** Before implementing new passes, rewrites,
  or lowerings, look for analogous cases in `third_party/iree` and follow their
  established conventions, idioms, and abstractions.
- **Prefer Interfaces & Inference Utilities Over Op Lists:** Use MLIR/IREE
  interfaces and analysis helpers (e.g., `TilingInterface`,
  `linalg::isaContractionOpInterface`, `linalg::inferContractionDims`,
  `iree_compiler::isComputeOp`, `TargetBackend::SupportedTypes`) instead of
  maintaining long hardcoded lists of concrete operation types.
- **Use Native Type Checking:** Prefer `llvm::isa<>` and `llvm::dyn_cast<>` over
  string comparisons for identifying operations, types, or attributes.
- **Utilize API Features & Upstream Utilities:** Trust the framework's APIs
  before writing defensive boilerplate, and inspect `#include`d MLIR/IREE
  `*Utils.h` and `*Interfaces.h` headers (e.g.,
  `mlir/Dialect/Utils/StaticValueUtils.h`, `mlir/Dialect/Utils/IndexingUtils.h`,
  `mlir/Dialect/Linalg/Utils/Utils.h`, `iree/compiler/Codegen/Utils/Utils.h`,
  `iree/compiler/Codegen/Utils/CPUUtils.h`) before writing or moving local
  helper functions. For example, MLIR's `walk()` safely supports erasing the
  currently visited operation in-place (in post-order, or in pre-order when
  returning `WalkResult::skip()`) without needing a separate collection vector,
  and `StaticValueUtils.h` provides batch utilities like `getConstantIntValues`
  and `getOffsetsSizesAndStrides`.
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
- **Preserve Semantic Guardrails:** Do not prematurely remove compiler
  directives (like unfoldable constants or optimization barriers) that protect
  intermediate passes from unwanted behavior like over-folding. If they need to
  be removed from the final output, do so in a dedicated final cleanup pass.
- **Use Worklists for Def-Use and Graph Walks:** Traverse producer-consumer or
  def-use chains iteratively with an `llvm::SmallVector` worklist (`push_back` /
  `pop_back_val`) and a visited set (`llvm::SmallPtrSet` or
  `llvm::SmallDenseSet`) rather than recursion, avoiding call-stack growth on
  deep operation chains.

## Code Formatting & Repository Hygiene

- Update `README.md` whenever you add, rename, or remove user-facing
  `--coralnpu-*` compiler flags or runtime options.
- When adding new top-level C/C++ targets in Bazel, add them to
  `refresh_compile_commands` in `BUILD.bazel`.
- Format modified code using `./scripts/format-code.sh`.
- Prefix commit titles with a lowercase component tag in brackets, e.g.,
  `[compiler]`, `[runtime]`, `[tiling]`, `[tests]`, `[tools]`, `[bazel]`,
  `[cmake]`, or `[patches]`.
