# Add Workload Metrics Report, Bazel Macro, and Metrics Guide

- **Author(s)**: Shaked Flur
- **Component Doc(s)**: [metrics.md](../metrics.md) *(updated in chained CL 2)*
- **Bug(s)**: None (searched component 2161972; no open issues found)

---

## 1. Summary & Motivation

Compile-time bundles (`coralnpu.metrics.v1` from `metrics/compiler/`) and
runtime dispatch statistics (`coralnpu.stats.v1` from `metrics/runtime/`) are
written to separate files by `iree-compile` and `iree-run-module`. Answering
why a workload took a given number of cycles or how much TCM, heap, stack, and
Zvt lowering coverage it achieved requires running both tools manually and
cross-referencing dispatch names across JSON files. In addition,
`coralnpu_bytecode_module` in `build_tools/bazel/bytecode_module.bzl` only
declares the `.vmfb` output, so automated tests cannot capture compile-time
bundles alongside simulator runs.

This change adds an offline report merger (`metrics/tools/report.py`), a
simulator test runner (`metrics/tools/workload_report.py`), and the
`coralnpu_workload_report` Bazel macro (`metrics/tools/defs.bzl`), and extends
`metrics/README.md` with the workload report guide.

- **Goals**:
  - Extend `coralnpu_bytecode_module` (`build_tools/bazel/bytecode_module.bzl`)
    with an optional `metrics_name` output capturing `coralnpu.metrics.v1`
    bundles.
  - Add `metrics/tools/report.py` (`//metrics/tools:report`,
    `//metrics/tools:report_lib`) to join `coralnpu.metrics.v1` bundles and
    `coralnpu.stats.v1` JSON Lines traces on `(executable, name)` and emit a
    Markdown report (`--format=pretty`) or `coralnpu.report.v1` JSON
    (`--format=json`).
  - Add `metrics/tools/workload_report.py` (`//metrics/tools:workload_report`)
    and `coralnpu_workload_report` (`metrics/tools/defs.bzl`) to compile an MLIR
    workload with bundles, run it on MPACT with stack peak collection, and write
    `report.md`, `report.json`, and `stats_<simulator>.jsonl` to the test
    outputs directory, failing only on execution or missing-metrics errors.
  - Add unit tests (`//metrics/tests/tools:report_test`), an end-to-end MLP
    report test in CI (`//metrics/tests/report:mlp_f32`), and docs in
    `metrics/README.md` and `README.md`.
- **Non-Goals**:
  - Enforcing cycle-count or memory-usage regression thresholds in CI tests, or
    collecting dynamic instruction mix / Zvt execution rates from simulator
    traces (both tracked in **Section 4**).
  - Modifying C/C++ compiler passes in `metrics/compiler/` or runtime C code in
    `metrics/runtime/`.

---

## 2. Proposed Change & Component Doc Delta

Section 3 of [`metrics.md`](../metrics.md) documents the post-change workload
reporting pipeline, schemas, and invariants:

- **Bazel Compile-Time Bundle Capture ([`metrics.md` §3.2](../metrics.md))**:
  When `metrics_name` is set on `coralnpu_bytecode_module`
  (`build_tools/bazel/bytecode_module.bzl`), the genrule dumps JSON compiler
  metrics bundles to `stdout` and redirects them to `metrics_name`. Normal
  builds without `metrics_name` remain unchanged and silent.
- **Offline Report Merger & Completeness Invariants
  ([`metrics.md` §3.1–3.2](../metrics.md))**:
  `metrics/tools/report.py` loads concatenated or directory-based compiler
  bundles and runtime JSON Lines traces and joins them on `(executable, name)`
  from the embedded ELF library header, attaching each executable's static
  `itcm_used` and `data_used` footprint to its dispatches (falling back to the
  runtime stats record and raising `ValueError` if they disagree). It emits
  `coralnpu.report.v1` JSON or space-aligned `## Dispatches` and `## Tile grid`
  Markdown tables. `summary.zvt = {candidates, lowered_candidates, rate}`
  (`lowered_candidates / candidates`, omitted when `candidates == 0`) counts
  only dispatches with `candidate == true`; `{candidate: false, lowered: true}`
  dispatches are excluded from `summary.zvt` and rendered as `no/yes` in
  `## Dispatches`. Per-run averages (`{cycles,instret}_per_run`) are omitted
  unless every successful run reports that counter, and sums
  (`cycles.{total,kernel}`, `instret.kernel`) are omitted unless every
  constituent run reports that counter. Unrun bundled dispatches show `n/a`
  runtime columns; runtime-only dispatches (`in_bundle = False`) are appended
  and flagged as missing compiler metrics.
- **Simulator Runner & `coralnpu_workload_report` Macro
  ([`metrics.md` §3.2](../metrics.md))**:
  `coralnpu_workload_report` (`metrics/tools/defs.bzl`) wraps
  `coralnpu_bytecode_module` and a `native_test` running
  `//metrics/tools:workload_report` on the configured `simulators` (default
  `["mpact"]`). Before each simulator run, `workload_report.py` removes any
  stale `stats_<simulator>.jsonl` in `--output-dir` and strips ambient
  `CORALNPU_SIMULATOR` and `CORALNPU_STATS_*` variables. On exit `0` or `1`, it
  writes `report.md` and `report.json` (even if a run fails); it exits with `0`
  (clean), `1` (simulator execution, simulator stats read/parse, or
  `find_problems` errors), or `2` (unreadable/malformed `--compiler-metrics`
  bundles or bundle/stats footprint mismatches; standalone `report.py` exits
  with `2` on any unreadable or inconsistent bundle or stats input).

### Alternatives Considered

| Option                                                                 | Approach                                                                                                    | Pros                                                                                                                                      | Cons                                                                                                                  |
| :--------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------- |
| **Option 1 (Chosen)**<br/>*`[Lens: Modularity]` + `[Lens: Precedent]`* | Standalone Python merger (`report.py`) + simulator test runner (`workload_report.py`) + Bazel `native_test` | Preserves one-way `metrics/` layer boundaries; supports both offline CLI inspection of saved JSON/JSONL files and cached Bazel test runs. | Requires two small Python entry points (`report.py` and `workload_report.py`).                                        |
| **Option 2**<br/>*`[Lens: Minimalist]`*                                | Load compiler bundles and emit the merged report directly inside `metrics/runtime/` C at device teardown    | Single binary (`iree-run-module`) emits the combined report without a Python post-processor.                                              | Couples the C runtime driver to compiler bundle schemas and adds a JSON parser to `metrics/runtime/`.                 |
| **Option 3**<br/>*`[Lens: Shifter]`*                                   | Monolithic Bazel-only test script or build-time `genrule` that compiles, simulates, and formats in one step | Fewer files in `metrics/tools/`.                                                                                                          | Cannot re-format saved `.metrics.json` and `.jsonl` files offline without re-running the simulator.                   |

**Decision Rationale**: Option 1 keeps `metrics/compiler/` and
`metrics/runtime/` decoupled from each other's schemas and lets developers run
`report.py` standalone on saved files. It also keeps simulation inside `bazel
test` (rather than `bazel build`) while writing `report.md` and `report.json` to
`$TEST_UNDECLARED_OUTPUTS_DIR` even on test failure.

---

## 3. Implementation & Verification Plan

- [ ] **Step 1 (`build_tools/bazel/bytecode_module.bzl`)**: Add optional
  `metrics_name = None` output to `coralnpu_bytecode_module` dumping JSON
  compiler metrics bundles to `$(execpath <metrics_name>)`.
- [ ] **Step 2 (`metrics/tools/report.py`, `metrics/tools/BUILD.bazel`)**:
  Implement `coralnpu.report.v1` loading (`load_bundles`, `read_stats`),
  merging (`build_report`), validation (`find_problems`), Markdown formatting
  (`format_pretty`), and CLI (`//metrics/tools:report_lib`,
  `//metrics/tools:report`).
- [ ] **Step 3 (`metrics/tools/workload_report.py`, `metrics/tools/defs.bzl`)**:
  Implement the simulator runner (`//metrics/tools:workload_report`) and the
  `coralnpu_workload_report` Bazel macro.
- [ ] **Tests (`metrics/tests/tools/*`, `metrics/tests/report/*`,
  `tests/BUILD.bazel`)**: *Positive*: Verify multi-run aggregation, memory
  maxima, static Zvt rates, Markdown formatting, and bundle loading in
  `//metrics/tests/tools:report_test`, and end-to-end MLP report generation in
  `//metrics/tests/report:mlp_f32` (`bazel test --config=dev
  //metrics/tests/tools:ci //metrics/tests/report:ci`). *Negative*: Verify
  partial sums and per-run averages are omitted when a run lacks a counter, and
  `find_problems` flags missing bundles/stats, unbundled dispatches, failed
  runs, and stack overflows.
- [ ] **Build, Doc & Archive Sync (`metrics/README.md`, `README.md`,
  `docs/design/metrics.md`, `docs/design/INDEX.md`)**: Document the workload
  report workflow, Bazel macro, 3-command CLI usage, and caveats; sync
  `docs/design/metrics.md` and `docs/design/INDEX.md`, `git mv` this doc to
  `docs/design/archive/`, and run `./scripts/format-code.sh`.

### Implementation Notes & Adaptations

- *(None yet)*

---

## 4. Follow-Up Items (For Reviewer Approval -> Buganizer `2161972`)

- [ ] **Add dynamic instruction mix and runtime-weighted Zvt execution rate to
      runtime stats and workload reports** (`P2` | `metrics/runtime/`,
      `metrics/tools/report.py`):
      Complement the compile-time `summary.zvt.rate` with runtime Zvt coverage
      weighted by dispatch executions/kernel cycles and simulator
      instruction-mix counters (retired Zvt matrix instructions vs. RVV vector
      and scalar instructions). This lets reports show both static lowering
      coverage and dynamic execution share.
- [ ] **Add optional performance and memory threshold checks to
      `workload_report.py`** (`P3` | `metrics/tools/workload_report.py`,
      `metrics/tools/defs.bzl`):
      Support opt-in cycle, DTCM/ITCM, and Zvt lowering rate assertions for
      pinned benchmark targets while keeping default `coralnpu_workload_report`
      targets failure-free on performance numbers.
