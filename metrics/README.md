# CoralNPU metrics

This directory holds the code that measures how CoralNPU runs a workload. The
compiler (`compiler/`), the runtime (`runtime/`) and the firmware (`crt/`) keep
only generic hooks, and the code here implements them.

| Directory   | Content                                                           |
| ----------- | ----------------------------------------------------------------- |
| `compiler/` | Per-executable metrics bundle (`--coralnpu-dump-metrics-format`). |
| `runtime/`  | Per-dispatch execution statistics (`--coralnpu_stats_format`).    |
| `tools/`    | Workload report, which merges the bundles and the statistics.     |
| `tests/`    | Tests of the metrics.                                             |

The dependencies go one way: the metrics code depends on the hook interfaces
of the core code, for example `//runtime/driver:run_hooks`, and the core code
does not depend on the metrics code. The exceptions are the two registrations,
which install the metrics: the driver registration
(`//runtime/driver/registration`) installs the statistics writer as the
dispatch observer of the devices, and the compiler plugin
(`compiler/Target/CoralNPUTarget.cpp`) passes the metrics hooks to the target
backend.

## Workload report

The workload report (`metrics/tools/report.py`) merges the compiler metrics
bundles and the runtime statistics of a workload, and reports these metrics
per dispatch and for the whole workload:

- **Cycles:** the cycles of each run, and of its kernel (the workgroup loop),
  which the firmware measures.
- **ITCM and DTCM use:** the use of the firmware ELF (bundle), the heap of the
  dispatch data and, with `--coralnpu_stats_collect=stack`, the stack peak.
- **Tile grid:** the workgroup, DTCM and vector tile sizes of the root op, and
  its Zvt grid (bundle).
- **Static Zvt rate:** the Zvt candidates that the compiler lowered to Zvt
  (bundle).

The `coralnpu_workload_report` Bazel macro (`metrics/tools/defs.bzl`)
compiles a workload with its bundles, runs it on MPACT with the statistics,
and writes the report to the test outputs. For example:

```shell
bazel test --config=dev //metrics/tests/report:mlp_f32
cat bazel-testlogs/metrics/tests/report/mlp_f32/test.outputs/report.md
```

Without Bazel, the same report takes three commands:

```shell
iree-compile --iree-hal-target-device=coralnpu \
  --coralnpu-dump-metrics-format=json --coralnpu-dump-metrics-dir=- \
  model.mlir -o model.vmfb > model.metrics.json
iree-run-module --module=model.vmfb --device=coralnpu --simulator=mpact \
  --coralnpu_stats_format=json --coralnpu_stats_file=stats.jsonl \
  --coralnpu_stats_collect=stack --input=...
python3 metrics/tools/report.py --compiler-metrics model.metrics.json \
  --runtime-stats stats.jsonl
```

`--format=json` writes the report as one `coralnpu.report.v1` JSON document,
whose keys the docstring of `report.py` describes. The report of the MLP
starts with a summary of the workload:

```markdown
- Static Zvt rate: 2 of 2 Zvt candidates lowered to Zvt (100.0%)
- mpact: 2 dispatch runs, 0 failed, 2169 cycles, 1589 in the kernels
```

Keep these limits in mind:

- MPACT is a functional simulator: it executes one instruction per cycle.
  Verilator gives the cycles of the RTL.
- The DTCM tile sizes fit the working set of a tile in the DTCM, and promoted
  RVV contractions and Zvt scratch tiles are allocated in the DTCM heap (in 4 KB
  pages), whereas unpromoted bindings stay in DDR.
- The `coralnpu-zvt` missed remarks of the compiler
  (`--remarks-filter=coralnpu-zvt`, see the top-level `README.md`) tell why a
  Zvt candidate is not lowered to Zvt.

## Compiler metrics

The target backend calls its hooks (`compiler/Target/CoralNPUTargetHooks.h`)
as it compiles each executable. The metrics hooks
(`metrics/compiler/MetricsTargetHooks.h`) fill a bundle
(`metrics/compiler/MetricsBundle.h`) and write it:

- at the end of the configuration pipeline and right after the matrix codegen,
  metrics passes record facts about each dispatch in the `coralnpu.metrics`
  attribute of its function (`metrics/compiler/MetricsAttr.h`): the tile grid of
  the root op and whether it is a Zvt candidate
  (`metrics/compiler/RecordTileGridPass.h`), and whether the matrix codegen
  lowered it to Zvt (`metrics/compiler/RecordZvtLoweringPass.h`);
- when the backend serializes the executable, the bundle gets the target, the
  dispatches with their `coralnpu.metrics` entries, and the ITCM and DTCM use of
  the linked firmware ELF (`metrics/compiler/MetricsElf.h`).

The `--coralnpu-dump-metrics-*` flags in the top-level `README.md` select the
format and the directory.

## Runtime statistics

The HAL device passes every dispatch to its dispatch observer
(`runtime/driver/coralnpu_dispatch_observer.h`). The statistics writer
(`metrics/runtime/stats_writer.h`) is that observer: for each dispatch, it
creates a collector (`metrics/runtime/stats_collector.h`) whose run hooks
(`runtime/driver/coralnpu_run_hooks.h`) the execution backend calls around the
run of the dispatch. The writer then writes the statistics that the collector
collected:

- the cycles of the run, and the cycles and retired instructions of the kernel,
  which the firmware measures and reports in its dispatch request
  (`crt/coralnpu_dispatch.h`);
- the memory footprint: the ITCM and DTCM that the firmware ELF uses
  (`firmware_memory.h`), the heap that the dispatch data uses and, as an opt-in,
  the peak stack use (`stack_peak.h`).

The `--coralnpu_stats_*` flags in the top-level `README.md` select the output
format, the output file and the opt-in collectors.
