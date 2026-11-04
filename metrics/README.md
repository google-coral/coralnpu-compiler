# CoralNPU metrics

This directory holds the code that measures how CoralNPU runs a workload. The
compiler (`compiler/`), the runtime (`runtime/`) and the firmware (`crt/`) keep
only generic hooks, and the code here implements them.

| Directory   | Content                                                           |
| ----------- | ----------------------------------------------------------------- |
| `compiler/` | Per-executable metrics bundle (`--coralnpu-dump-metrics-format`). |
| `runtime/`  | Per-dispatch execution statistics (`--coralnpu_stats_format`).    |
| `tests/`    | Tests of the metrics.                                             |

The dependencies go one way: the metrics code depends on the hook interfaces
of the core code, for example `//runtime/driver:run_hooks`, and the core code
does not depend on the metrics code. The exceptions are the two registrations,
which install the metrics: the driver registration
(`//runtime/driver/registration`) installs the statistics writer as the
dispatch observer of the devices, and the compiler plugin
(`compiler/Target/CoralNPUTarget.cpp`) passes the metrics hooks to the target
backend.

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
