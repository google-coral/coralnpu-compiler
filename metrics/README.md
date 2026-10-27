# CoralNPU metrics

This directory holds the code that measures how CoralNPU runs a workload. The
compiler (`compiler/`), the runtime (`runtime/`) and the firmware (`crt/`) keep
only generic hooks, and the code here implements them.

| Directory  | Content                                                        |
| ---------- | -------------------------------------------------------------- |
| `runtime/` | Per-dispatch execution statistics (`--coralnpu_stats_format`). |
| `tests/`   | Tests of the metrics.                                          |

The dependencies go one way: the metrics code depends on the hook interfaces
of the core code, for example `//runtime/driver:run_hooks`, and the core code
does not depend on the metrics code. The one exception is the driver
registration (`//runtime/driver/registration`), which installs the statistics
writer as the dispatch observer of the devices.

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
