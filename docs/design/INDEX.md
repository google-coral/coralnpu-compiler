# CoralNPU Compiler Living Component Documents Index

This index maps the living component architecture documents in `docs/design/`
to the source directories and compiler passes they cover.

- **For Agents (`/design-doc` and `/implement-design`)**: Read this index first
  to locate the relevant living component doc(s) before scanning raw C++ source
  files. When adding a new `<component>.md` document or expanding the scope of
  an existing one during `/design-doc draft` (and verifying during
  `/implement-design archive`), update the table below.

| Component Document                                         | Primary Source Paths & Passes                                                                                             | Scope & Contents                                                                                                                                                      |
| :--------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [dtcm_tiling.md](dtcm_tiling.md)                           | `compiler/Transforms/CoralNPUTileSizeSelection*.cpp`, `CoralNPUPromoteOperands.cpp`, `CoralNPULowerDTCMCopies.cpp`        | DTCM budget and footprint estimation, contraction reuse gating, operand promotion, and vectorized DTCM copies.                                                        |
| [matrix_codegen.md](matrix_codegen.md)                     | `compiler/Transforms/CoralNPUMatrixCodegen.cpp`, `compiler/Transforms/CoralNPUTileSizeSelectionDTCM.cpp`                  | Direct Zvt (Matrix) extension codegen (`1x1`, `1x2`, `1x4`, `2x2`, and partial tiles, CSR setup, microkernels).                                                       |
| [tiling_levels_explained.md](tiling_levels_explained.md)   | `compiler/Transforms/CoralNPUTileSizeSelection*.cpp` (`DTCM`, `Footprint`, `RVV`, `Utils`), `CoralNPUPromoteOperands.cpp` | Multi-level IREE/CoralNPU tiling hierarchy (Distribution, DTCM/Cache, Vector/Register), RVV vector tile and multi-op reduction refinement, and alignment computation. |
| [metrics.md](metrics.md)                                   | `metrics/runtime/`, `runtime/driver/coralnpu_run_hooks.h`, `crt/coralnpu_dispatch.h`                                      | Runtime dispatch statistics (`coralnpu.stats.v1`), firmware CSR counters, ELF memory footprint, and stack peak.                                                       |
