# CoralNPU Device Topology and Affinity Placement

This document explains how the CoralNPU compiler plugin and standard IREE
cooperate to partition a model across a host CPU (`local` / `llvm-cpu`) and a
CoralNPU accelerator (`coralnpu`) on a unified-memory SoC. For CLI flag usage,
see [README.md](../../README.md) ("Affinity placement" and "Affinity execution
profile report").

## 1. Summary

When compiling for both a host CPU and CoralNPU, standard IREE handles device
registration, operator fusion, dispatch region formation, and stream scheduling.
The CoralNPU plugin supplies the unified-memory device topology, the
per-dispatch hardware eligibility and roofline placement policy, and optional
per-affinity execution profile reporting. Standard IREE then partitions the
annotated dispatches into per-device `stream.cmd.execute` regions, synchronizes
device transitions with semaphores while eliding buffer copies across shared
DDR, and compiles each executable for its target backend (`llvm-cpu` or
`coralnpu`).

## 2. Compilation Pipeline

The table below shows the chronological handoff between standard IREE phases and
the passes that `CoralNPUSession`
([CoralNPUTarget.cpp](../../compiler/Target/CoralNPUTarget.cpp)) injects into
the pipeline:

| Phase                | Owner           | Pass / Hook                                                          | IR / Attribute Handoff                                                                                   |
| :------------------- | :-------------- | :------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------- |
| `DeviceAssignment`   | IREE            | `AssignTargetDevicesPass` & `MaterializeTargetDevicesPass`           | Creates `@__device_0` (host) and `@__device_1` (`coralnpu`) globals from `--iree-hal-target-device`.     |
| `Preprocessing`      | CoralNPU plugin | `CoralNPUMaterializeDeviceTopology`                                  | Sets `stream.topology` (unified memory) and `stream.affinity.default` (`@__device_0`) on `moduleOp`.     |
| `GlobalOptimization` | IREE            | `FoldGlobalsPass` & optimization passes                              | Preserves `@__device_1` referenced by `stream.topology` on `moduleOp` (§3.2).                            |
| `DispatchCreation`   | IREE            | `FormDispatchRegionsPass` & `ConvertDispatchRegionsToWorkgroupsPass` | Forms fused `flow.dispatch.workgroups` ops without per-dispatch `stream.affinity`.                       |
| `DispatchCreation`   | CoralNPU plugin | `CoralNPUAffinityAnnotation` (hook from patch `0010`)                | Sets `stream.affinity = #hal.device.affinity<@__device_N>` on each `flow.dispatch.workgroups`.           |
| `Flow` / `Stream`    | IREE            | `Stream` affinity & partitioning passes                              | Partitions dispatches into per-device `stream.cmd.execute` ops; elides copies across unified memory.     |
| `Stream` (end)       | CoralNPU plugin | `CoralNPUDumpAffinityExecutionProfile` (patch `0010`)                | Optional (`--coralnpu-dump-affinity-profile-format`): reports dispatches, bytes, and work per affinity.  |
| `HAL`                | IREE + plugin   | `CoralNPUTargetBackend` & `LinkingUtils` (patch `0009`)              | Compiles each executable for its target backend; links only executables owned by a single backend.       |

The pass implementations live in
[compiler/Transforms/](../../compiler/Transforms/).

## 3. Device Resolution & Topology Materialization

### 3.1 Device Discovery (`DeviceUtils`)

Both topology materialization and affinity placement resolve target devices
through `lookupDeviceAffinities`
([DeviceUtils.cpp](../../compiler/Transforms/DeviceUtils.cpp)). The helper runs
standard IREE's `HAL::DeviceAnalysis` on the module and scans the analyzed
device globals:

- **CoralNPU Affinity & Target Config**: Records the first device global whose
  target has `device_id == "coralnpu"`, together with its first executable
  target's configuration dictionary (`coralnpuTargetConfig`, which carries
  `+zvtbase` and `+zvl<N>b` `VLEN`).
- **Host Fallback Affinity**: Prefers the module's existing
  `stream.affinity.default` (via IREE's `Stream::AffinityAttr::lookupOrDefault`)
  when present and distinct from CoralNPU, or falls back to the first
  non-CoralNPU device global.
- **Host-Only Guard (`targetsOnlyHost`)**: Global preprocessing passes
  (`CoralNPURaiseApplyScale` and `CoralNPUPromoteBF16Accumulator`) call
  `targetsOnlyHost` ([DeviceUtils.h](../../compiler/Transforms/DeviceUtils.h))
  to detect modules that target only a host CPU. Host-only compilations skip
  raising 64-bit `tosa.apply_scale` expansions and preserve dynamic BF16
  accumulator shapes. Because `targetsOnlyHost` is a module-wide check in
  `Preprocessing` (before per-dispatch placement), multi-device compilations
  still apply both rewrites to operations later placed on the CPU.

### 3.2 Topology Materialization (`CoralNPUMaterializeDeviceTopology`)

When a module targets both CoralNPU and a host device,
`CoralNPUMaterializeDeviceTopology`
([CoralNPUMaterializeDeviceTopology.cpp](../../compiler/Transforms/CoralNPUMaterializeDeviceTopology.cpp))
attaches two module-level attributes if they are not already present:

- `stream.topology`: An `IREE::HAL::DeviceTopologyAttr` with bidirectional
  `unified_memory = true, transparent_access = true` links between the host and
  CoralNPU devices. Standard IREE's `Stream` pipeline reads this attribute to
  access shared DDR buffers in place across devices without inserting staging
  copies.
- `stream.affinity.default`: Sets the host device as the module's default
  affinity so host-bound operations and runtime housekeeping default to the CPU.

Running this pass during `Preprocessing` is required because no operations
reference `@__device_1` (`coralnpu`) until the end of `DispatchCreation`.
Without the `stream.topology` reference on `moduleOp`, standard IREE's
`FoldGlobalsPass` in `GlobalOptimization` would erase `@__device_1` as unused.

## 4. Dispatch Eligibility (`CoralNPUAffinityAnnotation`)

`CoralNPUAffinityAnnotation`
([CoralNPUAffinityAnnotation.cpp](../../compiler/Transforms/CoralNPUAffinityAnnotation.cpp))
runs at the end of `DispatchCreation`, after standard IREE has fused producer
and consumer operations into `flow.dispatch.workgroups` regions. Waiting until
dispatch regions are formed gives the pass the final fused operation body and
exact dispatch boundary I/O tensors without maintaining custom attribute
propagation across earlier IREE passes.

For each `flow.dispatch.workgroups` operation without an existing
`stream.affinity` attribute, the pass collects all compute operations in the
workgroup body via IREE's `isComputeOp` and identifies the primary operation via
IREE's `getRootOperation`. A dispatch is eligible for CoralNPU (`canRunOnNPU`)
when its root operation satisfies three conditions:

1. **Linalg Root**: The root operation implements `linalg::LinalgOp`. Non-Linalg
   compute roots (such as `iree_linalg_ext.sort` or `linalg.softmax` before it
   decomposes in target configuration) are routed to the host.
2. **Static Shapes & Supported Types**: Every operand and result of the root
   operation has a static shape and uses a scalar or element type in
   `getCoralNPUSupportedTypes` (`bf16`, `f32`, `index`, `i8`, `i16`, or `i32` in
   [Utils.cpp](../../compiler/Target/Utils.cpp)). Dispatches with dynamic shapes
   or unsupported types (such as `i64` or `f16`) stay on the host.
3. **Single-Combiner Reductions**: If the root operation has reduction loops,
   every yielded result must come directly from a single combiner operation
   (recognized by MLIR's `linalg::getCombinerOpKind`) that is the sole user of
   its region output argument (`isScalarReduction`). Reductions that yield
   coupled select chains (such as `argmax`, which pairs `arith.cmpf` with two
   `arith.select` ops) cannot be vectorized by IREE's CPU vectorizer. Because
   they would lower to a scalar loop on CoralNPU, they stay on the host even
   when `--coralnpu-roofline-speedup-threshold=0`.

If the root operation is eligible for CoralNPU but a fused non-root compute
operation uses an unsupported type, the pass emits a warning (`non-root linalg
operation has types not supported by CoralNPU in a dispatch eligible for
CoralNPU`).

When a module targets only CoralNPU, every eligible dispatch is placed on
CoralNPU and ineligible dispatches are left unannotated. When both devices are
present, ineligible dispatches receive the host affinity and eligible dispatches
are scored by the roofline model below.

## 5. Roofline Latency & Hop Refinement Model

### 5.1 Latency Equations & Hardware Constants

`preferCoralNPU`
([CoralNPUAffinityAnnotation.cpp](../../compiler/Transforms/CoralNPUAffinityAnnotation.cpp))
models a unified-memory SoC pairing an embedded Arm Cortex-A55 host CPU with
CoralNPU at 800 MHz (such as the Synaptics SL2619 Coralboard). It places an
eligible dispatch on CoralNPU when $T_{\text{cpu}} \ge \text{threshold} \cdot
T_{\text{npu}}$ (`--coralnpu-roofline-speedup-threshold`, default `1.0`; `0`
places all eligible dispatches on CoralNPU):

$$
T_{\text{npu}} = T_{\text{launch}} + H_{\text{npu}} \cdot T_{\text{hop}} +
  \max\!\left(\frac{\text{Ops}}{P_{\text{npu}}},\,
              \frac{\text{Bytes}}{B_{\text{npu}}}\right)
$$

$$
T_{\text{cpu}} = H_{\text{cpu}} \cdot T_{\text{hop}} +
  \max\!\left(\frac{\text{Ops}}{P_{\text{cpu}}},\,
              \frac{\text{Bytes}}{B_{\text{cpu}}}\right)
$$

Here $P_{\text{npu}}$ and $P_{\text{cpu}}$ are the effective compute rates
obtained by multiplying each device's 32-bit base peak by its precision scale
and tile utilization:

| Parameter                                 | CoralNPU (800 MHz)                                         | Host CPU (Cortex-A55)                             | Notes                                                    |
| :---------------------------------------- | :--------------------------------------------------------- | :------------------------------------------------ | :------------------------------------------------------- |
| **32-bit Base Peak**                      | **Zvt**: 102.4 GFLOPS (`16x16`) / **RVV**: 12.8 GFLOPS     | 8.0 GFLOPS                                        | 64 Zvt MAC PEs; 2 RVV FALUs/MACs at `VLEN = 128`.        |
| **Precision Scale**                       | $2\times$ (16-bit), $4\times$ (8-bit; RVV ints only)       | $2\times$ (16-bit), $4\times$ (8-bit), $1\times$ (`bf16`) | Zvt/CPU: narrowest `ins`; RVV: widest operand.           |
| **Tile Utilization**                      | **Zvt**: $\prod_{d \in \{M, N\}} d / \text{alignTo}(d, 16)$ | $1.0$                                             | $16 \times 16$ fill fraction on `outs[0]` (`zvtTileUtilization`). |
| **Memory Bandwidth ($B$)**                | 12.8 GB/s                                                  | 4.0 GB/s                                          | CoralNPU 128-bit `m_axi` peak at 800 MHz.                |
| **Dispatch Launch ($T_{\text{launch}}$)** | 800 ns                                                     | 0 ns                                              | Host-to-NPU dispatch launch overhead.                    |
| **Cross-Device Hop ($T_{\text{hop}}$)**   | 5,000 ns (5 $\mu$s)                                        | 5,000 ns (5 $\mu$s)                               | Semaphore wait per unique cross-device neighbor.         |

- **Zvt vs. RVV Path**: A dispatch uses the Zvt base peak (`64 * 2 * 0.8 =
  102.4` GFLOPS) when the CoralNPU target enables `+zvtbase` and
  `isZvtMatrixContractionShape`
  ([CoralNPUTileSizeSelectionUtils.cpp](../../compiler/Transforms/CoralNPUTileSizeSelectionUtils.cpp))
  accepts the root operation: (a) a contraction or convolution with no
  reduction-loop Linalg consumer in the dispatch, (b) a signed/FP MAC body and
  `f32`/`bf16 -> f32` or `i8`/`i16 -> i32` element types on the loaded source
  tensors via `isZvtReadableInputs` (looking through a single-consumer 1:1
  `arith.extf`/`arith.extsi` producer), (c) a zero `linalg.fill` or
  `dispatch.tensor.load` accumulator init (so contractions initialized with a
  broadcast bias fall back to RVV), and (d) $M \ge 3$, $N \ge 8$
  trailing-dimension tile alignment constraints (see
  [matrix_codegen.md](matrix_codegen.md)). Otherwise it uses the RVV base peak,
  $2 \cdot (\text{VLEN} / 32) \cdot 2 \cdot 0.8$ GFLOPS (`12.8 GFLOPS` at
  `VLEN = 128`). Both peak formulas include a factor of `2` (multiply +
  accumulate per MAC) to match `estimateComputeOps` counting both `mul` and
  `add` instructions in contraction bodies.
- **Precision Scaling Details**: Zvt and the host CPU select their tier from the
  narrowest integer or float element type among the root operation's DPS inputs
  (`getDpsInputs`, where Zvt looks through a fused 1:1
  `arith.extf`/`arith.extsi` producer via `getZvtLoadSource(input)`); if any
  DPS input is `bf16`, the host CPU stays at $1\times$ because it lacks a
  native BF16 extension. RVV instead inspects all operands of the root
  operation (`getOperandTypes`, which includes both DPS `ins` and `outs`
  initializers) and takes the widest element width while treating all
  floating-point types as 32-bit, so an `i8 -> i32` RVV reduction runs at the
  $1\times$ 32-bit RVV rate.
- **Zvt Tile Utilization (`zvtTileUtilization`)**: Multiplies $d /
  \text{alignTo}(d, 16)$ across the trailing two dimensions ($M$ and $N$) of the
  root operation's first DPS init tensor (`getDpsInits()[0]`).

### 5.2 Compute Operations, I/O Bytes & Streaming Cap

- **Compute Operations (`estimateComputeOps`)**: Sums the product of static loop
  ranges across every `linalg::LinalgOp` in the dispatch body, multiplied by the
  number of non-cast operations in each Linalg block (at least 1). Thus an
  $M \times N \times K$ contraction body (`mul` + `add`) contributes $2MNK$
  operations while a single-op elementwise body contributes $1\times$ its loop
  volume; the total `Ops` across the entire dispatch (including fused
  elementwise epilogues) are divided by the root operation's single
  $P_{\text{npu}}$ and $P_{\text{cpu}}$ rates. Operations implementing
  `CastOpInterface` (such as `arith.extf` and `arith.extsi`) are skipped so
  mixed-precision widening does not inflate the operation count.
- **I/O Bytes (`estimateIOBytes`)**: Sums the byte sizes of all
  `flow.dispatch.workgroups` input arguments and results. `estimateBytesForType`
  multiplies total elements by element bit width first and divides by 8 with
  `llvm::divideCeil` at the end so sub-byte types are never truncated to zero.
- **Long-Stream I/O Cap**: When a positive speedup threshold is configured, any
  dispatch whose total I/O exceeds **64 MiB** (for Zvt contractions and
  convolutions) or **16 MiB** (for RVV dispatches) is kept on the host. Above
  these thresholds (tuned on Gemma 4), the 128-bit `m_axi` peak overstates
  sustained throughput across the blocking `DBus2Axi` bridge.

### 5.3 Cross-Device Hops & Three-Pass Refinement

Because the host and CoralNPU share DDR (`unified_memory = true`), function
arguments, constants, and `util.global.load` weights require no device transfer
and cost zero hops. Only adjacent `flow.dispatch.workgroups` operations placed
on the *other* device incur a 5 $\mu$s synchronization hop (`npuHops` when
evaluating CoralNPU placement, `cpuHops` when evaluating host placement).

To count hops, `preferCoralNPU` walks both operand producers and result users of
the dispatch with a worklist, looking transparently through IREE's
`flow.tensor.reshape` and `flow.tensor.update` metadata/slice ops and
deduplicating neighbor dispatches:

1. **Pass 1 (Greedy Forward Placement)**: Walks `flow.dispatch.workgroups`
   operations in program order. Each eligible dispatch is placed using the
   affinities of its upstream producers (downstream consumers have no
   `stream.affinity` yet).
2. **Pass 2 (Reverse Refinement) & Pass 3 (Forward Refinement)**: Once all
   dispatches have an initial `stream.affinity`, the pass re-evaluates eligible
   dispatches in reverse order and then forward order with both producer and
   consumer hops populated.
3. **Hop-Reduction Guard**: During Passes 2 and 3, if a dispatch's current
   device already incurs no more cross-device hops than switching to the other
   device, the dispatch keeps its current affinity immediately. It re-runs the
   roofline latency comparison only when switching devices would strictly reduce
   the number of cross-device hops.

## 6. IR Example

In `@chain`
([affinity_annotation.mlir](../../tests/transforms/affinity_annotation.mlir)), a
$64 \times 256 \times 1$ GEMV (`~2.8x` standalone speedup) falls below
`--coralnpu-roofline-speedup-threshold=3`. When its result feeds a large
$128 \times 128 \times 128$ matmul on CoralNPU (`@__device_1`) through
`flow.tensor.reshape` and `flow.tensor.update`, avoiding the 5 $\mu$s
cross-device hop raises its effective speedup to `~3.6x` and keeps the chain on
CoralNPU. Conversely, the same GEMV fed by a host-only `iree_linalg_ext.sort`
(`@__device_0`) drops to `~1.5x` and stays on the host:

```mlir
module attributes {
  stream.affinity.default = #hal.device.affinity<@__device_0>,
  stream.topology = #hal.device.topology<links = [
    <@__device_0 -> @__device_1 = {transparent_access = true, unified_memory = true}>,
    <@__device_1 -> @__device_0 = {transparent_access = true, unified_memory = true}>
  ]>
} {
  // GEMV producer kept on CoralNPU (@__device_1) after reverse hop refinement:
  %pre = flow.dispatch.workgroups(...) ... attributes {
    stream.affinity = #hal.device.affinity<@__device_1>
  }
  %pre_1d = flow.tensor.reshape %pre : tensor<64x1xf32> -> tensor<64xf32>
  %upd_1d = flow.tensor.update %pre_1d, %arg0_1d[0] : tensor<64xf32> -> tensor<16384xf32>
  %lhs    = flow.tensor.reshape %upd_1d : tensor<16384xf32> -> tensor<128x128xf32>

  // Large matmul placed on CoralNPU (@__device_1, ~12.6x speedup):
  %mm = flow.dispatch.workgroups(%lhs, ...) ... attributes {
    stream.affinity = #hal.device.affinity<@__device_1>
  }
  // ... (a second GEMV consuming %mm via flow.tensor.reshape also stays on @__device_1)

  // Non-Linalg sort root is ineligible for CoralNPU and placed on host (@__device_0):
  %sorted = flow.dispatch.workgroups(...) ... attributes {
    stream.affinity = #hal.device.affinity<@__device_0>
  }
  // Consumer GEMV of %sorted stays on host (@__device_0) at threshold = 3 (~1.5x):
  %post = flow.dispatch.workgroups(%sorted, ...) ... attributes {
    stream.affinity = #hal.device.affinity<@__device_0>
  }
}
```

## 7. Affinity Execution Profile Reporting (`CoralNPUDumpAffinityExecutionProfile`)

When `--coralnpu-dump-affinity-profile-format={pretty|verbose|csv|json}` is set,
`CoralNPUDumpAffinityExecutionProfile`
([CoralNPUDumpAffinityExecutionProfile.cpp](../../compiler/Transforms/CoralNPUDumpAffinityExecutionProfile.cpp))
runs at the end of the `Stream` pipeline (`extendStreamPassPipeline`) and writes
a per-affinity execution summary to `--coralnpu-dump-affinity-profile-file`
(defaulting to `stderr` when empty, or `stdout` for `-`):

- **Device Name Resolution**: Walks every `stream.cmd.execute` region, looks up
  each operation's `IREE::Stream::AffinityAttr`, and resolves its `@__device_N`
  symbol to the underlying `util.global` `device_id` (`coralnpu`, `local`,
  `/`-joined `DeviceSelectAttr` targets, or `Default` (`Default (CPU)` in
  `pretty`/`verbose`) when unannotated).
- **Dispatch, Data Size & Work Accounting**: Counts `stream.cmd.dispatch`,
  `stream.cmd.fill`, and `stream.cmd.copy` operations per affinity. For each
  exported dispatch function with static binding shapes, `analyzeFunc` sums the
  element counts and byte sizes (`divideCeil(numElements * bitWidth, 8)`) of all
  `stream.binding.subspan` operations and multiplies them by the function's IR
  operation count (`opCount`) to report `Estimated Static Work (Op-Bytes)` and
  `Estimated Static Work (Op-Elems)` alongside the count of dynamic dispatches.

## 8. IREE Patches & Verification

### 8.1 IREE Patches

Root-level patches applied by
[patch-third_party.sh](../../scripts/patch-third_party.sh):

| Patch  | Name                                              | Purpose                                                                                                    |
| :----- | :------------------------------------------------ | :--------------------------------------------------------------------------------------------------------- |
| `0009` | `codegen-only-link-exclusively-owned-executables` | Makes `gatherExecutablesForTarget` require all variants to match `targetName`, preserving export ordinals. |
| `0010` | `pluginapi-add-pipeline-extension-hooks`          | Adds `extendDispatchCreationPassPipeline` and `extendStreamPassPipeline` hooks to IREE's `PluginSession`.  |

### 8.2 Tests

- Lit tests in [tests/transforms/](../../tests/transforms/):
  - [affinity_annotation.mlir](../../tests/transforms/affinity_annotation.mlir):
    Verifies FP32/BF16/INT16/INT8 matmuls, partial Zvt tile utilization,
    cast-free mixed-precision op counting, reshape/update hop chains, and
    launch-bound small matmuls. Also covers unsupported types (`i64`), dynamic
    shapes, non-Linalg roots (`iree_linalg_ext.sort`), non-single-combiner
    reductions (`argmax`), global weights, the 16 MiB I/O cap, `threshold = 0`,
    and CoralNPU-only (`CHECK-NO-HOST`) compilation.
  - [dump_affinity_execution_profile.mlir](../../tests/transforms/dump_affinity_execution_profile.mlir)
    and
    [dump_affinity_execution_profile_empty.mlir](../../tests/transforms/dump_affinity_execution_profile_empty.mlir):
    Verify `pretty`, `json`, and `csv` affinity profile reports at the end of
    the `Stream` pipeline.
  - [target_options.mlir](../../tests/transforms/target_options.mlir): Verifies
    CLI validation for `--coralnpu-roofline-speedup-threshold`.
- Multi-device PJRT integration in
  [pjrt_plugin/client.cc](../../pjrt_plugin/client.cc): Compiles uncommitted
  multi-device programs with both `--iree-hal-target-device=local` and
  `--iree-hal-target-device=coralnpu` so `CoralNPUAffinityAnnotation` partitions
  dispatches across the CPU and CoralNPU.

## TODOs

- [ ] **Calibrate Analytical Roofline Constants**: Calibrate the roofline
  compute rates, memory bandwidths, launch overhead, and cross-device hop
  penalty in `preferCoralNPU` against profiled hardware and RTL dispatches
  (`CoralNPUAffinityAnnotation.cpp`).
- [ ] **Support `linalg.softmax` (`AggregatedOpInterface`) and Non-`LinalgOp`
  Compute Roots**: Because `CoralNPUAffinityAnnotation` runs in
  `DispatchCreation` before `DecomposeSoftmaxPass` (which runs during target
  configuration) and requires a `linalg::LinalgOp` root, undecomposed
  `linalg.softmax` dispatches are routed to the host in multi-device
  compilations (`CoralNPUAffinityAnnotation.cpp`).
- [ ] **Move `ConvolutionToIGEMM` After Affinity Placement**:
  `CoralNPUSession::extendPreprocessingPassPipeline` runs IREE's
  `ConvolutionToIGEMMPass` with a filter matching `*coralnpu*` on per-op
  `stream.affinity`. Because `CoralNPUAffinityAnnotation` runs later in
  `DispatchCreation` and `MaterializeTargetDevicesPass` names device globals
  `@__device_N`, this filter only matches pre-annotated lit tests (see
  [matrix_codegen.md](matrix_codegen.md#20-convolutiontoigemm-preprocessing)).
- [ ] **Check Non-Root Compute Operations for Eligibility**: Only the root
  operation's operands and shape select the precision scaling tier and Zvt vs.
  RVV rate; fused non-root compute operations with unsupported types emit a
  warning rather than disqualifying the dispatch from CoralNPU placement
  (`CoralNPUAffinityAnnotation.cpp`).
- [ ] **Align Roofline Zvt Eligibility and I/O Traffic with DTCM Tiling**:
  `preferCoralNPU` checks `isZvtMatrixContractionShape` and counts each dispatch
  tensor once in `estimateIOBytes`. During codegen, target configuration
  restores $C_{\text{in}} = 1$ convolutions in
  `CoralNPURestoreConvInputChannelPass`, `CoralNPUTileSizeSelectionDTCM` falls
  back to RVV when a $16 \times 16$ Zvt tile exceeds the DTCM budget
  (`DTCMFootprintModel`), and tiled execution re-reads unpromoted or multi-tile
  contraction operands across workgroups (`CoralNPUAffinityAnnotation.cpp`,
  `CoralNPURestoreConvInputChannel.cpp`, `CoralNPUTileSizeSelectionDTCM.cpp`).
