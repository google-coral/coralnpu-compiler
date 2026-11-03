# DTCM Tiling and Promotion

This document gives an overview of how the CoralNPU compiler uses the DTCM (the
data tightly-coupled memory) for contractions. The details of the tiling levels
and of the passes are in [tiling_levels_explained.md](tiling_levels_explained.md).
The code TODOs for later changes have tags (e.g. `TODO(DMA)`).

## Summary

For an RVV contraction with reuse across both M and N (`linalg.matmul`,
`linalg.batch_matmul`, or a contraction `linalg.generic` with `M > 1` and
`N > 1`), the compiler:

1. picks a tile whose DTCM buffers fit in the DTCM tile budget;
2. copies the input tiles to DTCM;
3. computes the tile from DTCM;
4. computes each stored result into a DTCM staging buffer, and copies it to
   memory.

Each workgroup computes one DTCM tile. The copies are synchronous vector copies
for now; DMA comes later (`TODO(DMA)`). A Zvt contraction (`TODO(zvt)`) or a
zero-reuse contraction (`vecmat`, `matvec`, `dot`, where `M == 1` or `N == 1` so
each matrix element is read only once per reduction pass) gets DTCM tile sizes,
but no promotion (`hasContractionReuse` in
[CoralNPUTileSizeSelectionUtils.cpp](../../compiler/Transforms/CoralNPUTileSizeSelectionUtils.cpp)).

The DTCM pass decides between Zvt and RVV, but `CoralNPUMatrixCodegen` can also
lower the vector tiles of an RVV contraction with Zvt asm (e.g. the partial
tiles of 2 or 4 rows that the RVV strategy gives). Then the asm reads the DTCM
copies. If a fused operation reads the result (e.g. a bias), the asm drains the
accumulators to a scratch tile in DTCM, and the footprint counts it.

## Passes

```
  Step                         Pass (file)                          Result
  Strategy (IREE)              LLVMCPUSelectLoweringStrategyPass    pipeline, distribution, vector
                                                                    tile sizes
  DTCM tile sizes              CoralNPUTileSizeSelectionDTCM        distribution, cache_parallel,
                                                                    cache_reduction, Zvt/RVV vector
                                                                    tile sizes & pipeline,
                                                                    promote_operands (RVV only)
  Cache-level tiling (IREE)    LLVMCPU tiling passes                the loops of the DTCM tile
  Promotion                    CoralNPUPromoteOperands              DTCM buffers and copies
                               (afterCacheTilingHook, patch 0014)
  Vectorization and            LLVMCPU passes                       memref.alloc in DTCM (patch 0013),
  bufferization (IREE)                                              linalg.generic copies
  Copy lowering                CoralNPULowerDTCMCopies              loops of vector loads and stores
                               (pre-LLVM hook)
  Zvt codegen                  CoralNPUMatrixCodegen                Zvt asm, and scratch tiles in
                               (pre-LLVM hook)                      DTCM
  Local memory (IREE)          LLVMCPUAssignWorkgroupLocalMemory    one packed buffer, and the
                                                                    workgroup_local_memory size
```

The source files are in [compiler/Transforms](../../compiler/Transforms), and the
hooks are in [CoralNPUTargetBackend.cpp](../../compiler/Target/CoralNPUTargetBackend.cpp).

## DTCM Budget and Footprint

The DTCM tile budget is the DTCM size minus the stack and the reserve for static
data, rounded down to 4 KB: 24 KB for a 32 KB DTCM, and 1016 KB for a 1 MB DTCM.
The runtime gives local memory in pages of 4 KB, so a footprint that fits in the
budget also fits after the rounding to pages. The flags are in the
[README](../../README.md) ("DTCM tiling and stack").

The footprint of a tile is the sum of its buffers, each rounded up to 64 bytes:
the input slices that the fused operations read from memory, the output staging
buffers, an accumulator if the tile splits a reduction and the root result is
not stored directly, and the scratch tiles of the Zvt asm. A scratch tile has
`alignTo(M, 16) x N` accumulator elements for each distinct shape of the vector
tiles (with the remainders of loop peeling). `CoralNPUTileSizeSelectionFootprint`
computes the footprint, `CoralNPUPromoteOperands` creates the same buffers, and
`CoralNPUMatrixCodegen` creates the scratch tiles: keep the three in sync. See
[Level 1](tiling_levels_explained.md#2-level-1-cache-cacheparalleltiles--cachereductiontiles).

A root without promotion (a Zvt contraction, a zero-reuse contraction, a
convolution, or a reduction) reads its inputs one vector step of the reduction
loops at a time (or the full loop range if a reduction loop has no vector tile).
Thus its DTCM tile does not split these loops, and its input slices span one
step. The search shrinks the parallel loops first (outermost first, to exact
divisors of the loop range if it can), and then the reduction loops of a
promoted root.

## IR Shape

One workgroup tile of an i8 x i8 -> i32 matmul with a split K (chunk 128) and a
fused bias + `trunci` consumer, after the promotion (simplified):

```mlir
// In the workgroup loop body. The DTCM buffers come first.
%a_buf   = bufferization.alloc_tensor() {memory_space = #iree_codegen.workgroup_local} : tensor<32x128xi8>
%acc_buf = bufferization.alloc_tensor() {memory_space = #iree_codegen.workgroup_local} : tensor<32x64xi32>
%out_buf = bufferization.alloc_tensor() {memory_space = #iree_codegen.workgroup_local} : tensor<32x64xi8>
// (%b_buf and %bias_buf are the same.)
%init = linalg.fill ins(%c0_i32 : i32) outs(%acc_buf : tensor<32x64xi32>) -> tensor<32x64xi32>
%acc = scf.for %k = %c0 to %c512 step %c128 iter_args(%c = %init) -> (tensor<32x64xi32>) {
  %a_tile = tensor.extract_slice %a[%i, %k] [32, 128] [1, 1] : tensor<64x512xi8> to tensor<32x128xi8>
  %a_dtcm = bufferization.materialize_in_destination %a_tile in %a_buf
              : (tensor<32x128xi8>, tensor<32x128xi8>) -> tensor<32x128xi8>
  // (%b_dtcm is the same.)
  %r = linalg.matmul ins(%a_dtcm, %b_dtcm : tensor<32x128xi8>, tensor<128x64xi8>)
         outs(%c : tensor<32x64xi32>) -> tensor<32x64xi32>
  scf.yield %r : tensor<32x64xi32>
}
%res = linalg.generic {...bias add + trunci...} ins(%acc, %bias_dtcm : ...)
         outs(%out_buf : tensor<32x64xi8>) -> tensor<32x64xi8>
scf.forall.in_parallel {
  // The store copies the staged result from DTCM to memory when it bufferizes.
  tensor.parallel_insert_slice %res into %out[%i, %j] [32, 64] [1, 1] : tensor<32x64xi8> into tensor<64x128xi8>
}
```

- `materialize_in_destination` is not a linalg operation, so the vector-level
  tiling does not tile it, and the vectorizer does not fold it away. A
  `linalg.copy` does not work: the vectorizer folds the copy of a static tile,
  and bufferization then makes scalar copy loops.
- The buffers are static (the upper bounds of the tile sizes), at the start of
  the workgroup loop body. A partial tile uses a slice of its buffer, and loop
  peeling does not clone the buffers.
- IREE fuses a producer into the tile loop only through a slice of its result,
  and the tiling makes no slice of an input that no tiled loop indexes (e.g.
  the RHS when only M is tiled). The promotion first clones such elementwise
  producers (e.g. a dequantization) into the tile loop, so that their inputs
  get DTCM copies too.
- After bufferization, the copies are `linalg.generic` copies between memory and
  DTCM. `CoralNPULowerDTCMCopies` lowers them to loops of vector loads and
  stores along the rows. `TODO(DMA)`: this pass is where the copies must become
  DMA transfers.

## Runtime

1. The export gets the size of the packed buffer (`workgroup_local_memory`), and
   the executable library gives it as `local_memory_pages`.
2. The simulator runtime ([simulator_inline.c](../../runtime/sim/simulator_inline.c))
   allocates the local memory in the firmware heap (in DTCM), zeroes it, and
   writes its address and size to the dispatch request
   ([coralnpu_dispatch.h](../../crt/coralnpu_dispatch.h)).
3. The CRT ([coralnpu_main.c](../../crt/coralnpu_main.c)) gives them to each
   workgroup (`local_memory` and `local_memory_size` of the workgroup state).

The local memory does not keep its contents between workgroups.

## IREE Patches

The patches are at the repo root, and
[patch-third_party.sh](../../scripts/patch-third_party.sh) applies them.

```
  Patch   Name                                         Change
  0011    llvmcpu-riscv-register-space                 Limit the vector tile sizes of RISC-V
                                                       matmuls to the register file (32 x VLEN).
  0012    llvmcpu-matmul-batch-vector-sizes            Give the batch loops of matmuls the
                                                       vector size 1.
  0013    codegen-cpu-allocate-workgroup-local-memory  Allocate workgroup_local buffers with
                                                       memref.alloc in the entry block.
  0014    llvmcpu-add-after-cache-tiling-hook          A target hook after the cache-level tiling.
  0015    llvmcpu-peel-outer-loops-first               LLVMCPUPeel peels the outer loops first.
```

## Tests

- Lit tests in [tests/transforms](../../tests/transforms):
  - `tile_size_selection_dtcm.mlir`: the tile sizes, the budget, the footprint
    (also with and without the Zvt scratch tiles), and `promote_operands`;
  - `tile_size_selection_vector.mlir`: Zvt and RVV vector tile size selection,
    pipeline selection, and shape/divisor rules under the default DTCM budget;
  - `promote_operands.mlir`: the IR of the promotion;
  - `promote_operands_pipeline.mlir`: the `workgroup_local_memory` size after the
    whole pipeline (also with the Zvt scratch tiles), and the vector loops of
    the copies;
  - `lower_dtcm_copies.mlir`: the copy lowering.
- IREE lit tests for the patches (e.g. `bufferize_workgroup_local_memory.mlir`
  for patch 0013, and `peel.mlir` for patch 0015).
- Simulator tests in [tests/models/linalg](../../tests/models/linalg):
  `matmul_bias_relu_trunc_i8` (bias, output staging, and a split K) and
  `dequant_matmul_f32` (narrow input tiles), together with the other
  contraction tests and the models in [tests/examples](../../tests/examples).

## Limits

The main limits of the current implementation:

- The copies are synchronous vector copies (`TODO(DMA)`).
- Zvt contractions get no promotion (`TODO(zvt)`).
- The footprint model copies a part of the rules of `CoralNPUMatrixCodegen` to
  count its scratch tiles, and it does not count the scratch tile of the BF16
  vecmat path. One decision for Zvt or RVV would remove the copy (`TODO(zvt)`).
- The tile search gives a large matmul a thin tile with a full K, so the reuse
  in DTCM is low (`TODO(search)`).
- A copy of a narrow input (e.g. i8 weights before a fused dequantization)
  keeps the narrow type, so the inner loop dequantizes each element again for
  each use (`TODO(promotion)`).
- Only the `DoubleTilingExpert` pipeline (`addMultiTilingExpertPassPipeline`)
  calls the promotion hook, and convolutions get DTCM tile sizes but no
  promotion (`TODO(scope)`).

## Upstream Mapping

```
  Now (CoralNPU)                             Upstream
  DTCM pass after the strategy pass          A generic "fit to local memory" step after
                                             setLoweringConfigForComputeOps
  promote_operands (extra config key)        promote_operands/promotion_types in
                                             #iree_cpu.lowering_config
  CoralNPUPromoteOperands (hook 0014)        LLVMCPUPromoteOperandsPass and a promotion
                                             attribute interface
  cpuAllocationFn (patch 0013)               The same patch (a bug fix)
  Budget flags                               The local memory size in the
                                             #hal.executable.target config
  CoralNPULowerDTCMCopies                    Vectorized memref copies in LLVMCPU, and a
                                             #coralnpu.dma promotion attribute for DMA
```
