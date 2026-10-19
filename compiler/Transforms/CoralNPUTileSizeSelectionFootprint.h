// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef COMPILER_TRANSFORMS_CORALNPUTILESIZESELECTIONFOOTPRINT_H_
#define COMPILER_TRANSFORMS_CORALNPUTILESIZESELECTIONFOOTPRINT_H_

// MLIR:
#include "mlir/IR/Operation.h"

// LLVM:
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::coralnpu_compiler {

// Estimates the DTCM footprint of one DTCM tile of a dispatch: the sum of the
// DTCM buffers that the tile of the root op and of the compute ops fused with
// it need. Each buffer is rounded up to 64 bytes:
// - an input slice for each tensor that the fused ops read from memory (e.g. a
//   binding), in its type in memory (e.g. i8 before a fused extsi), times the
//   number of buffers for each input slice (e.g. 2 for double buffering);
// - an output staging buffer for each result that leaves the fused ops (e.g. a
//   result that is stored to a binding), in the stored type;
// - an accumulator for the root result, if the DTCM tile splits a reduction
//   loop and the root result has no output staging buffer;
// - the scratch tiles of the Zvt asm: CoralNPUMatrixCodegen can lower the
//   vector tiles of a root contraction or convolution to Zvt asm, and it
//   drains the accumulators of a root result that a fused op reads to scratch
//   tiles in DTCM.
// The footprint of a root that is not a linalg op (e.g. pack or unpack) counts
// only the operands of the root.
// CoralNPUPromoteOperands creates the input, output staging and accumulator
// buffers, and CoralNPUMatrixCodegen the scratch tiles: keep the three in sync.
class DTCMFootprintModel {
 public:
  // `computeOps` are the compute ops of the dispatch, which includes
  // `rootOp`. `buffersPerInput` is the number of DTCM buffers for each input
  // slice.
  DTCMFootprintModel(Operation *rootOp, ArrayRef<Operation *> computeOps,
                     int64_t buffersPerInput);

  // Returns the estimated footprint in bytes for the DTCM tile sizes
  // `tileSizes` of the root loops. A tile size of 0, or of the full loop
  // range, does not tile the loop. `vectorParallelSizes` and
  // `vectorReductionSizes` are the vector tile sizes of the root loops (0 for
  // a loop without a vector tile); the scratch tiles, and the temporaries of an
  // unpack root, depend on them. If `streamReduction`, the root reads its
  // inputs one vector step of the reduction loops at a time (e.g. a root
  // without promotion), so the input slices span one step of these loops.
  int64_t estimate(ArrayRef<int64_t> tileSizes,
                   ArrayRef<int64_t> vectorParallelSizes,
                   ArrayRef<int64_t> vectorReductionSizes,
                   bool streamReduction) const;

 private:
  Operation *rootOp;
  llvm::SmallPtrSet<Operation *, 8> computeOps;
  int64_t buffersPerInput;
};

}  // namespace mlir::coralnpu_compiler

#endif  // COMPILER_TRANSFORMS_CORALNPUTILESIZESELECTIONFOOTPRINT_H_
