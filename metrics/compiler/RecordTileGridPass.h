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

// Records the tile grid that the configuration pipeline chose for each
// dispatch in the metrics bundle.

#ifndef METRICS_COMPILER_RECORDTILEGRIDPASS_H_
#define METRICS_COMPILER_RECORDTILEGRIDPASS_H_

// MLIR:
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

// Standard library:
#include <memory>

namespace mlir::coralnpu_compiler {

// Returns a pass that records, in the `coralnpu.metrics` attribute of each
// dispatch function (MetricsAttr.h), the tile grid of the root op as
// `tile_grid {op, loops, workgroup, dtcm, vector, zvt_grid}`: the loop ranges
// of the root op and its workgroup, DTCM and vector tile sizes, one per loop.
// At each level, positive tile sizes are clamped to the tile size of the
// enclosing level, and a loop left untiled (`0`) inherits the tile size of the
// enclosing level. On Zvt targets, it also records `zvt {candidate}`: whether
// the root op is a Zvt contraction or convolution. For a candidate, `zvt_grid`
// is the number of 16x16 Zvt tiles of a vector tile along M and N. Runs at the
// end of the configuration pipeline, on dispatches whose root op is a static
// `TilingInterface` op with a CPU lowering configuration.
std::unique_ptr<InterfacePass<FunctionOpInterface>> createRecordTileGridPass();

// Registers `RecordTileGridPass` with the MLIR pass registry.
void registerRecordTileGridPass();

}  // namespace mlir::coralnpu_compiler

#endif  // METRICS_COMPILER_RECORDTILEGRIDPASS_H_
