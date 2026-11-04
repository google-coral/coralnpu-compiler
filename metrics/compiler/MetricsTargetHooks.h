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

// The compiler metrics of the CoralNPU target backend: a metrics bundle per
// executable (MetricsBundle.h), which the hooks of the backend
// (compiler/Target/CoralNPUTargetHooks.h) fill as it serializes the
// executable.

#ifndef METRICS_COMPILER_METRICSTARGETHOOKS_H_
#define METRICS_COMPILER_METRICSTARGETHOOKS_H_

#include "compiler/Target/CoralNPUTargetHooks.h"
#include "metrics/compiler/MetricsOptions.h"

// Standard library:
#include <memory>

namespace mlir::coralnpu_compiler {

// Target facts that the metrics bundle takes from the CoralNPU options.
struct MetricsTargetConfig {
  int dtcmSizeKb = 0;
  int numVectorRegisters = 0;
};

// Returns the hooks that write the metrics bundle of each executable as
// |options| request, or null if they request none.
std::shared_ptr<CoralNPUTargetHooks> createMetricsTargetHooks(
    const MetricsOptions &options, const MetricsTargetConfig &config);

}  // namespace mlir::coralnpu_compiler

#endif  // METRICS_COMPILER_METRICSTARGETHOOKS_H_
