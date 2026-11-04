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

// The options of the compiler metrics (--coralnpu-dump-metrics-*).

#ifndef METRICS_COMPILER_METRICSOPTIONS_H_
#define METRICS_COMPILER_METRICSOPTIONS_H_

// IREE:
#include "iree/compiler/Utils/OptionUtils.h"

// LLVM:
#include "llvm/Support/CommandLine.h"

// Standard library:
#include <string>

namespace mlir::coralnpu_compiler {

// Output format of the compiler metrics bundle.
enum class MetricsFormat {
  None,
  Pretty,
  Json,
};

// The options of the compiler metrics: whether to write a metrics bundle per
// executable, in which format, and where.
struct MetricsOptions {
  MetricsFormat format = MetricsFormat::None;
  // Directory of the bundle files; "-" writes to stdout and an empty
  // directory to stderr.
  std::string dir;

  // Binds the options to their flags.
  void bindOptions(iree_compiler::OptionsBinder &binder) {
    static llvm::cl::OptionCategory category("CoralNPU Metrics");
    binder.opt<MetricsFormat>(
        "coralnpu-dump-metrics-format", format, llvm::cl::cat(category),
        llvm::cl::desc("Dumps the per-executable compiler metrics bundle "
                       "(coralnpu.metrics.v1) in the specified format."),
        llvm::cl::values(
            clEnumValN(MetricsFormat::None, "none", "Disabled."),
            clEnumValN(MetricsFormat::Pretty, "pretty", "Pretty printed."),
            clEnumValN(MetricsFormat::Json, "json", "JSON.")));
    binder.opt<std::string>(
        "coralnpu-dump-metrics-dir", dir, llvm::cl::cat(category),
        llvm::cl::desc("Directory to write the metrics bundles to, one file "
                       "per executable (<executable>.metrics.{json,txt}). "
                       "'-' for stdout, empty for stderr."));
  }

  // Returns whether the metrics bundle is requested.
  bool enabled() const { return format != MetricsFormat::None; }
};

}  // namespace mlir::coralnpu_compiler

#endif  // METRICS_COMPILER_METRICSOPTIONS_H_
