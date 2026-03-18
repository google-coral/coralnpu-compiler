#pragma once

#include "mlir/Pass/PassManager.h"

namespace mlir::coralnpu_compiler {

void buildTOSAInputConversionPassPipeline(OpPassManager &passManager);

void registerTOSAConversionPasses();

} // namespace mlir::coralnpu_compiler
