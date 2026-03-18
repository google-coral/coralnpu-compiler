#include "Pipelines.h"

#include "compiler/plugins/input/TOSA/InputConversion/Passes.h"

#include "iree/compiler/Preprocessing/Common/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir::iree_compiler;

namespace mlir::coralnpu_compiler {
namespace {

void registerTOSAConversionPassPipeline() {
  PassPipelineRegistration<> tosa(
      "coralnpu-tosa-input-transformation-pipeline",
      "Runs the TOSA IREE flow dialect transformation pipeline, with "
      "specializations for CoralNPU",
      [](OpPassManager &passManager) {
        buildTOSAInputConversionPassPipeline(passManager);
      });
}

} // namespace

void buildTOSAInputConversionPassPipeline(OpPassManager &passManager) {
  // TODO(sflur): do any specialzed lowering from TOSA
  // passManager.addNestedPass<func::FuncOp>(createTosaToPreFlowConversionPass());
  // passManager.addPass(mlir::createCanonicalizerPass());

  // use the standard tosa pipeline
  iree_compiler::buildTOSAInputConversionPassPipeline(passManager);

  passManager.addNestedPass<func::FuncOp>(
      iree_compiler::Preprocessing::createMakeSingleDispatchForFunctionPass());
}

void registerTOSAConversionPasses() {
  // Passes
  // TODO

  // Pipelines.
  registerTOSAConversionPassPipeline();
}

} // namespace mlir::coralnpu_compiler
