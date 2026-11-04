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

#include "compiler/Target/CoralNPUTargetBackend.h"
#include "metrics/compiler/MetricsOptions.h"
#include "metrics/compiler/MetricsTargetHooks.h"

// IREE:
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/PluginAPI/Client.h"

// MLIR:
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

// LLVM:
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "coralnpu-target"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace mlir::coralnpu_compiler {

// The options of the plugin: those of the target backend and those of the
// compiler metrics (metrics/compiler).
struct CoralNPUPluginOptions {
  CoralNPUOptions target;
  MetricsOptions metrics;

  // Binds the options to their flags.
  void bindOptions(OptionsBinder &binder) {
    target.bindOptions(binder);
    metrics.bindOptions(binder);
  }
};

namespace {

struct CoralNPUTargetDevice final : public IREE::HAL::TargetDevice {
  CoralNPUTargetDevice(const CoralNPUOptions & /*options*/) {}

  IREE::HAL::DeviceTargetAttr getDefaultDeviceTarget(
      MLIRContext *context,
      const IREE::HAL::TargetRegistry &targetRegistry) const override {
    Builder b(context);
    auto configAttr = b.getDictionaryAttr({});

    SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
    targetRegistry.getTargetBackend("coralnpu")
        ->getDefaultExecutableTargets(context, "coralnpu", configAttr,
                                      executableTargetAttrs);

    return IREE::HAL::DeviceTargetAttr::get(context,
                                            b.getStringAttr("coralnpu"),
                                            configAttr, executableTargetAttrs);
  }
};

struct CoralNPUSession
    : public PluginSession<CoralNPUSession, CoralNPUPluginOptions,
                           PluginActivationPolicy::DefaultActivated> {
  static void registerPasses() {
    registerCoralNPUPasses();
    registerMetricsPasses();
  }

  // Registers the CoralNPU HAL target device.
  void populateHALTargetDevices(IREE::HAL::TargetDeviceList &targets) override {
    targets.add("coralnpu", [=]() {
      return std::make_shared<CoralNPUTargetDevice>(options.target);
    });
  }

  // Registers the CoralNPU HAL target backend and wires its metrics hooks.
  virtual void populateHALTargetBackends(
      IREE::HAL::TargetBackendList &targets) override {
    targets.add("coralnpu", [=]() {
      // The metrics observe the backend through its hooks, if requested.
      MetricsTargetConfig metricsConfig;
      metricsConfig.dtcmSizeKb = options.target.dtcmSizeKb;
      metricsConfig.numVectorRegisters = options.target.numVectorRegisters;
      return std::make_shared<CoralNPUTargetBackend>(
          options.target,
          createMetricsTargetHooks(options.metrics, metricsConfig));
    });
  }

  virtual void onRegisterDialects(DialectRegistry &registry) override {
    // TODO:
  }

  LogicalResult onActivate() override {
    LLVM_DEBUG(llvm::dbgs() << "Coral plugin activated\n");
    if (failed(options.target.validate(context))) {
      return failure();
    }
    return success();
  }

  // Materializes stream.topology in Preprocessing so @__device_1 is referenced
  // on moduleOp and not erased by FoldGlobalsPass during GlobalOptimization.
  void extendPreprocessingPassPipeline(OpPassManager &passManager) override {
    passManager.addPass(createCoralNPUMaterializeDeviceTopologyPass());

    // Run before GlobalOptimization's integer optimizations and before the
    // affinity annotation estimates the cost of each dispatch.
    passManager.addPass(createCoralNPURaiseApplyScalePass());

    // Run in global preprocessing so host and device agree and the trailing
    // cast is generalized and fused into its producer during DispatchCreation.
    passManager.addPass(createCoralNPUPromoteBF16AccumulatorPass());

    // Matches per-op affinity to a device named *coralnpu*, which only the lit
    // test has. Enabling it needs earlier placement and a conv allowlist:
    // depthwise/dilated im2col fails to lower.
    passManager.nest<IREE::Util::FuncOp>().addPass(createConvolutionToIGEMMPass(
        /*configFn=*/std::nullopt, [](Operation *op) {
          auto affinity = op->getAttrOfType<IREE::HAL::DeviceAffinityAttr>(
              "stream.affinity");
          return affinity &&
                 affinity.getDevice().getLeafReference().getValue().contains(
                     "coralnpu");
        }));
  }

  // Adds passes to the |buildDispatchCreationPassPipeline| pipeline at the end.
  void extendDispatchCreationPassPipeline(OpPassManager &passManager) override {
    passManager.addPass(createCoralNPUAffinityAnnotationPass(
        {options.target.rooflineSpeedupThreshold}));
  }

  // Adds the affinity profile dump at the end of the Stream pipeline, the
  // earliest point at which stream affinities are resolved.
  void extendStreamPassPipeline(OpPassManager &passManager) override {
    if (options.target.dumpAffinityProfileFormat == DumpOutputFormat::None) {
      return;
    }
    passManager.addPass(createCoralNPUDumpAffinityExecutionProfilePass(
        {options.target.dumpAffinityProfileFormat,
         options.target.dumpAffinityProfileFile}));
  }
};

}  // namespace
}  // namespace mlir::coralnpu_compiler

IREE_DEFINE_COMPILER_OPTION_FLAGS(
    mlir::coralnpu_compiler::CoralNPUPluginOptions);

extern "C" bool iree_register_compiler_plugin_coralnpu(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<mlir::coralnpu_compiler::CoralNPUSession>(
      "coralnpu");
  return true;
}
