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
#include "compiler/Transforms/Passes.h"

// IREE headers
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/PluginAPI/Client.h"

// MLIR headers
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

// LLVM headers
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "coralnpu-target"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace mlir::coralnpu_compiler {

namespace {

llvm::cl::opt<int64_t> clAffinityIOMinThersholdBytes(
    "coralnpu-affinity-io-min-threshold-bytes",
    llvm::cl::desc(
        "Minimum estimated dispatch input/output bytes required before "
        "placing an operation on CoralNPU."),
    llvm::cl::init(0));

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
    : public PluginSession<CoralNPUSession, CoralNPUOptions,
                           PluginActivationPolicy::DefaultActivated> {
  static void registerPasses() { registerCoralNPUPasses(); }

  void populateHALTargetDevices(IREE::HAL::TargetDeviceList &targets) override {
    targets.add("coralnpu", [=]() {
      return std::make_shared<CoralNPUTargetDevice>(options);
    });
  }

  virtual void populateHALTargetBackends(
      IREE::HAL::TargetBackendList &targets) override {
    targets.add("coralnpu", [=]() {
      return std::make_shared<CoralNPUTargetBackend>(options);
    });
  }

  virtual void onRegisterDialects(DialectRegistry &registry) override {
    // TODO:
  }

  LogicalResult onActivate() override {
    LLVM_DEBUG(llvm::dbgs() << "Coral plugin activated\n");
    if (failed(options.validate(context))) {
      return failure();
    }
    return success();
  }

  // Adds passes to the |buildPreprocessingPassPipeline| pipeline at the end.
  void extendPreprocessingPassPipeline(OpPassManager &passManager) override {
    passManager.addPass(
        createCoralNPUAffinityAnnotationPass({clAffinityIOMinThersholdBytes}));
  }
};

}  // namespace
}  // namespace mlir::coralnpu_compiler

IREE_DEFINE_COMPILER_OPTION_FLAGS(mlir::coralnpu_compiler::CoralNPUOptions);

extern "C" bool iree_register_compiler_plugin_coralnpu(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<mlir::coralnpu_compiler::CoralNPUSession>(
      "coralnpu");
  return true;
}
