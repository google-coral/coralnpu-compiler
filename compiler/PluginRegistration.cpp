#include "Pipelines/Pipelines.h"

#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/PluginAPI/Client.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace mlir::coralnpu_compiler {
namespace {

struct CoralNPUOptions {
  bool flag = false;

  void bindOptions(OptionsBinder &binder) {
    static llvm::cl::OptionCategory category("CoralNPU Example Plugin");
    binder.opt<bool>("coralnpu-example-flag", flag,
                     llvm::cl::desc("Dummy CoralNPU flag"),
                     llvm::cl::cat(category));
  }
};

class CoralNPUTargetDevice final : public IREE::HAL::TargetDevice {
public:
  CoralNPUTargetDevice(const CoralNPUOptions & /*options*/) {}

  IREE::HAL::DeviceTargetAttr getDefaultDeviceTarget(
      MLIRContext *context,
      const IREE::HAL::TargetRegistry &targetRegistry) const override {
    Builder b(context);
    auto configAttr = b.getDictionaryAttr({});

    SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
    targetRegistry.getTargetBackend("llvm-cpu")
        ->getDefaultExecutableTargets(context, "coralnpu", configAttr,
                                      executableTargetAttrs);

    return IREE::HAL::DeviceTargetAttr::get(context,
                                            b.getStringAttr("coralnpu"),
                                            configAttr, executableTargetAttrs);
  }
};

class CoralNPUSession
    : public PluginSession<CoralNPUSession, CoralNPUOptions,
                           PluginActivationPolicy::DefaultActivated> {
  /***
   *** From PluginSession
   ***/
public:
  // static void globalInitialize() {}

  static void registerPasses() { registerTOSAConversionPasses(); }

  // static void registerGlobalDialects(DialectRegistry &registry) {}

  /***
   *** From AbstractPluginSession:
   ***/
public:
  // Populates new HAL target devices, if any, into the given list.
  // Targets will be merged into the plugin session-owned registry.
  void populateHALTargetDevices(IREE::HAL::TargetDeviceList &targets) override {
    targets.add("coralnpu", [=]() {
      return std::make_shared<CoralNPUTargetDevice>(options);
    });
  }

  // Populates new HAL target backends, if any, into the given list.
  // Targets will be merged into the plugin session-owned registry.
  // virtual void
  //     populateHALTargetBackends(IREE::HAL::TargetBackendList &targets) {}

protected:
  // Called from registerDialects() prior to initializing the context and
  // prior to onActivate().
  // virtual void onRegisterDialects(DialectRegistry &registry) {}

  // Called from the activate() method once pre-conditions are verified and the
  // context is set.
  LogicalResult onActivate() override {
    mlir::emitRemark(mlir::UnknownLoc::get(context))
        << "Coral plugin activated";
    // CoralNPUOptions is avilable as PluginSession::options
    return success();
  }

  /***
   *** From PipelineExtensions
   ***/
public:
  // Registers dialects used by the instance.
  // virtual void registerDialects(DialectRegistry &registry) {}

  // Adds passes to the input preprocessing pipeline, which allows to process
  // the raw input to IREE. This applies to builtin Type enum input pipelines.
  // virtual void extendInputConversionPreprocessingPassPipeline(
  //     OpPassManager &passManager, InputDialectOptions::Type inputType) {}

  // Adds input type mnemonics that this instance supports. At least one plugin
  // must advertise support for a custom input type in order for it to be
  // considered valid.
  void populateCustomInputConversionTypes(
      llvm::StringSet<> &typeMnemonics) override {
    typeMnemonics.insert("tosa-coralnpu");
    // TODO:
    // typeMnemonics.insert("linalg-coralnpu");
  }

  // Adds input type mnemonics that this instance supports, if those types are
  // detected in |module|.
  // Requires that |registerDialects| has been called first.
  // virtual void populateDetectedCustomInputConversionTypes(
  //     ModuleOp &module, llvm::StringSet<> &typeMnemonics) {}

  // Adds passes to the input preprocessing pipeline for the given
  // InputDialectOptions::Type::plugin type with the given mnemonic.
  // Returns true if extensions were made.
  bool extendCustomInputConversionPassPipeline(
      OpPassManager &passManager, std::string_view typeMnemonic) override {
    if (typeMnemonic == "tosa-coralnpu") {
      coralnpu_compiler::buildTOSAInputConversionPassPipeline(passManager);
      return true;
    }

    // TODO
    // if (typeMnemonic == "linalg-coralnpu") {
    //   return true;
    // }

    return false;
  }

  // Adds passes to the |buildPreprocessingPassPipeline| pipeline at the end.
  // virtual void extendPreprocessingPassPipeline(OpPassManager &passManager) {}
};

} // namespace
} // namespace mlir::coralnpu_compiler

IREE_DEFINE_COMPILER_OPTION_FLAGS(mlir::coralnpu_compiler::CoralNPUOptions);

extern "C" bool iree_register_compiler_plugin_coralnpu(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<mlir::coralnpu_compiler::CoralNPUSession>(
      "coralnpu");
  return true;
}
