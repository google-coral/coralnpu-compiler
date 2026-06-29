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

#include "CheckTestGenerator.h"

// CoralNPU headers
#include "tools/check_gen/CompileUtils.h"
#include "tools/check_gen/ParseUtils.h"
#include "tools/check_gen/PrecompiledBinary.h"

// IREE headers
#include "iree/compiler/ConstEval/Runtime.h"
#include "iree/compiler/embedding_api.h"

// MLIR headers
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"

// StableHLO headers
#include "stablehlo/transforms/Passes.h"

// LLVM headers
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

// Standard C/C++ headers
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace mlir::check_gen {

namespace {

TypedAttr getArgAttr(OpBuilder &builder, int64_t val, Type expectedType) {
  if (isa<IndexType>(expectedType)) {
    return builder.getIndexAttr(val);
  }
  if (auto intType = dyn_cast<IntegerType>(expectedType)) {
    if (intType.getWidth() == 32) {
      return builder.getI32IntegerAttr(static_cast<int32_t>(val));
    }
    if (intType.getWidth() == 64) {
      return builder.getI64IntegerAttr(val);
    }
    if (intType.getWidth() == 8) {
      return builder.getI8IntegerAttr(static_cast<int8_t>(val));
    }
    if (intType.getWidth() == 16) {
      return builder.getI16IntegerAttr(static_cast<int16_t>(val));
    }
    if (intType.getWidth() == 1) {
      return builder.getBoolAttr(val != 0);
    }
  }
  if (isa<FloatType>(expectedType)) {
    return builder.getF32FloatAttr(static_cast<float>(val));
  }
  return builder.getIndexAttr(val);
}

TypedAttr getVmfbArgAttr(OpBuilder &builder, int64_t val, char cc) {
  switch (cc) {
    case 'i':
      return builder.getI32IntegerAttr(static_cast<int32_t>(val));
    case 'I':
      return builder.getI64IntegerAttr(val);
    default:
      return builder.getIndexAttr(val);
  }
}

}  // namespace

CheckTestGenerator::CheckTestGenerator(
    MLIRContext *context, iree_compiler_session_t *session,
    std::vector<std::vector<std::vector<int64_t>>> instances,
    std::vector<std::string> inputFiles, llvm::StringRef outputDir,
    llvm::StringRef defaultGenPath)
    : context(context),
      session(session),
      instances(std::move(instances)),
      inputFiles(std::move(inputFiles)),
      outputDir(outputDir.str()),
      defaultGenPath(defaultGenPath.str()) {}

std::shared_ptr<PrecompiledBinary> CheckTestGenerator::getOrLoadBinary(
    llvm::StringRef path, Location loc) {
  if (auto it = binaryCache.find(path); it != binaryCache.end()) {
    return it->second;
  }

  llvm::StringRef ext = llvm::sys::path::extension(path);
  std::shared_ptr<PrecompiledBinary> binary;

  if (ext == ".vmfb") {
    std::ifstream file(path.str(), std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
      llvm::errs() << "Failed to open VMFB generator: " << path << "\n";
      return nullptr;
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
      llvm::errs() << "Failed to read VMFB generator: " << path << "\n";
      return nullptr;
    }

    binary = std::make_shared<PrecompiledBinary>();
    if (failed(binary->init(loc, buffer.data(), buffer.size()))) {
      llvm::errs() << "Failed to initialize VMFB generator binary: " << path
                   << "\n";
      return nullptr;
    }
  } else if (ext == ".mlir") {
    std::string errorMessage;
    auto input = openInputFile(path, &errorMessage);
    if (!input) {
      llvm::errs() << errorMessage << "\n";
      return nullptr;
    }

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());

    OwningOpRef<ModuleOp> moduleOp =
        parseSourceFile<ModuleOp>(sourceMgr, context);
    if (!moduleOp) {
      llvm::errs() << "Failed to parse MLIR generator: " << path << "\n";
      return nullptr;
    }

    iree_compiler_invocation_t *inv = ireeCompilerInvocationCreate(session);
    void *binaryData = nullptr;
    uint64_t binarySize = 0;
    iree_compiler_output_t *output =
        compileModule(inv, *moduleOp, binaryData, binarySize);
    if (!output) {
      llvm::errs() << "Failed to compile MLIR generator: " << path << "\n";
      ireeCompilerInvocationDestroy(inv);
      return nullptr;
    }

    binary = std::make_shared<PrecompiledBinary>();
    if (failed(binary->init(loc, binaryData, binarySize))) {
      llvm::errs() << "Failed to initialize compiled generator binary: " << path
                   << "\n";
      ireeCompilerOutputDestroy(output);
      ireeCompilerInvocationDestroy(inv);
      return nullptr;
    }
    ireeCompilerOutputDestroy(output);
    ireeCompilerInvocationDestroy(inv);
  } else {
    llvm::errs() << "Unsupported generator file extension: " << ext << " for "
                 << path << "\n";
    return nullptr;
  }

  binaryCache[path] = binary;
  return binary;
}

bool CheckTestGenerator::loadDefaultGenerator(Type expectedType,
                                              GeneratorInfo &gen) {
  if (defaultGenPath.empty()) {
    std::string typeStr;
    llvm::raw_string_ostream os(typeStr);
    expectedType.print(os);
    llvm::errs()
        << "Error: No default generators file specified, but needed for type "
        << typeStr << "\n";
    return false;
  }

  return loadGenerator(defaultGenPath, expectedType, gen);
}

bool CheckTestGenerator::run() {
  if (!parseAndMerge()) return false;

  for (size_t instIdx = 0; instIdx < instances.size(); ++instIdx) {
    if (!processInstance(instIdx)) {
      return false;
    }
  }

  return true;
}

bool CheckTestGenerator::parseAndMerge() {
  mergedModuleOp = ModuleOp::create(Builder(context).getUnknownLoc());
  SymbolTable symbolTable(*mergedModuleOp);

  testFunc = parseAndMergeFunc(inputFiles[0], symbolTable);
  if (!testFunc) return false;

  if (inputFiles.size() - 1 > testFunc.getNumArguments()) {
    llvm::errs() << "Error: Too many generators provided\n";
    return false;
  }

  for (size_t argIdx = 0; argIdx < testFunc.getNumArguments(); ++argIdx) {
    Type argType = testFunc.getArgument(argIdx).getType();

    GeneratorInfo gen;
    if (argIdx + 1 >= inputFiles.size() ||
        inputFiles[argIdx + 1] == "default") {
      if (!loadDefaultGenerator(argType, gen)) {
        return false;
      }
    } else {
      if (!loadGenerator(inputFiles[argIdx + 1], argType, gen)) {
        return false;
      }
    }

    generators.push_back(std::move(gen));
  }

  if (!validateInputs()) return false;

  return true;
}

func::FuncOp CheckTestGenerator::parseAndMergeFunc(StringRef filename,
                                                   SymbolTable &symbolTable) {
  std::string errorMessage;
  auto input = openInputFile(filename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());

  OwningOpRef<ModuleOp> moduleRef =
      parseSourceFile<ModuleOp>(sourceMgr, context);
  if (!moduleRef) {
    llvm::errs() << "Failed to parse MLIR file: " << filename << "\n";
    return nullptr;
  }

  auto functions = moduleRef->getOps<func::FuncOp>();
  if (functions.empty()) {
    llvm::errs() << "Error: File " << filename
                 << " does not contain any func.func op\n";
    return nullptr;
  }

  if (llvm::range_size(functions) > 1) {
    llvm::errs() << "Error: File " << filename
                 << " contains multiple func.func ops\n";
    return nullptr;
  }

  auto originalFunc = *functions.begin();
  func::FuncOp cloned = originalFunc.clone();
  symbolTable.insert(cloned);

  return cloned;
}

bool CheckTestGenerator::validateInputs() {
  for (const auto &[instIdx, instance] : llvm::enumerate(instances)) {
    if (instance.size() != generators.size()) {
      llvm::errs() << "Error: Number of argument groups in instance " << instIdx
                   << " (" << instance.size()
                   << ") does not match number of generators ("
                   << generators.size() << ")\n";
      return false;
    }
    for (const auto &[g, gen] : llvm::enumerate(generators)) {
      if (gen.numArguments != instance[g].size()) {
        llvm::errs() << "Error: Generator " << gen.filename << " expects "
                     << gen.numArguments << " arguments, but instance "
                     << instIdx << " group " << g << " has "
                     << instance[g].size() << " values\n";
        return false;
      }
    }
  }

  return true;
}

bool CheckTestGenerator::processInstance(size_t instIdx) {
  const auto &instance = instances[instIdx];

  auto inputAttrs = evaluateGenerators(instance);
  if (inputAttrs.empty()) {
    return false;
  }

  auto refinedTestModuleOp = refineShapes(inputAttrs, instIdx);
  if (!refinedTestModuleOp) {
    return false;
  }

  auto refinedTestFunc =
      dyn_cast_or_null<func::FuncOp>(refinedTestModuleOp->lookupSymbol("main"));
  if (!refinedTestFunc) {
    llvm::errs() << "Failed to find refined test function\n";
    return false;
  }

  OwningOpRef<ModuleOp> checkModuleOp = refinedTestModuleOp->clone();
  auto checkTestFunc =
      dyn_cast_or_null<func::FuncOp>(checkModuleOp->lookupSymbol("main"));

  auto outputAttrs = evaluateRefinedTest(*refinedTestModuleOp, checkTestFunc,
                                         inputAttrs, instIdx);
  if (outputAttrs.empty()) {
    return false;
  }

  if (!generateCheckTest(checkTestFunc, inputAttrs, outputAttrs, instance)) {
    llvm::errs() << "Failed to generate check test for instance " << instIdx
                 << "\n";
    return false;
  }

  return true;
}

std::vector<TypedAttr> CheckTestGenerator::evaluateGenerators(
    const std::vector<std::vector<int64_t>> &instance) {
  std::vector<TypedAttr> inputAttrs;
  OpBuilder builder(context);
  size_t argIdx = 0;
  for (const auto &[gen, genInstArgs] : llvm::zip_equal(generators, instance)) {
    std::vector<TypedAttr> genArgsAttrs;
    for (size_t i = 0; i < genInstArgs.size(); ++i) {
      int64_t val = genInstArgs[i];
      char cc = 'I';  // Default fallback
      if (gen.callingConvention.size() > 1 + i) {
        char c = gen.callingConvention[1 + i];
        if (c != '_' && c != '\0') {
          cc = c;
        }
      }
      genArgsAttrs.push_back(getVmfbArgAttr(builder, val, cc));
    }

    std::vector<Type> expectedTypes = {testFunc.getArgument(argIdx).getType()};

    auto genOutputs =
        evaluateFunction(*gen.binary, testFunc.getLoc(), gen.funcName,
                         expectedTypes, genArgsAttrs);
    if (genOutputs.empty()) {
      return {};
    }
    llvm::append_range(inputAttrs, genOutputs);
    argIdx++;
  }
  return inputAttrs;
}

OwningOpRef<ModuleOp> CheckTestGenerator::refineShapes(
    const std::vector<TypedAttr> &inputAttrs, size_t instIdx) {
  OwningOpRef<ModuleOp> refinedTestModuleOp =
      ModuleOp::create(Builder(context).getUnknownLoc());
  SymbolTable refinedSymbolTable(*refinedTestModuleOp);
  refinedSymbolTable.insert(testFunc.clone());

  std::vector<Type> refinedTypes;
  for (auto attr : inputAttrs) {
    refinedTypes.push_back(attr.getType());
  }
  PassManager pmRefine(context);
  // TODO(sflur): support other dialects
  mlir::stablehlo::createStablehloRemoveDynamismPipeline(pmRefine,
                                                         refinedTypes);
  if (failed(pmRefine.run(*refinedTestModuleOp))) {
    llvm::errs() << "Failed to run shape refinement pipeline for instance "
                 << instIdx << "\n";
    return nullptr;
  }
  return refinedTestModuleOp;
}

std::vector<TypedAttr> CheckTestGenerator::evaluateRefinedTest(
    ModuleOp refinedTestModuleOp, func::FuncOp checkTestFunc,
    const std::vector<TypedAttr> &inputAttrs, size_t instIdx) {
  iree_compiler_invocation_t *testInv = ireeCompilerInvocationCreate(session);
  void *testBinaryData = nullptr;
  uint64_t testBinarySize = 0;
  iree_compiler_output_t *testOutput = compileModule(
      testInv, refinedTestModuleOp, testBinaryData, testBinarySize);
  if (!testOutput) {
    llvm::errs() << "Failed to compile refined test module for instance "
                 << instIdx << "\n";
    ireeCompilerInvocationDestroy(testInv);
    return {};
  }

  PrecompiledBinary testBinary;
  if (failed(testBinary.init(checkTestFunc.getLoc(), testBinaryData,
                             testBinarySize))) {
    llvm::errs() << "Failed to initialize test binary for instance " << instIdx
                 << "\n";
    ireeCompilerOutputDestroy(testOutput);
    ireeCompilerInvocationDestroy(testInv);
    return {};
  }

  auto outputAttrs =
      evaluateFunction(testBinary, checkTestFunc.getLoc(), "main",
                       checkTestFunc.getResultTypes(), inputAttrs);

  ireeCompilerOutputDestroy(testOutput);
  ireeCompilerInvocationDestroy(testInv);

  return outputAttrs;
}

bool CheckTestGenerator::generateCheckTest(
    func::FuncOp refinedTestFunc, const std::vector<TypedAttr> &inputAttrs,
    const std::vector<TypedAttr> &outputAttrs,
    const std::vector<std::vector<int64_t>> &instance) {
  OwningOpRef<ModuleOp> outModule =
      ModuleOp::create(Builder(context).getUnknownLoc());
  OpBuilder outBuilder(outModule->getBodyRegion());

  auto clonedTestFunc = refinedTestFunc.clone();
  outBuilder.insert(clonedTestFunc);

  auto refinedArgTypes = clonedTestFunc.getArgumentTypes();
  auto refinedResultTypes = clonedTestFunc.getResultTypes();
  std::string testFuncName = clonedTestFunc.getName().str();
  Location loc = clonedTestFunc.getLoc();

  std::string instSuffix = "";
  for (size_t i = 0; i < instance.size(); ++i) {
    if (i > 0) {
      instSuffix += "-";
    } else {
      instSuffix += "_";
    }
    for (size_t j = 0; j < instance[i].size(); ++j) {
      if (j > 0) {
        instSuffix += "_";
      }
      instSuffix += std::to_string(instance[i][j]);
    }
  }
  std::string testFileBaseName = llvm::sys::path::stem(inputFiles[0]).str();
  std::string checkFuncName = (testFileBaseName + instSuffix);

  auto checkFuncType = outBuilder.getFunctionType({}, {});
  auto checkFunc =
      func::FuncOp::create(outBuilder, loc, checkFuncName, checkFuncType);
  checkFunc.addEntryBlock();

  auto *entryBlock = &checkFunc.front();
  OpBuilder funcBuilder(entryBlock, entryBlock->begin());

  std::vector<Value> callArgs;
  if (failed(addConstantInputs(funcBuilder, loc, inputAttrs, refinedArgTypes,
                               callArgs))) {
    return false;
  }

  auto callOp = func::CallOp::create(
      funcBuilder, loc, FlatSymbolRefAttr::get(context, testFuncName),
      refinedResultTypes, callArgs);

  if (failed(addAssertions(funcBuilder, loc, callOp, outputAttrs))) {
    return false;
  }

  func::ReturnOp::create(funcBuilder, loc);

  if (failed(inlineAndCleanup(*outModule, testFuncName))) {
    return false;
  }

  std::string outPath = outputDir + "/" + checkFuncName + "_check.mlir";
  if (failed(writeModuleToFile(*outModule, outPath))) {
    return false;
  }

  return true;
}

LogicalResult CheckTestGenerator::addConstantInputs(
    OpBuilder &funcBuilder, Location loc,
    const std::vector<TypedAttr> &inputAttrs, ArrayRef<Type> refinedArgTypes,
    std::vector<Value> &callArgs) {
  for (size_t i = 0; i < inputAttrs.size(); ++i) {
    auto attr = inputAttrs[i];

    OperationState cstState(loc, "util.unfoldable_constant");
    cstState.addAttribute("value", attr);
    cstState.addTypes(attr.getType());
    auto cstOp = funcBuilder.create(cstState);
    Value cstVal = cstOp->getResult(0);

    if (cstVal.getType() != refinedArgTypes[i]) {
      llvm::errs() << "Error: Refined argument type mismatch\n";
      return failure();
    }
    callArgs.push_back(cstVal);
  }
  return success();
}

LogicalResult CheckTestGenerator::addAssertions(
    OpBuilder &funcBuilder, Location loc, func::CallOp callOp,
    const std::vector<TypedAttr> &outputAttrs) {
  for (size_t i = 0; i < outputAttrs.size(); ++i) {
    auto expectedAttr = outputAttrs[i];
    auto resVal = callOp.getResult(i);

    if (resVal.getType() != expectedAttr.getType()) {
      llvm::errs() << "Error: Refined result type mismatch\n";
      return failure();
    }

    OperationState expectedCstState(loc, "util.unfoldable_constant");
    expectedCstState.addAttribute("value", expectedAttr);
    expectedCstState.addTypes(expectedAttr.getType());
    auto expectedCst = funcBuilder.create(expectedCstState);

    bool isFloat = false;
    if (auto shapedType = dyn_cast<ShapedType>(resVal.getType())) {
      isFloat = isa<FloatType>(shapedType.getElementType());
    } else {
      isFloat = isa<FloatType>(resVal.getType());
    }
    std::string checkOp =
        isFloat ? "check.expect_almost_eq" : "check.expect_eq";
    OperationState checkState(loc, checkOp);
    checkState.addOperands({resVal, expectedCst->getResult(0)});
    if (isFloat) {
      // Add relative tolerance to support FMA vs non-FMA comparison for large values.
      checkState.addAttribute("rtol", funcBuilder.getF32FloatAttr(1e-6f));
    }
    funcBuilder.create(checkState);
  }
  return success();
}

LogicalResult CheckTestGenerator::inlineAndCleanup(
    ModuleOp outModule, llvm::StringRef testFuncName) {
  PassManager pmInline(context);
  pmInline.addPass(createInlinerPass(llvm::StringMap<OpPassManager>{},
                                     [](OpPassManager &) {}));
  if (failed(pmInline.run(outModule))) {
    llvm::errs() << "Failed to run inliner pass\n";
    return failure();
  }

  if (auto op = outModule.lookupSymbol(testFuncName)) {
    op->erase();
  }
  return success();
}

static bool areTypesCompatible(Type genType, Type expType) {
  if (genType == expType) return true;

  auto genRTT = dyn_cast<RankedTensorType>(genType);
  auto expRTT = dyn_cast<RankedTensorType>(expType);
  if (!genRTT || !expRTT) return false;

  if (genRTT.getElementType() != expRTT.getElementType()) return false;
  if (genRTT.getEncoding() != expRTT.getEncoding()) return false;
  if (genRTT.getRank() != expRTT.getRank()) return false;

  auto genShape = genRTT.getShape();
  auto expShape = expRTT.getShape();
  for (auto [genDim, expDim] : llvm::zip_equal(genShape, expShape)) {
    if (genDim == expDim) continue;
    if (genDim == ShapedType::kDynamic) continue;
    return false;
  }

  return true;
}

bool CheckTestGenerator::loadGenerator(llvm::StringRef path, Type expectedType,
                                       GeneratorInfo &gen) {
  gen.filename = path.str();
  auto binary = getOrLoadBinary(path, testFunc.getLoc());
  if (!binary) return false;

  iree_vm_module_t *module = binary->getModule();
  iree_vm_module_signature_t sig = iree_vm_module_signature(module);

  for (iree_host_size_t i = 0; i < sig.export_function_count; ++i) {
    iree_vm_function_t func;
    iree_status_t status = iree_vm_module_lookup_function_by_ordinal(
        module, IREE_VM_FUNCTION_LINKAGE_EXPORT, i, &func);
    if (!iree_status_is_ok(status)) continue;

    iree_string_view_t name = iree_vm_function_name(&func);
    std::string nameStr(name.data, name.size);

    if (nameStr.rfind("__", 0) == 0) {
      continue;
    }

    iree_string_view_t declView = iree_vm_function_lookup_attr_by_name(
        &func, iree_make_cstring_view("iree.abi.declaration"));
    if (declView.size == 0) {
      continue;
    }
    llvm::StringRef declStr(declView.data, declView.size);
    Type resType = parseTypeFromDecl(declStr, context);
    if (!resType) {
      llvm::errs() << "Warning: Failed to parse declaration: " << declStr
                   << "\n";
      continue;
    }

    if (areTypesCompatible(resType, expectedType)) {
      gen.funcName = nameStr;
      iree_vm_function_signature_t funcSig = iree_vm_function_signature(&func);
      gen.callingConvention = std::string(funcSig.calling_convention.data,
                                          funcSig.calling_convention.size);
      iree_host_size_t argCount = 0;
      iree_host_size_t resultCount = 0;
      status = iree_vm_function_call_count_arguments_and_results(
          &funcSig, &argCount, &resultCount);
      if (!iree_status_is_ok(status)) {
        llvm::errs() << "Failed to count arguments and results for generator: "
                     << gen.filename << "\n";
        return false;
      }
      if (resultCount != 1) {
        llvm::errs() << "Error: Generator " << gen.filename
                     << " must have exactly 1 result, but has " << resultCount
                     << "\n";
        return false;
      }
      gen.numArguments = argCount;
      gen.binary = binary;
      return true;
    }
  }
  llvm::errs()
      << "Failed to find function in generator matching expected type in: "
      << path << "\n";
  return false;
}

}  // namespace mlir::check_gen
