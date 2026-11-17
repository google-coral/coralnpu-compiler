// RUN: iree-compile --mlir-disable-threading --iree-hal-target-device=coralnpu \
// RUN:   --compile-to=executable-configurations %s | FileCheck %s --check-prefix=DEFAULT
// RUN: rm -rf %t.dir
// RUN: iree-compile --mlir-disable-threading --iree-hal-target-device=coralnpu \
// RUN:   --coralnpu-target-cpu-features=+m,+f,+zvl128b,+zve32f,+zvtbase,+zvt16e,+zvti16i32mm \
// RUN:   --iree-hal-dump-executable-files-to=%t.dir %s -o /dev/null
// RUN: cat %t.dir/*.s | FileCheck %s
// RUN: not iree-compile --mlir-disable-threading --iree-hal-target-device=coralnpu \
// RUN:   --coralnpu-target-cpu-features=+m,+f,+zvl128b,+zve32f,+zvtbase,+zvt16e \
// RUN:   %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

// Default CoralNPU CPU features include +zvti16i32mm, and the assembler accepts
// vtmms.tvv with +zvti16i32mm alone while rejecting +zvtbase,+zvt16e without an
// integer multiply sub-extension.
// DEFAULT: #hal.executable.target<"coralnpu"{{.*}}cpu_features = "{{.*}}+zvti16i32mm
// CHECK: vtmms.tvv
// ERR: instruction requires the following: either 'Zvti8i32mm' or 'Zvti16i32mm'

func.func @matmul_i16_i32(%a: tensor<16x8xi16>, %b: tensor<8x16xi16>) -> tensor<16x16xi32> {
  %c0 = arith.constant 0 : i32
  %empty = tensor.empty() : tensor<16x16xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%empty : tensor<16x16xi32>) -> tensor<16x16xi32>
  %mm = linalg.matmul ins(%a, %b : tensor<16x8xi16>, tensor<8x16xi16>)
                      outs(%fill : tensor<16x16xi32>) -> tensor<16x16xi32>
  return %mm : tensor<16x16xi32>
}
