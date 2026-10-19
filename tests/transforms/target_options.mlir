// RUN: not %iree_compile --coralnpu-dtcm-size-kb=0 %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERR-ZERO
// RUN: not %iree_compile --coralnpu-stack-size-kb=1 %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERR-STACK
// RUN: not %iree_compile --coralnpu-dtcm-reserve-kb=-1 %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERR-RESERVE
// RUN: not %iree_compile --coralnpu-dtcm-buffers-per-operand=0 %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERR-BUFFERS
// RUN: not %iree_compile --coralnpu-dtcm-size-kb=11 %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERR-BUDGET
// RUN: not %iree_compile --coralnpu-roofline-speedup-threshold=-1 %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERR-THRESHOLD
// RUN: not %iree_compile --coralnpu-stack-size-kb=8 --coralnpu-roofline-speedup-threshold=0 --coralnpu-linker-script-path=/nonexistent/path.ld %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERR-LINK
// RUN: %iree_compile --coralnpu-dtcm-size-kb=1024 --coralnpu-roofline-speedup-threshold=0 --coralnpu-dump-affinity-profile-format=pretty --mlir-print-ir-after=iree-codegen-math-transform %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS
// RUN: %iree_compile --coralnpu-roofline-speedup-threshold=0 --coralnpu-fast-math=false --mlir-print-ir-after=iree-codegen-math-transform %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-NO-FAST-MATH
// RUN: %iree_compile --coralnpu-stack-size-kb=8 --coralnpu-roofline-speedup-threshold=0 --compile-to=executable-configurations %s | FileCheck %s --check-prefix=CHECK-STACK

// CHECK-ERR-ZERO: coralnpu-dtcm-size-kb must be positive, got 0

// CHECK-ERR-STACK: coralnpu-stack-size-kb must be at least 2, got 1

// CHECK-ERR-RESERVE: coralnpu-dtcm-reserve-kb must not be negative, got -1

// CHECK-ERR-BUFFERS: coralnpu-dtcm-buffers-per-operand must be at least 1, got 0

// CHECK-ERR-BUDGET: coralnpu-dtcm-size-kb (11) minus coralnpu-stack-size-kb (4) and coralnpu-dtcm-reserve-kb (4) must leave at least 4 KB for the DTCM tiles

// The stack allocations of a dispatch can use the stack size minus 1 KB.
// CHECK-STACK: #hal.executable.target<"coralnpu"{{.*}}max_stack_allocation_size = 7168 : i64

// CHECK-ERR-THRESHOLD: coralnpu-roofline-speedup-threshold must be a finite, non-negative value, got -1

// CHECK-ERR-LINK: --defsym=__stack_size__=8192
// CHECK-ERR-LINK: failed to serialize executables

// CHECK-SUCCESS: Execution Profile by Affinity:
// CHECK-SUCCESS:   Affinity: coralnpu
// CHECK-SUCCESS:     Dispatches: 1
// 214877.484 is the constant term of CoralNPUApproximateMath's tanh.
// CHECK-SUCCESS: 214877.484

// CHECK-NO-FAST-MATH: IR Dump After MathTransformPass
// CHECK-NO-FAST-MATH-NOT: 214877.484

module {
  func.func @main(%arg0: tensor<4x8xf32>, %arg1: tensor<8x4xf32>) -> tensor<4x4xf32> {
    %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>
    %1 = "stablehlo.tanh"(%0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }
}
