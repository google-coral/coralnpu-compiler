// RUN: (%coralnpu_compile --coralnpu-dtcm-size-kb=0 %s || true) 2>&1 | FileCheck %s --check-prefix=CHECK-ERR-ZERO
// RUN: (%coralnpu_compile --coralnpu-linker-script-path=/nonexistent/path.ld %s || true) 2>&1 | FileCheck %s --check-prefix=CHECK-ERR-LINK
// RUN: %coralnpu_compile --coralnpu-dtcm-size-kb=1024 --coralnpu-dump-affinity-profile-format=pretty %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-SUCCESS

// CHECK-ERR-ZERO: coralnpu-dtcm-size-kb must be positive, got 0

// CHECK-ERR-LINK: failed to serialize executables

// CHECK-SUCCESS: Execution Profile by Affinity:
// CHECK-SUCCESS:   Affinity: coralnpu
// CHECK-SUCCESS:     Dispatches: 1

module {
  func.func @main(%arg0: tensor<4x8xi32>, %arg1: tensor<8x4xi32>) -> tensor<4x4xi32> {
    %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<4x8xi32>, tensor<8x4xi32>) -> tensor<4x4xi32>
    return %0 : tensor<4x4xi32>
  }
}
