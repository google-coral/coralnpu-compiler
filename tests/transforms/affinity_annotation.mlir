// RUN: %coralnpu_compile --compile-to=preprocessing --coralnpu-affinity-io-min-threshold-bytes=1000 %s | FileCheck %s

// CHECK-LABEL: @big_matmul
func.func @big_matmul(
    %arg0: tensor<32x32xf32>,
    %arg1: tensor<32x32xf32>,
    %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // 32x32xf32 = 4096 bytes per tensor, total > 1000 bytes.
  // CHECK: linalg.matmul
  // CHECK-SAME: stream.affinity = #hal.device.affinity<@{{.*}}>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<32x32xf32>, tensor<32x32xf32>)
                     outs(%arg2 : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// CHECK-LABEL: @small_matmul
func.func @small_matmul(
    %arg0: tensor<2x2xf32>,
    %arg1: tensor<2x2xf32>,
    %arg2: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // 2x2xf32 = 16 bytes per tensor, total 48 bytes < 1000 bytes.
  // CHECK: linalg.matmul
  // CHECK-NOT: stream.affinity
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<2x2xf32>, tensor<2x2xf32>)
                     outs(%arg2 : tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: @unsupported_i64
func.func @unsupported_i64(
    %arg0: tensor<32x32xi64>,
    %arg1: tensor<32x32xi64>,
    %arg2: tensor<32x32xi64>) -> tensor<32x32xi64> {
  // i64 is not supported on CoralNPU
  // CHECK: linalg.matmul
  // CHECK-NOT: stream.affinity
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<32x32xi64>, tensor<32x32xi64>)
                     outs(%arg2 : tensor<32x32xi64>) -> tensor<32x32xi64>
  return %0 : tensor<32x32xi64>
}

// CHECK-LABEL: @dynamic_shape
func.func @dynamic_shape(
    %arg0: tensor<?x?xf32>,
    %arg1: tensor<?x?xf32>,
    %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // Dynamic shapes are not supported on CoralNPU
  // CHECK: linalg.matmul
  // CHECK-NOT: stream.affinity
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @constants_and_metadata
func.func @constants_and_metadata() -> tensor<32x32xf32> {
  // CHECK: arith.constant
  // CHECK-NOT: stream.affinity
  %cst = arith.constant dense<1.000000e+00> : tensor<32x32xf32>
  // CHECK: tensor.empty
  // CHECK-NOT: stream.affinity
  %empty = tensor.empty() : tensor<32x32xf32>
  // CHECK: linalg.matmul
  // CHECK-SAME: stream.affinity = #hal.device.affinity<@{{.*}}>
  %0 = linalg.matmul ins(%cst, %cst : tensor<32x32xf32>, tensor<32x32xf32>)
                     outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

