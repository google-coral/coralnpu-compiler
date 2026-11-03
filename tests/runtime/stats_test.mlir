// Two independent functions: iree-check-module runs each on its own device, so
// the stats of both must end up in the same output.
module {
  func.func @add_i32() {
    %lhs = arith.constant dense<[1, 2, 3, 4, 5, 6, 7, 8]> : tensor<8xi32>
    %rhs = arith.constant dense<[10, 20, 30, 40, 50, 60, 70, 80]> : tensor<8xi32>
    %expected = arith.constant dense<[11, 22, 33, 44, 55, 66, 77, 88]> : tensor<8xi32>
    %0 = util.optimization_barrier %lhs : tensor<8xi32>
    %1 = util.optimization_barrier %rhs : tensor<8xi32>
    %2 = tensor.empty() : tensor<8xi32>
    %3 = linalg.elementwise <add> ins(%0, %1 : tensor<8xi32>, tensor<8xi32>) outs(%2 : tensor<8xi32>) -> tensor<8xi32>
    %4 = util.optimization_barrier %expected : tensor<8xi32>
    "check.expect_eq"(%3, %4) : (tensor<8xi32>, tensor<8xi32>) -> ()
    return
  }

  func.func @mul_i32() {
    %lhs = arith.constant dense<[1, 2, 3, 4, 5, 6, 7, 8]> : tensor<8xi32>
    %rhs = arith.constant dense<[10, 20, 30, 40, 50, 60, 70, 80]> : tensor<8xi32>
    %expected = arith.constant dense<[10, 40, 90, 160, 250, 360, 490, 640]> : tensor<8xi32>
    %0 = util.optimization_barrier %lhs : tensor<8xi32>
    %1 = util.optimization_barrier %rhs : tensor<8xi32>
    %2 = tensor.empty() : tensor<8xi32>
    %3 = linalg.elementwise <mul> ins(%0, %1 : tensor<8xi32>, tensor<8xi32>) outs(%2 : tensor<8xi32>) -> tensor<8xi32>
    %4 = util.optimization_barrier %expected : tensor<8xi32>
    "check.expect_eq"(%3, %4) : (tensor<8xi32>, tensor<8xi32>) -> ()
    return
  }
}
