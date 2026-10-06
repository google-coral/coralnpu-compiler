// Kernels with known binding sizes, to check where the runtime places the
// program and the dispatch data. The inputs come from the command line, so
// each binding is a buffer of its own.
module {
  func.func @add(%lhs: tensor<1000xi32>, %rhs: tensor<1000xi32>) -> tensor<1000xi32> {
    %0 = tensor.empty() : tensor<1000xi32>
    %1 = linalg.elementwise <add> ins(%lhs, %rhs : tensor<1000xi32>, tensor<1000xi32>) outs(%0 : tensor<1000xi32>) -> tensor<1000xi32>
    return %1 : tensor<1000xi32>
  }

  // The dynamic size is passed to the dispatch as push constants.
  func.func @add_dynamic(%lhs: tensor<?xi32>, %rhs: tensor<?xi32>) -> tensor<?xi32> {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %lhs, %c0 : tensor<?xi32>
    %0 = tensor.empty(%dim) : tensor<?xi32>
    %1 = linalg.elementwise <add> ins(%lhs, %rhs : tensor<?xi32>, tensor<?xi32>) outs(%0 : tensor<?xi32>) -> tensor<?xi32>
    return %1 : tensor<?xi32>
  }
}
