func.func @main(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %min = arith.constant 0.0 : f64
  %max = arith.constant 1.0 : f64
  %seed = arith.constant 12345 : i32
  %barrier_seed = util.optimization_barrier %seed : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %empty = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %0 = linalg.fill_rng_2d ins(%min, %max, %barrier_seed : f64, f64, i32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
