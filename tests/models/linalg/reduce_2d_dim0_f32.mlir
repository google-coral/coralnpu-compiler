// RUN: %template_path
func.func @main(%arg0: tensor<?x?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0.0 : f32
  %c1_idx = arith.constant 1 : index
  %n = tensor.dim %arg0, %c1_idx : tensor<?x?xf32>
  %empty = tensor.empty(%n) : tensor<?xf32>
  %fill = linalg.fill ins(%c0 : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
  %0 = linalg.reduce ins(%arg0 : tensor<?x?xf32>) outs(%fill : tensor<?xf32>) dimensions = [0] (%in: f32, %out: f32) {
    %1 = arith.addf %in, %out : f32
    linalg.yield %1 : f32
  }
  return %0 : tensor<?xf32>
}
