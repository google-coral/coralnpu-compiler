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

// A two-layer f32 MLP: dense + bias + ReLU, then dense + logistic. The dense
// layers run on Zvt, and their elementwise epilogues on RVV.
func.func @main(%x: tensor<16x64xf32>, %w0: tensor<64x32xf32>,
                %b0: tensor<32xf32>, %w1: tensor<32x16xf32>)
    -> tensor<16x16xf32> {
  %0 = stablehlo.dot_general %x, %w0, contracting_dims = [1] x [0]
      : (tensor<16x64xf32>, tensor<64x32xf32>) -> tensor<16x32xf32>
  %1 = stablehlo.broadcast_in_dim %b0, dims = [1]
      : (tensor<32xf32>) -> tensor<16x32xf32>
  %2 = stablehlo.add %0, %1 : tensor<16x32xf32>
  %zero = stablehlo.constant dense<0.0> : tensor<16x32xf32>
  %3 = stablehlo.maximum %2, %zero : tensor<16x32xf32>
  %4 = stablehlo.dot_general %3, %w1, contracting_dims = [1] x [0]
      : (tensor<16x32xf32>, tensor<32x16xf32>) -> tensor<16x16xf32>
  %5 = stablehlo.logistic %4 : tensor<16x16xf32>
  return %5 : tensor<16x16xf32>
}
