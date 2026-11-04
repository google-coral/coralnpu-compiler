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

// Tile grid of the root op of each dispatch: its loop ranges, and its
// workgroup, DTCM and vector tile sizes.

// RUN: iree-compile --mlir-disable-threading --iree-hal-target-device=coralnpu --coralnpu-dump-metrics-format=pretty --coralnpu-dump-metrics-dir=- %s -o /dev/null | FileCheck %s
// RUN: iree-compile --mlir-disable-threading --iree-hal-target-device=coralnpu --coralnpu-dump-metrics-format=json --coralnpu-dump-metrics-dir=- %s -o /dev/null | FileCheck %s --check-prefix=CHECK-JSON

// The whole matmul fits DTCM, and its vector tile is the 32x32 Zvt tile, a
// 2x2 grid of 16x16 Zvt tiles.
// CHECK-LABEL: CoralNPU Metrics: matmul_dispatch_0
// CHECK:         tile_grid: {"dtcm":[64,64,128],"loops":[64,64,128],"op":"linalg.matmul","vector":[32,32,1],"workgroup":[64,64,128],"zvt_grid":[2,2]}

// The large matmul is split into 32x128 DTCM tiles, which are also the
// workgroup tiles; K is streamed one step at a time.
// CHECK-LABEL: CoralNPU Metrics: matmul_large_dispatch_0
// CHECK:         tile_grid: {"dtcm":[32,128,512],"loops":[256,256,512],"op":"linalg.matmul","vector":[32,32,1],"workgroup":[32,128,512],"zvt_grid":[2,2]}

// The elementwise add is collapsed to one loop.
// CHECK-LABEL: CoralNPU Metrics: add_dispatch_0
// CHECK:         tile_grid: {"dtcm":[1536],"loops":[6144],"op":"linalg.generic","vector":[32],"workgroup":[1536]}

// Dynamic loop ranges have no compile-time tile grid.
// CHECK-LABEL: CoralNPU Metrics: matmul_dynamic_dispatch_0
// CHECK-NOT:     tile_grid:

// CHECK-JSON:      "name": "matmul_dispatch_0_matmul_64x64x128_f32",
// CHECK-JSON:      "constant_count": 0,
// CHECK-JSON-NEXT: "tile_grid": {
// CHECK-JSON-NEXT:   "dtcm": [
// CHECK-JSON-NEXT:     64,
// CHECK-JSON-NEXT:     64,
// CHECK-JSON-NEXT:     128
// CHECK-JSON-NEXT:   ],
// CHECK-JSON-NEXT:   "loops": [

func.func @matmul(%lhs: tensor<64x128xf32>, %rhs: tensor<128x64xf32>) -> tensor<64x64xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<64x64xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
  %0 = linalg.matmul ins(%lhs, %rhs : tensor<64x128xf32>, tensor<128x64xf32>) outs(%fill : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

func.func @matmul_large(%lhs: tensor<256x512xf32>, %rhs: tensor<512x256xf32>) -> tensor<256x256xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<256x256xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<256x256xf32>) -> tensor<256x256xf32>
  %0 = linalg.matmul ins(%lhs, %rhs : tensor<256x512xf32>, tensor<512x256xf32>) outs(%fill : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %0 : tensor<256x256xf32>
}

func.func @add(%lhs: tensor<64x96xf32>, %rhs: tensor<64x96xf32>) -> tensor<64x96xf32> {
  %empty = tensor.empty() : tensor<64x96xf32>
  %0 = linalg.elementwise <add> ins(%lhs, %rhs : tensor<64x96xf32>, tensor<64x96xf32>) outs(%empty : tensor<64x96xf32>) -> tensor<64x96xf32>
  return %0 : tensor<64x96xf32>
}

func.func @matmul_dynamic(%lhs: tensor<?x128xf32>, %rhs: tensor<128x64xf32>) -> tensor<?x64xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %m = tensor.dim %lhs, %c0 : tensor<?x128xf32>
  %empty = tensor.empty(%m) : tensor<?x64xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<?x64xf32>) -> tensor<?x64xf32>
  %0 = linalg.matmul ins(%lhs, %rhs : tensor<?x128xf32>, tensor<128x64xf32>) outs(%fill : tensor<?x64xf32>) -> tensor<?x64xf32>
  return %0 : tensor<?x64xf32>
}
