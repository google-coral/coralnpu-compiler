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

// Static Zvt lowering: whether the root op of each dispatch is a Zvt candidate
// and whether the matrix codegen lowered it to Zvt, with the missed remarks
// that explain why not. The RUN lines use
// --coralnpu-roofline-speedup-threshold=0 to place the small dispatches on
// CoralNPU; by default, the roofline model keeps them on the host.

// RUN: rm -rf %t && mkdir -p %t
// RUN: %iree_compile --coralnpu-roofline-speedup-threshold=0 --coralnpu-dump-metrics-format=pretty --coralnpu-dump-metrics-dir=- --remarks-filter=coralnpu-zvt --remarks-output-file=%t/remarks.yaml %s -o /dev/null | FileCheck %s
// RUN: FileCheck %s --check-prefix=CHECK-REMARKS --input-file=%t/remarks.yaml
// RUN: %iree_compile --coralnpu-roofline-speedup-threshold=0 --coralnpu-dump-metrics-format=pretty --coralnpu-dump-metrics-dir=- --coralnpu-dtcm-size-kb=12 --coralnpu-dtcm-buffers-per-operand=25 --remarks-filter=coralnpu-zvt --remarks-output-file=%t/small_dtcm.yaml %s -o /dev/null | FileCheck %s --check-prefix=CHECK-SMALL
// RUN: FileCheck %s --check-prefix=CHECK-SMALL-REMARKS --input-file=%t/small_dtcm.yaml

// The 32x32 vector tile is a 2x2 grid of 16x16 Zvt tiles.
// CHECK-LABEL: CoralNPU Metrics: matmul_f32_dispatch_0
// CHECK:         tile_grid: {{.*}}"vector":[32,32,1],{{.*}}"zvt_grid":[2,2]}
// CHECK-NEXT:    zvt: {"candidate":true,"lowered":true}

// With K = 1, the result of the contraction also feeds the store of the tile.
// CHECK-LABEL: CoralNPU Metrics: matmul_k1_dispatch_0
// CHECK:         tile_grid: {"dtcm":[16,16,1],"loops":[16,16,1],"op":"linalg.matmul","vector":[16,16,1],"workgroup":[16,16,1]}
// CHECK-NEXT:    zvt: {"candidate":true,"lowered":false}

// i32 x i32 -> i32 is not a Zvt type.
// CHECK-LABEL: CoralNPU Metrics: matmul_i32_dispatch_0
// CHECK:         tile_grid: {{.*}}"vector":[2,16,1],"workgroup":[16,16,16]}
// CHECK-NEXT:    zvt: {"candidate":false,"lowered":false}

// A fused non-widening input producer prevents the Zvt unit from loading the
// inputs directly.
// CHECK-LABEL: CoralNPU Metrics: matmul_fused_input_dispatch_0
// CHECK:         tile_grid: {"dtcm":[16,16,16],"loops":[16,16,16],"op":"linalg.matmul","vector":[4,16,1],"workgroup":[16,16,16]}
// CHECK-NEXT:    zvt: {"candidate":true,"lowered":false}

// CHECK-REMARKS:      --- !Missed
// CHECK-REMARKS-NEXT: Pass: 'coralnpu-zvt:matrix-codegen'
// CHECK-REMARKS-NEXT: Name: ZvtLowering
// CHECK-REMARKS:      Function: matmul_k1_dispatch_0_matmul_16x16x1_f32
// CHECK-REMARKS:      Reason: the contraction result has several uses
// CHECK-REMARKS:      --- !Missed
// CHECK-REMARKS-NEXT: Pass: 'coralnpu-zvt:dtcm-tiling'
// CHECK-REMARKS-NEXT: Name: ZvtLowering
// CHECK-REMARKS:      Function: matmul_fused_input_dispatch_0_matmul_16x16x16_f32
// CHECK-REMARKS:      Reason: the Zvt unit cannot load the fused inputs, so the vector tile keeps the RVV sizes
// CHECK-REMARKS:      --- !Missed
// CHECK-REMARKS-NEXT: Pass: 'coralnpu-zvt:matrix-codegen'
// CHECK-REMARKS-NEXT: Name: ZvtLowering
// CHECK-REMARKS:      Function: matmul_fused_input_dispatch_0_matmul_16x16x16_f32
// CHECK-REMARKS:      Reason: partial or converted tiles need a fused K loop
// CHECK-REMARKS-NOT:  --- !Missed

// In a 4 KB DTCM budget with 25 buffers per operand, no Zvt tile fits, and the
// 4x16x1 RVV tile has no inner K loop for a partial Zvt tile.
// CHECK-SMALL-LABEL: CoralNPU Metrics: matmul_f32_dispatch_0
// CHECK-SMALL:         tile_grid: {"dtcm":[4,16,1],"loops":[64,96,128],"op":"linalg.matmul","vector":[4,16,1],"workgroup":[4,16,128]}
// CHECK-SMALL-NEXT:    zvt: {"candidate":true,"lowered":false}

// CHECK-SMALL-REMARKS:      Pass: 'coralnpu-zvt:dtcm-tiling'
// CHECK-SMALL-REMARKS:      Function: matmul_f32_dispatch_0_matmul_64x96x128_f32
// CHECK-SMALL-REMARKS:      Reason: no Zvt tile fits the DTCM, so the vector tile keeps the RVV sizes

func.func @matmul_f32(%lhs: tensor<64x128xf32>, %rhs: tensor<128x96xf32>) -> tensor<64x96xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<64x96xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<64x96xf32>) -> tensor<64x96xf32>
  %0 = linalg.matmul ins(%lhs, %rhs : tensor<64x128xf32>, tensor<128x96xf32>) outs(%fill : tensor<64x96xf32>) -> tensor<64x96xf32>
  return %0 : tensor<64x96xf32>
}

func.func @matmul_k1(%lhs: tensor<16x1xf32>, %rhs: tensor<1x16xf32>) -> tensor<16x16xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<16x16xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
  %0 = linalg.matmul ins(%lhs, %rhs : tensor<16x1xf32>, tensor<1x16xf32>) outs(%fill : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}

func.func @matmul_i32(%lhs: tensor<16x16xi32>, %rhs: tensor<16x16xi32>) -> tensor<16x16xi32> {
  %cst = arith.constant 0 : i32
  %empty = tensor.empty() : tensor<16x16xi32>
  %fill = linalg.fill ins(%cst : i32) outs(%empty : tensor<16x16xi32>) -> tensor<16x16xi32>
  %0 = linalg.matmul ins(%lhs, %rhs : tensor<16x16xi32>, tensor<16x16xi32>) outs(%fill : tensor<16x16xi32>) -> tensor<16x16xi32>
  return %0 : tensor<16x16xi32>
}

func.func @matmul_fused_input(%lhs: tensor<16x16xi8>, %rhs: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %cst = arith.constant 0.0 : f32
  %lhs_empty = tensor.empty() : tensor<16x16xf32>
  %lhs_f32 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%lhs : tensor<16x16xi8>) outs(%lhs_empty : tensor<16x16xf32>) {
  ^bb0(%in: i8, %out: f32):
    %cast = arith.sitofp %in : i8 to f32
    linalg.yield %cast : f32
  } -> tensor<16x16xf32>
  %empty = tensor.empty() : tensor<16x16xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
  %0 = linalg.matmul ins(%lhs_f32, %rhs : tensor<16x16xf32>, tensor<16x16xf32>) outs(%fill : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}
