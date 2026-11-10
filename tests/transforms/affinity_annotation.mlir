// RUN: %iree_compile --compile-to=dispatch-creation %s | FileCheck %s
// RUN: %iree_compile --compile-to=dispatch-creation --coralnpu-roofline-speedup-threshold=3 %s | FileCheck %s --check-prefix=CHECK-HIGH-SPEEDUP
// RUN: %iree_compile --compile-to=dispatch-creation --coralnpu-roofline-speedup-threshold=0 %s | FileCheck %s --check-prefix=CHECK-ZERO
// RUN: iree-compile --mlir-disable-threading --iree-hal-target-device=coralnpu --compile-to=dispatch-creation %s | FileCheck %s --check-prefix=CHECK-NO-HOST

// Speedups (T_cpu / T_npu) at the default constants are noted per case.
// Only dispatches on the other device cost a hop.

// CHECK-LABEL: @large_matmul
// CHECK-HIGH-SPEEDUP-LABEL: @large_matmul
func.func @large_matmul(
    %arg0: tensor<128x128xf32>,
    %arg1: tensor<128x128xf32>,
    %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> {
  // Large matmul, ~12.6.
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_1>
  // CHECK-HIGH-SPEEDUP: flow.dispatch.workgroups
  // CHECK-HIGH-SPEEDUP: stream.affinity = #hal.device.affinity<@__device_1>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2 : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// CHECK-LABEL: @int8_matmul
// CHECK-HIGH-SPEEDUP-LABEL: @int8_matmul
func.func @int8_matmul(
    %arg0: tensor<64x64xi8>,
    %arg1: tensor<64x64xi8>,
    %arg2: tensor<64x64xi8>) -> tensor<64x64xi8> {
  // 8-bit tier on both devices, ~1.5 (~5.9 if only CoralNPU got it).
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_1>
  // CHECK-HIGH-SPEEDUP: flow.dispatch.workgroups
  // CHECK-HIGH-SPEEDUP: stream.affinity = #hal.device.affinity<@__device_0>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<64x64xi8>, tensor<64x64xi8>)
                     outs(%arg2 : tensor<64x64xi8>) -> tensor<64x64xi8>
  return %0 : tensor<64x64xi8>
}

// CHECK-LABEL: @bf16_matmul
// CHECK-HIGH-SPEEDUP-LABEL: @bf16_matmul
func.func @bf16_matmul(
    %arg0: tensor<32x32xbf16>,
    %arg1: tensor<32x32xbf16>,
    %arg2: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  // Promoted to an f32 accumulator, so Zvt runs it at 2x (bf16 inputs) and the
  // host at 1x (no BF16 extension), ~5.9.
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_1>
  // CHECK-HIGH-SPEEDUP: flow.dispatch.workgroups
  // CHECK-HIGH-SPEEDUP: stream.affinity = #hal.device.affinity<@__device_1>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<32x32xbf16>, tensor<32x32xbf16>)
                     outs(%arg2 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %0 : tensor<32x32xbf16>
}

// CHECK-LABEL: @partial_tile_matmul
// CHECK-HIGH-SPEEDUP-LABEL: @partial_tile_matmul
func.func @partial_tile_matmul(
    %arg0: tensor<4x4096xf32>,
    %arg1: tensor<4096x8xf32>) -> tensor<4x8xf32> {
  // The 4x8 result fills 1/8 of a 16x16 Zvt tile, so Zvt runs at 1/8 of its
  // rate, ~2.3 (~3.04 at the full rate).
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_1>
  // CHECK-HIGH-SPEEDUP: flow.dispatch.workgroups
  // CHECK-HIGH-SPEEDUP: stream.affinity = #hal.device.affinity<@__device_0>
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<4x8xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<4x4096xf32>, tensor<4096x8xf32>)
                     outs(%fill : tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: @mixed_precision_matmul
// CHECK-HIGH-SPEEDUP-LABEL: @mixed_precision_matmul
func.func @mixed_precision_matmul(
    %arg0: tensor<32x32xbf16>,
    %arg1: tensor<32x32xbf16>) -> tensor<32x32xf32> {
  // Casts are not counted, ~1.4 (~1.5 if the two extf ops were).
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_1>
  // CHECK-HIGH-SPEEDUP: flow.dispatch.workgroups
  // CHECK-HIGH-SPEEDUP: stream.affinity = #hal.device.affinity<@__device_0>
  %empty = tensor.empty() : tensor<32x32xf32>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<32x32xbf16>, tensor<32x32xbf16>)
                     outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// CHECK-LABEL: @i16_i32_matmul
// CHECK-HIGH-SPEEDUP-LABEL: @i16_i32_matmul
func.func @i16_i32_matmul(
    %arg0: tensor<32x32xi16>,
    %arg1: tensor<32x32xi16>,
    %arg2: tensor<32x32xi32>) -> tensor<32x32xi32> {
  // 16-bit Zvt tier on CoralNPU vs 16-bit on host, ~2.3 (~0.69 on RVV).
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_1>
  // CHECK-HIGH-SPEEDUP: flow.dispatch.workgroups
  // CHECK-HIGH-SPEEDUP: stream.affinity = #hal.device.affinity<@__device_0>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<32x32xi16>, tensor<32x32xi16>)
                     outs(%arg2 : tensor<32x32xi32>) -> tensor<32x32xi32>
  return %0 : tensor<32x32xi32>
}

// CHECK-LABEL: @i16_i32_non_zvt_producer
// CHECK-HIGH-SPEEDUP-LABEL: @i16_i32_non_zvt_producer
func.func @i16_i32_non_zvt_producer(
    %arg0: tensor<32x32xi8>,
    %arg1: tensor<32x32xi16>,
    %arg2: tensor<32x32xi32>) -> tensor<32x32xi32> {
  // Fused arith.extui producer fails isZvtReadableInputs, so CoralNPU scores
  // it on RVV (~0.69 < 1.0) and keeps it on the host.
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_0>
  // CHECK-HIGH-SPEEDUP: flow.dispatch.workgroups
  // CHECK-HIGH-SPEEDUP: stream.affinity = #hal.device.affinity<@__device_0>
  %empty = tensor.empty() : tensor<32x32xi16>
  %lhs = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<32x32xi8>) outs(%empty : tensor<32x32xi16>) {
  ^bb0(%in: i8, %out: i16):
    %u = arith.extui %in : i8 to i16
    linalg.yield %u : i16
  } -> tensor<32x32xi16>
  %0 = linalg.matmul ins(%lhs, %arg1 : tensor<32x32xi16>, tensor<32x32xi16>)
                     outs(%arg2 : tensor<32x32xi32>) -> tensor<32x32xi32>
  return %0 : tensor<32x32xi32>
}

// CHECK-LABEL: @i16_i32_generic_extsi_producer
// CHECK-HIGH-SPEEDUP-LABEL: @i16_i32_generic_extsi_producer
func.func @i16_i32_generic_extsi_producer(
    %arg0: tensor<32x32xi16>,
    %arg1: tensor<32x32xi16>,
    %arg2: tensor<32x32xi32>) -> tensor<32x32xi32> {
  // getZvtLoadSource peels 1:1 extsi producers to 16-bit tier (~2.4 < 3.0;
  // ~4.8 >= 3.0 if scored as i32 without getZvtLoadSource).
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_1>
  // CHECK-HIGH-SPEEDUP: flow.dispatch.workgroups
  // CHECK-HIGH-SPEEDUP: stream.affinity = #hal.device.affinity<@__device_0>
  %e = tensor.empty() : tensor<32x32xi32>
  %lhs = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<32x32xi16>) outs(%e : tensor<32x32xi32>) {
  ^bb0(%in: i16, %out: i32):
    %s = arith.extsi %in : i16 to i32
    linalg.yield %s : i32
  } -> tensor<32x32xi32>
  %rhs = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg1 : tensor<32x32xi16>) outs(%e : tensor<32x32xi32>) {
  ^bb0(%in: i16, %out: i32):
    %s = arith.extsi %in : i16 to i32
    linalg.yield %s : i32
  } -> tensor<32x32xi32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%lhs, %rhs : tensor<32x32xi32>, tensor<32x32xi32>) outs(%arg2 : tensor<32x32xi32>) {
  ^bb0(%a: i32, %b: i32, %c: i32):
    %m = arith.muli %a, %b : i32
    %s = arith.addi %c, %m : i32
    linalg.yield %s : i32
  } -> tensor<32x32xi32>
  return %0 : tensor<32x32xi32>
}

// CHECK-LABEL: @chain
// CHECK-HIGH-SPEEDUP-LABEL: @chain
func.func @chain(
    %arg0: tensor<128x128xf32>,
    %arg1: tensor<128x128xf32>,
    %arg2: tensor<128x128xf32>,
    %arg3: tensor<64x256xf32>,
    %arg4: tensor<256x1xf32>,
    %arg5: tensor<64x1xf32>) -> (tensor<64x1xf32>, tensor<64x1xf32>) {
  // The same GEMV (~2.8 alone) stays on CoralNPU at threshold 3 when its
  // consumer or producer is there, even through a flow.tensor.reshape or
  // flow.tensor.update (~3.6), but goes to the host when its input comes from
  // the host sort (~1.5).
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_1>
  // CHECK: flow.tensor.reshape
  // CHECK: flow.tensor.update
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_1>
  // CHECK: flow.tensor.reshape
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_1>
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_0>
  // CHECK-HIGH-SPEEDUP: flow.dispatch.workgroups
  // CHECK-HIGH-SPEEDUP: stream.affinity = #hal.device.affinity<@__device_1>
  // CHECK-HIGH-SPEEDUP: flow.dispatch.workgroups
  // CHECK-HIGH-SPEEDUP: stream.affinity = #hal.device.affinity<@__device_1>
  // CHECK-HIGH-SPEEDUP: flow.dispatch.workgroups
  // CHECK-HIGH-SPEEDUP: stream.affinity = #hal.device.affinity<@__device_1>
  // CHECK-HIGH-SPEEDUP: flow.dispatch.workgroups
  // CHECK-HIGH-SPEEDUP: stream.affinity = #hal.device.affinity<@__device_0>
  // CHECK-HIGH-SPEEDUP: flow.dispatch.workgroups
  // CHECK-HIGH-SPEEDUP: stream.affinity = #hal.device.affinity<@__device_0>
  %pre = linalg.matmul ins(%arg3, %arg4 : tensor<64x256xf32>, tensor<256x1xf32>)
                       outs(%arg5 : tensor<64x1xf32>) -> tensor<64x1xf32>
  %pre_1d = tensor.collapse_shape %pre [[0, 1]] : tensor<64x1xf32> into tensor<64xf32>
  %arg0_1d = tensor.collapse_shape %arg0 [[0, 1]] : tensor<128x128xf32> into tensor<16384xf32>
  %upd_1d = tensor.insert_slice %pre_1d into %arg0_1d[0] [64] [1] : tensor<64xf32> into tensor<16384xf32>
  %lhs = tensor.expand_shape %upd_1d [[0, 1]] output_shape [128, 128] : tensor<16384xf32> into tensor<128x128xf32>
  %0 = linalg.matmul ins(%lhs, %arg1 : tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2 : tensor<128x128xf32>) -> tensor<128x128xf32>
  %1 = tensor.collapse_shape %0 [[0, 1]] : tensor<128x128xf32> into tensor<16384xf32>
  %2 = tensor.expand_shape %1 [[0, 1]] output_shape [64, 256] : tensor<16384xf32> into tensor<64x256xf32>
  %3 = linalg.matmul ins(%2, %arg4 : tensor<64x256xf32>, tensor<256x1xf32>)
                     outs(%arg5 : tensor<64x1xf32>) -> tensor<64x1xf32>
  %sorted = iree_linalg_ext.sort dimension(1) outs(%arg3 : tensor<64x256xf32>) {
  ^bb0(%lhs_elem: f32, %rhs_elem: f32):
    %cmp = arith.cmpf olt, %lhs_elem, %rhs_elem : f32
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<64x256xf32>
  %4 = linalg.matmul ins(%sorted, %arg4 : tensor<64x256xf32>, tensor<256x1xf32>)
                     outs(%arg5 : tensor<64x1xf32>) -> tensor<64x1xf32>
  return %3, %4 : tensor<64x1xf32>, tensor<64x1xf32>
}

// CHECK-LABEL: @small_matmul
// CHECK-HIGH-SPEEDUP-LABEL: @small_matmul
// CHECK-ZERO-LABEL: @small_matmul
// CHECK-NO-HOST-LABEL: @small_matmul
func.func @small_matmul(
    %arg0: tensor<2x2xf32>,
    %arg1: tensor<2x2xf32>,
    %arg2: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // Launch-bound, ~0.02; CoralNPU at 0 or without a host (where
  // CoralNPU is the only device, @__device_0).
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_0>
  // CHECK-ZERO: stream.affinity = #hal.device.affinity<@__device_1>
  // CHECK-NO-HOST: stream.affinity = #hal.device.affinity<@__device_0>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<2x2xf32>, tensor<2x2xf32>)
                     outs(%arg2 : tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// The extra labels bound the preceding checks.
// CHECK-LABEL: @unsupported_i64
// CHECK-ZERO-LABEL: @unsupported_i64
// CHECK-NO-HOST-LABEL: @unsupported_i64
func.func @unsupported_i64(
    %arg0: tensor<32x32xi64>,
    %arg1: tensor<32x32xi64>,
    %arg2: tensor<32x32xi64>) -> tensor<32x32xi64> {
  // i64 is not supported on CoralNPU, assigned to host device.
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_0>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<32x32xi64>, tensor<32x32xi64>)
                     outs(%arg2 : tensor<32x32xi64>) -> tensor<32x32xi64>
  return %0 : tensor<32x32xi64>
}

// CHECK-LABEL: @dynamic_shape
func.func @dynamic_shape(
    %arg0: tensor<?x?xf32>,
    %arg1: tensor<?x?xf32>,
    %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // Dynamic shapes are not supported on CoralNPU, assigned to host device.
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_0>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @sort
func.func @sort(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  // Non-linalg root (LinalgExt) is assigned to the host device.
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_0>
  %0 = iree_linalg_ext.sort dimension(0) outs(%arg0 : tensor<128xf32>) {
  ^bb0(%lhs: f32, %rhs: f32):
    %cmp = arith.cmpf olt, %lhs, %rhs : f32
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<128xf32>
  return %0 : tensor<128xf32>
}

// CHECK-LABEL: @elementwise_add
func.func @elementwise_add(
    %arg0: tensor<32x32xf32>,
    %arg1: tensor<32x32xf32>,
    %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // Supported on CoralNPU and memory-bound, ~1.7.
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_1>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<32x32xf32>, tensor<32x32xf32>)
    outs(%arg2 : tensor<32x32xf32>) {
  ^bb0(%in1: f32, %in2: f32, %out: f32):
    %add = arith.addf %in1, %in2 : f32
    linalg.yield %add : f32
  } -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

util.global private @weight = #stream.parameter.named<"model"::"weight"> : tensor<64x256xf32>
util.global private @big_weight = #stream.parameter.named<"model"::"big_weight"> : tensor<2048x4096xf32>

// CHECK-LABEL: @global_weight
func.func @global_weight(
    %arg0: tensor<256x1xf32>,
    %arg1: tensor<64x1xf32>) -> tensor<64x1xf32> {
  // The GEMV from @chain; a global weight costs no hop, ~2.8.
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_1>
  %w = util.global.load @weight : tensor<64x256xf32>
  %0 = linalg.matmul ins(%w, %arg0 : tensor<64x256xf32>, tensor<256x1xf32>)
                     outs(%arg1 : tensor<64x1xf32>) -> tensor<64x1xf32>
  return %0 : tensor<64x1xf32>
}

// CHECK-LABEL: @spill
func.func @spill(
    %arg0: tensor<1x2048xf32>,
    %arg1: tensor<1x4096xf32>) -> tensor<1x4096xf32> {
  // 32 MiB of weights exceed the 16 MiB I/O limit (~3.2 otherwise).
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_0>
  %w = util.global.load @big_weight : tensor<2048x4096xf32>
  %0 = linalg.matmul ins(%arg0, %w : tensor<1x2048xf32>, tensor<2048x4096xf32>)
                     outs(%arg1 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
  return %0 : tensor<1x4096xf32>
}

// CHECK-LABEL: @argmax
// CHECK-ZERO-LABEL: @argmax
func.func @argmax(%arg0: tensor<4096xf32>) -> tensor<i32> {
  // cmpf + selects have no single combiner, so CoralNPU would run a scalar
  // loop: it stays on the host even at threshold 0.
  // CHECK: flow.dispatch.workgroups
  // CHECK: stream.affinity = #hal.device.affinity<@__device_0>
  // CHECK-ZERO: flow.dispatch.workgroups
  // CHECK-ZERO: stream.affinity = #hal.device.affinity<@__device_0>
  %cst = arith.constant 0xFF800000 : f32
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<f32>
  %1 = tensor.empty() : tensor<i32>
  %2 = linalg.fill ins(%cst : f32) outs(%0 : tensor<f32>) -> tensor<f32>
  %3 = linalg.fill ins(%c0_i32 : i32) outs(%1 : tensor<i32>) -> tensor<i32>
  %4:2 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>,
                     affine_map<(d0) -> ()>],
    iterator_types = ["reduction"]
  } ins(%arg0 : tensor<4096xf32>) outs(%2, %3 : tensor<f32>, tensor<i32>) {
  ^bb0(%in: f32, %max: f32, %idx: i32):
    %i = linalg.index 0 : index
    %ii = arith.index_cast %i : index to i32
    %gt = arith.cmpf ogt, %in, %max : f32
    %m = arith.select %gt, %in, %max : f32
    %a = arith.select %gt, %ii, %idx : i32
    linalg.yield %m, %a : f32, i32
  } -> (tensor<f32>, tensor<i32>)
  return %4#1 : tensor<i32>
}
