// RUN: iree-opt --pass-pipeline='builtin.module(coralnpu-raise-apply-scale)' %s | FileCheck %s

// A per-channel rescale: dynamic multiplier and shift, with the double round
// selected when shift > 31.
// CHECK-LABEL: func.func @dynamic_double_round
// CHECK: %[[R:.*]] = tosa.apply_scale %arg0, %arg1, %arg2 rounding_mode<DOUBLE_ROUND> : (i32, i32, i8) -> i32
// CHECK: return %[[R]] : i32
func.func @dynamic_double_round(%v: i32, %m: i32, %s: i8) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c31_i32 = arith.constant 31 : i32
  %c1_i64 = arith.constant 1 : i64
  %c2p30 = arith.constant 1073741824 : i64
  %cm2p30 = arith.constant -1073741824 : i64
  %s32 = arith.extui %s : i8 to i32
  %v64 = arith.extsi %v : i32 to i64
  %m64 = arith.extsi %m : i32 to i64
  %mul = arith.muli %v64, %m64 : i64
  %s64 = arith.extui %s : i8 to i64
  %shl = arith.shli %c1_i64, %s64 : i64
  %round = arith.shrui %shl, %c1_i64 : i64
  %base = arith.addi %mul, %round : i64
  %positive = arith.cmpi sge, %v, %c0_i32 : i32
  %dir = arith.select %positive, %c2p30, %cm2p30 : i64
  %double = arith.addi %dir, %base : i64
  %over31 = arith.cmpi sgt, %s32, %c31_i32 : i32
  %acc = arith.select %over31, %double, %base : i64
  %shr = arith.shrsi %acc, %s64 : i64
  %r = arith.trunci %shr : i64 to i32
  return %r : i32
}

// Per-tensor rescales of an i8 value (an input with a zero point of 0), which
// tosa.apply_scale takes extended to i32: one with shift 33 > 31, so the double
// round is always applied, and one with shift 32, where trunci(shrsi) folds to
// trunci(shrui).
// CHECK-LABEL: func.func @constant_double_round
// CHECK-DAG: %[[M:.*]] = arith.constant 1435305426 : i32
// CHECK-DAG: %[[S33:.*]] = arith.constant 33 : i8
// CHECK-DAG: %[[S32:.*]] = arith.constant 32 : i8
// CHECK: %[[V0:.*]] = arith.extsi %arg0 : i8 to i32
// CHECK: %[[R0:.*]] = tosa.apply_scale %[[V0]], %[[M]], %[[S33]] rounding_mode<DOUBLE_ROUND> : (i32, i32, i8) -> i32
// CHECK: %[[V1:.*]] = arith.extsi %arg0 : i8 to i32
// CHECK: %[[R1:.*]] = tosa.apply_scale %[[V1]], %[[M]], %[[S32]] rounding_mode<DOUBLE_ROUND> : (i32, i32, i8) -> i32
// CHECK: return %[[R0]], %[[R1]] : i32, i32
func.func @constant_double_round(%v: i8) -> (i32, i32) {
  %c0_i32 = arith.constant 0 : i32
  %cm = arith.constant 1435305426 : i64
  %c2p30 = arith.constant 1073741824 : i64
  %cm2p30 = arith.constant -1073741824 : i64
  %c2p31 = arith.constant 2147483648 : i64
  %c2p32 = arith.constant 4294967296 : i64
  %c32 = arith.constant 32 : i64
  %c33 = arith.constant 33 : i64
  %v32 = arith.extsi %v : i8 to i32
  %v64 = arith.extsi %v : i8 to i64
  %mul = arith.muli %v64, %cm : i64
  %positive = arith.cmpi sge, %v32, %c0_i32 : i32
  %dir = arith.select %positive, %c2p30, %cm2p30 : i64
  %base0 = arith.addi %mul, %c2p32 : i64
  %acc0 = arith.addi %dir, %base0 : i64
  %shr0 = arith.shrsi %acc0, %c33 : i64
  %r0 = arith.trunci %shr0 : i64 to i32
  %base1 = arith.addi %mul, %c2p31 : i64
  %acc1 = arith.addi %dir, %base1 : i64
  %shr1 = arith.shrui %acc1, %c32 : i64
  %r1 = arith.trunci %shr1 : i64 to i32
  return %r0, %r1 : i32, i32
}

// A per-tensor rescale of an i8 value with shift 10 <= 31, where double and
// single round are the same.
// CHECK-LABEL: func.func @constant_single_round
// CHECK-DAG: %[[S:.*]] = arith.constant 10 : i8
// CHECK-DAG: %[[V:.*]] = arith.extsi %arg0 : i8 to i32
// CHECK: tosa.apply_scale %[[V]], %{{.*}}, %[[S]] rounding_mode<SINGLE_ROUND> : (i32, i32, i8) -> i32
func.func @constant_single_round(%v: i8) -> i32 {
  %cm = arith.constant 1073741824 : i64
  %cr = arith.constant 512 : i64
  %c10 = arith.constant 10 : i64
  %v64 = arith.extsi %v : i8 to i64
  %mul = arith.muli %v64, %cm : i64
  %base = arith.addi %mul, %cr : i64
  %shr = arith.shrsi %base, %c10 : i64
  %r = arith.trunci %shr : i64 to i32
  return %r : i32
}

// A rounding constant other than (1 << shift) >> 1, or a double round that
// does not select by the sign of the value, is not a rescale.
// CHECK-LABEL: func.func @not_a_rescale
// CHECK-NOT: tosa.apply_scale
// CHECK: return
func.func @not_a_rescale(%v: i32, %positive: i1) -> (i32, i32) {
  %cm = arith.constant 1073741824 : i64
  %cr = arith.constant 513 : i64
  %c10 = arith.constant 10 : i64
  %v64 = arith.extsi %v : i32 to i64
  %mul = arith.muli %v64, %cm : i64
  %base = arith.addi %mul, %cr : i64
  %shr = arith.shrsi %base, %c10 : i64
  %r = arith.trunci %shr : i64 to i32
  %c2p30 = arith.constant 1073741824 : i64
  %cm2p30 = arith.constant -1073741824 : i64
  %c2p32 = arith.constant 4294967296 : i64
  %c33 = arith.constant 33 : i64
  %dir = arith.select %positive, %c2p30, %cm2p30 : i64
  %base1 = arith.addi %mul, %c2p32 : i64
  %acc1 = arith.addi %dir, %base1 : i64
  %shr1 = arith.shrsi %acc1, %c33 : i64
  %r1 = arith.trunci %shr1 : i64 to i32
  return %r, %r1 : i32, i32
}

// An identity scf.if (d == 1) broadcast (after folding tensor.dim of
// tensor.expand_shape) folds to its source, while an scf.if conditioned on an
// unrelated dimension is preserved.
// CHECK-LABEL: func.func @fold_identity_broadcast_if
// CHECK: %[[EXP:.*]] = tensor.expand_shape %arg0 {{\[\[}}0, 1], [2]] output_shape [%arg1, 8, 16]
// CHECK: %[[KEEP:.*]] = scf.if
// CHECK-NOT: scf.if
// CHECK: return %[[EXP]], %[[KEEP]] : tensor<?x8x16xi32>, tensor<?x8x16xi32>
func.func @fold_identity_broadcast_if(%flat: tensor<?x16xi32>, %d: index, %other: index) -> (tensor<?x8x16xi32>, tensor<?x8x16xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %x = tensor.expand_shape %flat [[0, 1], [2]] output_shape [%d, 8, 16] : tensor<?x16xi32> into tensor<?x8x16xi32>
  %dim = tensor.dim %x, %c0 : tensor<?x8x16xi32>
  %cmp = arith.cmpi eq, %dim, %c1 : index
  %e = tensor.empty(%dim) : tensor<?x8x16xi32>
  %r = scf.if %cmp -> (tensor<?x8x16xi32>) {
    %b = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%x : tensor<?x8x16xi32>) outs(%e : tensor<?x8x16xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<?x8x16xi32>
    scf.yield %b : tensor<?x8x16xi32>
  } else {
    scf.yield %x : tensor<?x8x16xi32>
  }
  %cmp_bad = arith.cmpi eq, %other, %c1 : index
  %e_bad = tensor.empty(%other) : tensor<?x8x16xi32>
  %keep = scf.if %cmp_bad -> (tensor<?x8x16xi32>) {
    %b = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%x : tensor<?x8x16xi32>) outs(%e_bad : tensor<?x8x16xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<?x8x16xi32>
    scf.yield %b : tensor<?x8x16xi32>
  } else {
    scf.yield %x : tensor<?x8x16xi32>
  }
  return %r, %keep : tensor<?x8x16xi32>, tensor<?x8x16xi32>
}
