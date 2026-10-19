// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(coralnpu-lower-dtcm-copies,canonicalize,cse)))))" \
// RUN:   %s | FileCheck %s

// CoralNPULowerDTCMCopies lowers the copies between memory and DTCM
// (#iree_codegen.workgroup_local) to loops of vector loads and stores along the
// rows. A vector fills the native vector size (32 bytes here: 8 x f32 or
// 32 x i8). The last vector of a row has a mask if the length of the rows is
// not a multiple of the vector size. The other copies do not change. The
// canonicalizer removes the loops with one iteration.

// A static row of 8 f32 from memory to DTCM: one full vector.
#target = #hal.executable.target<"coralnpu", "coralnpu-elf", {native_vector_size = 32 : i64}>
// CHECK-LABEL: func.func @copy_to_dtcm(
//  CHECK-SAME:     %[[SRC:.+]]: memref<8xf32, strided<[1], offset: ?>>, %[[DST:.+]]: memref<8xf32, #iree_codegen.workgroup_local>)
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-NOT:   scf.for
//       CHECK:   %[[V:.+]] = vector.load %[[SRC]][%[[C0]]] : memref<8xf32, strided<[1], offset: ?>>, vector<8xf32>
//       CHECK:   vector.store %[[V]], %[[DST]][%[[C0]]] : memref<8xf32, #iree_codegen.workgroup_local>, vector<8xf32>
//   CHECK-NOT:   linalg.generic
hal.executable private @copy_to_dtcm {
  hal.executable.variant public @coralnpu target(#target) {
    builtin.module {
      func.func @copy_to_dtcm(%src: memref<8xf32, strided<[1], offset: ?>>, %dst: memref<8xf32, #iree_codegen.workgroup_local>) {
        linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%src : memref<8xf32, strided<[1], offset: ?>>) outs(%dst : memref<8xf32, #iree_codegen.workgroup_local>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        }
        return
      }
    }
  }
}

// -----

// A tile with a dynamic number of rows of 160 f32: a loop over the rows, and a
// loop of 20 full vectors along each row.
#target = #hal.executable.target<"coralnpu", "coralnpu-elf", {native_vector_size = 32 : i64}>
// CHECK-LABEL: func.func @copy_rows_to_dtcm(
//  CHECK-SAME:     %[[SRC:.+]]: memref<?x160xf32, strided<[320, 1], offset: ?>>, %[[DST:.+]]: memref<?x160xf32, strided<[160, 1]>, #iree_codegen.workgroup_local>)
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
//   CHECK-DAG:   %[[C160:.+]] = arith.constant 160 : index
//       CHECK:   %[[ROWS:.+]] = memref.dim %[[SRC]], %[[C0]]
//       CHECK:   scf.for %[[I:.+]] = %[[C0]] to %[[ROWS]] step %[[C1]] {
//       CHECK:     scf.for %[[J:.+]] = %[[C0]] to %[[C160]] step %[[C8]] {
//       CHECK:       %[[V:.+]] = vector.load %[[SRC]][%[[I]], %[[J]]] : {{.*}}, vector<8xf32>
//       CHECK:       vector.store %[[V]], %[[DST]][%[[I]], %[[J]]] : {{.*}}, vector<8xf32>
hal.executable private @copy_rows_to_dtcm {
  hal.executable.variant public @coralnpu target(#target) {
    builtin.module {
      func.func @copy_rows_to_dtcm(%src: memref<?x160xf32, strided<[320, 1], offset: ?>>, %dst: memref<?x160xf32, strided<[160, 1]>, #iree_codegen.workgroup_local>) {
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%src : memref<?x160xf32, strided<[320, 1], offset: ?>>) outs(%dst : memref<?x160xf32, strided<[160, 1]>, #iree_codegen.workgroup_local>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        }
        return
      }
    }
  }
}

// -----

// Rows of 40 i8 (vectors of 32 i8): the last vector of each row has a mask.
#target = #hal.executable.target<"coralnpu", "coralnpu-elf", {native_vector_size = 32 : i64}>
// CHECK-LABEL: func.func @copy_masked_rows(
//  CHECK-SAME:     %[[SRC:.+]]: memref<4x40xi8, strided<[64, 1], offset: ?>>, %[[DST:.+]]: memref<4x40xi8, #iree_codegen.workgroup_local>)
//   CHECK-DAG:   %[[C32:.+]] = arith.constant 32 : index
//   CHECK-DAG:   %[[C40:.+]] = arith.constant 40 : index
//   CHECK-DAG:   %[[ZERO:.+]] = arith.constant dense<0> : vector<32xi8>
//       CHECK:   scf.for %[[I:.+]] =
//       CHECK:     scf.for %[[J:.+]] = %{{.+}} to %[[C40]] step %[[C32]] {
//       CHECK:       %[[REST:.+]] = arith.subi %[[C40]], %[[J]] : index
//       CHECK:       %[[LENGTH:.+]] = arith.minsi %[[REST]], %[[C32]] : index
//       CHECK:       %[[MASK:.+]] = vector.create_mask %[[LENGTH]] : vector<32xi1>
//       CHECK:       %[[V:.+]] = vector.maskedload %[[SRC]][%[[I]], %[[J]]], %[[MASK]], %[[ZERO]]
//       CHECK:       vector.maskedstore %[[DST]][%[[I]], %[[J]]], %[[MASK]], %[[V]]
hal.executable private @copy_masked_rows {
  hal.executable.variant public @coralnpu target(#target) {
    builtin.module {
      func.func @copy_masked_rows(%src: memref<4x40xi8, strided<[64, 1], offset: ?>>, %dst: memref<4x40xi8, #iree_codegen.workgroup_local>) {
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%src : memref<4x40xi8, strided<[64, 1], offset: ?>>) outs(%dst : memref<4x40xi8, #iree_codegen.workgroup_local>) {
        ^bb0(%in: i8, %out: i8):
          linalg.yield %in : i8
        }
        return
      }
    }
  }
}

// -----

// A staged result from DTCM to memory, with a dynamic row length (a
// memref.copy): masked vectors.
#target = #hal.executable.target<"coralnpu", "coralnpu-elf", {native_vector_size = 32 : i64}>
// CHECK-LABEL: func.func @copy_from_dtcm(
//  CHECK-SAME:     %[[SRC:.+]]: memref<2x?xf32, strided<[32, 1]>, #iree_codegen.workgroup_local>, %[[DST:.+]]: memref<2x?xf32, strided<[450, 1], offset: ?>>)
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
//       CHECK:   %[[ROW:.+]] = memref.dim %[[SRC]], %[[C1]]
//       CHECK:   scf.for %[[I:.+]] =
//       CHECK:     scf.for %[[J:.+]] = %{{.+}} to %[[ROW]] step %[[C8]] {
//       CHECK:       %[[REST:.+]] = arith.subi %[[ROW]], %[[J]] : index
//       CHECK:       %[[LENGTH:.+]] = arith.minsi %[[REST]], %[[C8]] : index
//       CHECK:       %[[MASK:.+]] = vector.create_mask %[[LENGTH]] : vector<8xi1>
//       CHECK:       %[[V:.+]] = vector.maskedload %[[SRC]][%[[I]], %[[J]]], %[[MASK]]
//       CHECK:       vector.maskedstore %[[DST]][%[[I]], %[[J]]], %[[MASK]], %[[V]]
//   CHECK-NOT:   memref.copy
hal.executable private @copy_from_dtcm {
  hal.executable.variant public @coralnpu target(#target) {
    builtin.module {
      func.func @copy_from_dtcm(%src: memref<2x?xf32, strided<[32, 1]>, #iree_codegen.workgroup_local>, %dst: memref<2x?xf32, strided<[450, 1], offset: ?>>) {
        memref.copy %src, %dst : memref<2x?xf32, strided<[32, 1]>, #iree_codegen.workgroup_local> to memref<2x?xf32, strided<[450, 1], offset: ?>>
        return
      }
    }
  }
}

// -----

// Copies that do not touch DTCM, or whose rows are not contiguous, do not
// change.
#target = #hal.executable.target<"coralnpu", "coralnpu-elf", {native_vector_size = 32 : i64}>
// CHECK-LABEL: func.func @other_copies(
//   CHECK-NOT:   vector.load
//       CHECK:   memref.copy
//       CHECK:   linalg.generic
//   CHECK-NOT:   vector.load
//       CHECK:   return
hal.executable private @other_copies {
  hal.executable.variant public @coralnpu target(#target) {
    builtin.module {
      func.func @other_copies(%a: memref<8xf32>, %b: memref<8xf32>, %column: memref<4xf32, strided<[160], offset: ?>>, %c: memref<4xf32, #iree_codegen.workgroup_local>) {
        memref.copy %a, %b : memref<8xf32> to memref<8xf32>
        linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%column : memref<4xf32, strided<[160], offset: ?>>) outs(%c : memref<4xf32, #iree_codegen.workgroup_local>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        }
        return
      }
    }
  }
}

// -----

// Rows of 4 f32, shorter than the native vector (8 x f32): one short vector of
// 4 f32 for each row, without a mask.
#target = #hal.executable.target<"coralnpu", "coralnpu-elf", {native_vector_size = 32 : i64}>
// CHECK-LABEL: func.func @copy_short_rows(
//  CHECK-SAME:     %[[SRC:.+]]: memref<3x4xf32, strided<[64, 1], offset: ?>>, %[[DST:.+]]: memref<3x4xf32, #iree_codegen.workgroup_local>)
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//       CHECK:   scf.for %[[I:.+]] = %[[C0]] to %[[C3]] step %[[C1]] {
//  CHECK-NEXT:     %[[V:.+]] = vector.load %[[SRC]][%[[I]], %[[C0]]] : memref<3x4xf32, strided<[64, 1], offset: ?>>, vector<4xf32>
//  CHECK-NEXT:     vector.store %[[V]], %[[DST]][%[[I]], %[[C0]]] : memref<3x4xf32, #iree_codegen.workgroup_local>, vector<4xf32>
//   CHECK-NOT:   vector.maskedload
hal.executable private @copy_short_rows {
  hal.executable.variant public @coralnpu target(#target) {
    builtin.module {
      func.func @copy_short_rows(%src: memref<3x4xf32, strided<[64, 1], offset: ?>>, %dst: memref<3x4xf32, #iree_codegen.workgroup_local>) {
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%src : memref<3x4xf32, strided<[64, 1], offset: ?>>) outs(%dst : memref<3x4xf32, #iree_codegen.workgroup_local>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        }
        return
      }
    }
  }
}

// -----

// Rows of 3 f32, shorter than the native vector (8 x f32) and not a power of
// 2: one masked power-of-2 vector of 4 f32 for each row.
#target = #hal.executable.target<"coralnpu", "coralnpu-elf", {native_vector_size = 32 : i64}>
// CHECK-LABEL: func.func @copy_short_masked_rows(
//  CHECK-SAME:     %[[SRC:.+]]: memref<2x3xf32, strided<[64, 1], offset: ?>>, %[[DST:.+]]: memref<2x3xf32, #iree_codegen.workgroup_local>)
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
//       CHECK:   scf.for %[[I:.+]] = %[[C0]] to %[[C2]] step %[[C1]] {
//  CHECK-NEXT:     %[[MASK:.+]] = vector.constant_mask [3] : vector<4xi1>
//  CHECK-NEXT:     %[[V:.+]] = vector.maskedload %[[SRC]][%[[I]], %[[C0]]], %[[MASK]], %[[ZERO]] : memref<2x3xf32, strided<[64, 1], offset: ?>>, vector<4xi1>, vector<4xf32> into vector<4xf32>
//  CHECK-NEXT:     vector.maskedstore %[[DST]][%[[I]], %[[C0]]], %[[MASK]], %[[V]] : memref<2x3xf32, #iree_codegen.workgroup_local>, vector<4xi1>, vector<4xf32>
hal.executable private @copy_short_masked_rows {
  hal.executable.variant public @coralnpu target(#target) {
    builtin.module {
      func.func @copy_short_masked_rows(%src: memref<2x3xf32, strided<[64, 1], offset: ?>>, %dst: memref<2x3xf32, #iree_codegen.workgroup_local>) {
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%src : memref<2x3xf32, strided<[64, 1], offset: ?>>) outs(%dst : memref<2x3xf32, #iree_codegen.workgroup_local>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        }
        return
      }
    }
  }
}

// -----

// A target without native_vector_size: the vectors have 16 bytes (4 x f32), so
// a row of 8 f32 takes 2 full vectors.
#target = #hal.executable.target<"coralnpu", "coralnpu-elf", {}>
// CHECK-LABEL: func.func @copy_default_vector_size(
//  CHECK-SAME:     %[[SRC:.+]]: memref<2x8xf32, strided<[64, 1], offset: ?>>, %[[DST:.+]]: memref<2x8xf32, #iree_codegen.workgroup_local>)
//   CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//   CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
//       CHECK:   scf.for %[[I:.+]] =
//       CHECK:     scf.for %[[J:.+]] = %{{.+}} to %[[C8]] step %[[C4]] {
//       CHECK:       %[[V:.+]] = vector.load %[[SRC]][%[[I]], %[[J]]] : {{.*}}, vector<4xf32>
//       CHECK:       vector.store %[[V]], %[[DST]][%[[I]], %[[J]]] : {{.*}}, vector<4xf32>
hal.executable private @copy_default_vector_size {
  hal.executable.variant public @coralnpu target(#target) {
    builtin.module {
      func.func @copy_default_vector_size(%src: memref<2x8xf32, strided<[64, 1], offset: ?>>, %dst: memref<2x8xf32, #iree_codegen.workgroup_local>) {
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%src : memref<2x8xf32, strided<[64, 1], offset: ?>>) outs(%dst : memref<2x8xf32, #iree_codegen.workgroup_local>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        }
        return
      }
    }
  }
}

// -----

// A copy without elements (rows of length 0) gets no vectors. (The
// canonicalizer then erases the copy.)
#target = #hal.executable.target<"coralnpu", "coralnpu-elf", {native_vector_size = 32 : i64}>
// CHECK-LABEL: func.func @copy_no_elements(
//   CHECK-NOT:   vector.
//       CHECK:   return
hal.executable private @copy_no_elements {
  hal.executable.variant public @coralnpu target(#target) {
    builtin.module {
      func.func @copy_no_elements(%src: memref<2x0xf32, strided<[64, 1], offset: ?>>, %dst: memref<2x0xf32, #iree_codegen.workgroup_local>) {
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%src : memref<2x0xf32, strided<[64, 1], offset: ?>>) outs(%dst : memref<2x0xf32, #iree_codegen.workgroup_local>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        }
        return
      }
    }
  }
}
