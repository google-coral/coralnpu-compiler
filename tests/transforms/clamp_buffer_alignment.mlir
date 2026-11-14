// RUN: iree-opt \
// RUN:   --pass-pipeline="builtin.module(coralnpu-clamp-buffer-alignment)" \
// RUN:   %s | FileCheck %s

// Positive cases: explicit alignments > 16 (32, 64) on scalar/complex
// memref.alloc, memref.alloca (including nested inside scf.for), memref.global,
// and llvm.mlir.global clamp to 16 bytes.

// CHECK-LABEL: memref.global "private" constant @global_i8_align64 : memref<32xi8> = dense<1> alignment = 16
memref.global "private" constant @global_i8_align64 : memref<32xi8> = dense<1> alignment = 64

// CHECK-LABEL: memref.global "private" constant @global_f32_align32 : memref<8xf32> = dense<0.000000e+00> alignment = 16
memref.global "private" constant @global_f32_align32 : memref<8xf32> = dense<0.0> alignment = 32

// CHECK-LABEL: llvm.mlir.global private constant @llvm_global_i8_align64(dense<1> : tensor<32xi8>) {addr_space = 0 : i32, alignment = 16 : i64} : !llvm.array<32 x i8>
llvm.mlir.global private constant @llvm_global_i8_align64(dense<1> : tensor<32xi8>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<32 x i8>

// CHECK-LABEL: llvm.mlir.global private constant @llvm_global_nested_array_align32() {addr_space = 0 : i32, alignment = 16 : i64} : !llvm.array<4 x array<4 x i8>>
llvm.mlir.global private constant @llvm_global_nested_array_align32() {addr_space = 0 : i32, alignment = 32 : i64} : !llvm.array<4 x array<4 x i8>>

// Negative global cases: unset alignment, alignment <= 16, non-constant
// `memref.global` / `llvm.mlir.global`, or VectorType element (including inside
// LLVMArrayType or LLVMStructType) remain unchanged.

// CHECK-LABEL: memref.global "private" constant @global_no_align : memref<16xi8> = dense<0>
// CHECK-NOT: alignment
memref.global "private" constant @global_no_align : memref<16xi8> = dense<0>

// CHECK-LABEL: memref.global "private" constant @global_align8 : memref<16xi8> = dense<0> alignment = 8
memref.global "private" constant @global_align8 : memref<16xi8> = dense<0> alignment = 8

// CHECK-LABEL: memref.global "private" @global_mutable_align64 : memref<32xi8> = uninitialized alignment = 64
memref.global "private" @global_mutable_align64 : memref<32xi8> = uninitialized alignment = 64

// CHECK-LABEL: memref.global "private" constant @global_vector_align64 : memref<2xvector<8xf32>> = uninitialized alignment = 64
memref.global "private" constant @global_vector_align64 : memref<2xvector<8xf32>> = uninitialized alignment = 64

// CHECK-LABEL: llvm.mlir.global private constant @llvm_global_no_align(dense<0> : tensor<16xi8>) {addr_space = 0 : i32} : !llvm.array<16 x i8>
// CHECK-NOT: alignment
llvm.mlir.global private constant @llvm_global_no_align(dense<0> : tensor<16xi8>) {addr_space = 0 : i32} : !llvm.array<16 x i8>

// CHECK-LABEL: llvm.mlir.global private constant @llvm_global_align4(dense<0> : tensor<4xi8>) {addr_space = 0 : i32, alignment = 4 : i64} : !llvm.array<4 x i8>
llvm.mlir.global private constant @llvm_global_align4(dense<0> : tensor<4xi8>) {addr_space = 0 : i32, alignment = 4 : i64} : !llvm.array<4 x i8>

// CHECK-LABEL: llvm.mlir.global private @llvm_global_mutable_align64() {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<32 x i8>
llvm.mlir.global private @llvm_global_mutable_align64() {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<32 x i8>

// CHECK-LABEL: llvm.mlir.global private constant @llvm_global_vec_align64() {addr_space = 0 : i32, alignment = 64 : i64} : vector<8xf32>
llvm.mlir.global private constant @llvm_global_vec_align64() {addr_space = 0 : i32, alignment = 64 : i64} : vector<8xf32>

// CHECK-LABEL: llvm.mlir.global private constant @llvm_global_array_vec_align64() {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<2 x vector<8xf32>>
llvm.mlir.global private constant @llvm_global_array_vec_align64() {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<2 x vector<8xf32>>

// CHECK-LABEL: llvm.mlir.global private constant @llvm_global_struct_vec_align64() {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.struct<(i32, vector<8xf32>)>
llvm.mlir.global private constant @llvm_global_struct_vec_align64() {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.struct<(i32, vector<8xf32>)>

// CHECK-LABEL: func.func @clamp_alloc_and_alloca()
func.func @clamp_alloc_and_alloca() {
  // CHECK: memref.alloc() alignment = 16 : memref<32xi8, #iree_codegen.workgroup_local>
  %alloc_dtcm_64 = memref.alloc() alignment = 64 : memref<32xi8, #iree_codegen.workgroup_local>
  // CHECK: memref.alloc() alignment = 16 : memref<16xf32>
  %alloc_default_32 = memref.alloc() alignment = 32 : memref<16xf32>
  // CHECK: memref.alloc() alignment = 16 : memref<4xcomplex<f32>>
  %alloc_complex_64 = memref.alloc() alignment = 64 : memref<4xcomplex<f32>>

  // CHECK: memref.alloca() alignment = 16 : memref<16xi8>
  %alloca_64 = memref.alloca() alignment = 64 : memref<16xi8>
  // CHECK: memref.alloca() alignment = 16 : memref<8xf32>
  %alloca_32 = memref.alloca() alignment = 32 : memref<8xf32>

  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  // CHECK: scf.for
  scf.for %i = %c0 to %c4 step %c1 {
    // CHECK: memref.alloca() alignment = 16 : memref<8xf32>
    %nested_alloca = memref.alloca() alignment = 32 : memref<8xf32>
  }
  return
}

// CHECK-LABEL: func.func @preserve_small_and_vector_alignments()
func.func @preserve_small_and_vector_alignments() {
  // Unset or <= 16 B alignments stay unchanged.
  // CHECK: memref.alloc() : memref<32xi8>
  %alloc_none = memref.alloc() : memref<32xi8>
  // CHECK: memref.alloca() : memref<4xi8>
  %alloca_none = memref.alloca() : memref<4xi8>
  // CHECK: memref.alloca() alignment = 1 : memref<4xi8>
  %alloca_1 = memref.alloca() alignment = 1 : memref<4xi8>
  // CHECK: memref.alloca() alignment = 4 : memref<4xi32>
  %alloca_4 = memref.alloca() alignment = 4 : memref<4xi32>
  // CHECK: memref.alloc() alignment = 8 : memref<4xi64>
  %alloc_8 = memref.alloc() alignment = 8 : memref<4xi64>
  // CHECK: memref.alloc() alignment = 16 : memref<16xi8>
  %alloc_16 = memref.alloc() alignment = 16 : memref<16xi8>

  // Vector element types stay unchanged.
  // CHECK: memref.alloc() alignment = 64 : memref<2xvector<8xf32>>
  %alloc_vec = memref.alloc() alignment = 64 : memref<2xvector<8xf32>>
  // CHECK: memref.alloca() alignment = 32 : memref<1xvector<8xf32>>
  %alloca_vec = memref.alloca() alignment = 32 : memref<1xvector<8xf32>>

  // Buffers used by vector.type_cast stay unchanged.
  // CHECK: %[[CAST_ALLOC:.*]] = memref.alloc() alignment = 64 : memref<2x4xf32>
  // CHECK: vector.type_cast %[[CAST_ALLOC]] : memref<2x4xf32> to memref<vector<2x4xf32>>
  %alloc_type_cast = memref.alloc() alignment = 64 : memref<2x4xf32>
  %cast0 = vector.type_cast %alloc_type_cast : memref<2x4xf32> to memref<vector<2x4xf32>>

  // CHECK: %[[CAST_ALLOCA:.*]] = memref.alloca() alignment = 32 : memref<2x4xf32>
  // CHECK: vector.type_cast %[[CAST_ALLOCA]] : memref<2x4xf32> to memref<vector<2x4xf32>>
  %alloca_type_cast = memref.alloca() alignment = 32 : memref<2x4xf32>
  %cast1 = vector.type_cast %alloca_type_cast : memref<2x4xf32> to memref<vector<2x4xf32>>
  return
}

// Operations nested inside llvm.func are skipped by the pre-order walk.
// CHECK-LABEL: llvm.func @skip_llvm_func_body()
llvm.func @skip_llvm_func_body() {
  // CHECK: memref.alloca() alignment = 64 : memref<16xi8>
  %0 = memref.alloca() alignment = 64 : memref<16xi8>
  llvm.return
}
