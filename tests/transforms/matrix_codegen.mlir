// RUN: iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(coralnpu-matrix-codegen)))))" %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(coralnpu-matrix-codegen)))))" %s | FileCheck %s --check-prefix=NOVSET

// Whole-file invariant: any vset* clears mtype and vtype.altfmt
// (RvvFrontEnd.sv), so the pass must emit none at all.
// NOVSET-NOT: vsetivli
// NOVSET-NOT: vsetvli
// NOVSET-NOT: vsetvl

#target = #hal.executable.target<"coralnpu", "coralnpu-elf", {
  cpu_features = "+zvtbase"
}>

// Same backend, but the Zvt feature is absent.
#target_no_zvt = #hal.executable.target<"coralnpu", "coralnpu-elf", {
  cpu_features = "+v"
}>

// This pass is registered in the shared LLVMCPU pipeline, so it must leave
// non-CoralNPU targets untouched.
#target_llvmcpu = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  cpu_features = "+avx512f"
}>

// CHECK-LABEL: hal.executable private @matmul
module {
  hal.executable private @matmul {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul
        func.func @matmul(%arg0: memref<16x1xf32>, %arg1: memref<1x16xf32>, %arg2: memref<16x16xf32>) {
          %c0 = arith.constant 0 : index
          %cst_0 = arith.constant 0.0 : f32

          %lhs = vector.transfer_read %arg0[%c0, %c0], %cst_0 : memref<16x1xf32>, vector<16x1xf32>
          %rhs = vector.transfer_read %arg1[%c0, %c0], %cst_0 : memref<1x16xf32>, vector<1x16xf32>
          %acc = vector.transfer_read %arg2[%c0, %c0], %cst_0 : memref<16x16xf32>, vector<16x16xf32>

          // Every block programs its own mtype/vtype; any vset* would clear
          // them.
          // CHECK: llvm.inline_asm {{.*}}li t5, 16417{{.*}}li t6, 18{{.*}}msetmtype t5, t6{{.*}}msettn zero, t6{{.*}}vtzero mt0{{.*}}vtfmm.tvv mt0
          // CHECK-NOT: vsetivli
          %res = vector.contract {
            indexing_maps = [
              affine_map<(d0, d1, d2) -> (d0, d2)>,
              affine_map<(d0, d1, d2) -> (d2, d1)>,
              affine_map<(d0, d1, d2) -> (d0, d1)>
            ],
            iterator_types = ["parallel", "parallel", "reduction"]
          } %lhs, %rhs, %acc : vector<16x1xf32>, vector<1x16xf32> into vector<16x16xf32>

          // Writeback drains mt0 row by row: vtmv.v.t + vse32.v, 16 times.
          // CHECK: llvm.inline_asm {{.*}}msetmtype t5, t6{{.*}}li t0, 0{{.*}}vtmv.v.t v0, t0{{.*}}vse32.v v0, (t4){{.*}}add t4, t4, t5{{.*}}addi t0, t0, 1{{.*}}bnez t6
          vector.transfer_write %res, %arg2[%c0, %c0] : vector<16x16xf32>, memref<16x16xf32>
          return
        }
      }
    }
  }

  // CHECK-LABEL: hal.executable private @matmul_loop
  hal.executable private @matmul_loop {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_loop
        func.func @matmul_loop(%arg0: memref<16x8xf32>, %arg1: memref<8x16xf32>, %arg2: memref<16x16xf32>) {
          // CHECK: llvm.inline_asm {{.*}}li t5, 16417{{.*}}msetmtype t5, t6{{.*}}vtzero mt0{{.*}}vlse32.v{{.*}}vtfmm.tvv mt0{{.*}}vse32.v
          %c0 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c1 = arith.constant 1 : index
          %cst_0 = arith.constant 0.0 : f32

          %acc_init = vector.transfer_read %arg2[%c0, %c0], %cst_0 : memref<16x16xf32>, vector<16x16xf32>
          %res = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %acc_init) -> (vector<16x16xf32>) {
            %lhs = vector.transfer_read %arg0[%c0, %iv], %cst_0 : memref<16x8xf32>, vector<16x1xf32>
            %rhs = vector.transfer_read %arg1[%iv, %c0], %cst_0 : memref<8x16xf32>, vector<1x16xf32>
            %next = vector.contract {
              indexing_maps = [
                affine_map<(d0, d1, d2) -> (d0, d2)>,
                affine_map<(d0, d1, d2) -> (d2, d1)>,
                affine_map<(d0, d1, d2) -> (d0, d1)>
              ],
              iterator_types = ["parallel", "parallel", "reduction"]
            } %lhs, %rhs, %acc : vector<16x1xf32>, vector<1x16xf32> into vector<16x16xf32>
            scf.yield %next : vector<16x16xf32>
          }
          vector.transfer_write %res, %arg2[%c0, %c0] : vector<16x16xf32>, memref<16x16xf32>
          return
        }
      }
    }
  }

  // Writeback via unrolled vector.extract + vector.store instead of
  // vector.transfer_write. The destination address comes from the row-0 store;
  // every extract/store in the chain is replaced by the tile drain.
  // CHECK-LABEL: hal.executable private @matmul_loop_extract_store
  hal.executable private @matmul_loop_extract_store {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_loop_extract_store
        func.func @matmul_loop_extract_store(%arg0: memref<16x8xf32>, %arg1: memref<8x16xf32>, %arg2: memref<16x16xf32>) {
          // CHECK: llvm.inline_asm {{.*}}vtzero mt0{{.*}}vtfmm.tvv mt0{{.*}}vtmv.v.t v0, t0{{.*}}vse32.v
          // CHECK-NOT: vector.extract
          // CHECK-NOT: vector.store
          // CHECK-NOT: scf.for
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c8 = arith.constant 8 : index
          %cst_0 = arith.constant 0.0 : f32

          %acc_init = vector.transfer_read %arg2[%c0, %c0], %cst_0 : memref<16x16xf32>, vector<16x16xf32>
          %res = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %acc_init) -> (vector<16x16xf32>) {
            %lhs = vector.transfer_read %arg0[%c0, %iv], %cst_0 : memref<16x8xf32>, vector<16x1xf32>
            %rhs = vector.transfer_read %arg1[%iv, %c0], %cst_0 : memref<8x16xf32>, vector<1x16xf32>
            %next = vector.contract {
              indexing_maps = [
                affine_map<(d0, d1, d2) -> (d0, d2)>,
                affine_map<(d0, d1, d2) -> (d2, d1)>,
                affine_map<(d0, d1, d2) -> (d0, d1)>
              ],
              iterator_types = ["parallel", "parallel", "reduction"]
            } %lhs, %rhs, %acc : vector<16x1xf32>, vector<1x16xf32> into vector<16x16xf32>
            scf.yield %next : vector<16x16xf32>
          }
          %row0 = vector.extract %res[0] : vector<16xf32> from vector<16x16xf32>
          vector.store %row0, %arg2[%c0, %c0] : memref<16x16xf32>, vector<16xf32>
          %row1 = vector.extract %res[1] : vector<16xf32> from vector<16x16xf32>
          vector.store %row1, %arg2[%c1, %c0] : memref<16x16xf32>, vector<16xf32>
          return
        }
      }
    }
  }

  // The accumulator is seeded from %arg3 but drained to %arg2, so the loop
  // cannot be folded into one microkernel. The loop survives and the tiles are
  // zeroed once before it instead of inside the multiply block.
  // CHECK-LABEL: hal.executable private @matmul_loop_unfused
  hal.executable private @matmul_loop_unfused {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_loop_unfused
        func.func @matmul_loop_unfused(%arg0: memref<16x8xf32>, %arg1: memref<8x16xf32>, %arg2: memref<16x16xf32>, %arg3: memref<16x16xf32>) {
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c8 = arith.constant 8 : index
          %cst_0 = arith.constant 0.0 : f32

          %acc_init = vector.transfer_read %arg3[%c0, %c0], %cst_0 : memref<16x16xf32>, vector<16x16xf32>
          // CHECK: llvm.inline_asm {{.*}}msetmtype t5, t6{{.*}}vtzero mt0
          // CHECK: scf.for
          %res = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %acc_init) -> (vector<16x16xf32>) {
            %lhs = vector.transfer_read %arg0[%c0, %iv], %cst_0 : memref<16x8xf32>, vector<16x1xf32>
            %rhs = vector.transfer_read %arg1[%iv, %c0], %cst_0 : memref<8x16xf32>, vector<1x16xf32>
            // The multiply block re-programs mtype: a vset* may sit between it
            // and the vtzero block.
            // CHECK: llvm.inline_asm {{.*}}msetmtype t5, t6{{.*}}vtfmm.tvv mt0
            // CHECK-NOT: vtzero
            %next = vector.contract {
              indexing_maps = [
                affine_map<(d0, d1, d2) -> (d0, d2)>,
                affine_map<(d0, d1, d2) -> (d2, d1)>,
                affine_map<(d0, d1, d2) -> (d0, d1)>
              ],
              iterator_types = ["parallel", "parallel", "reduction"]
            } %lhs, %rhs, %acc : vector<16x1xf32>, vector<1x16xf32> into vector<16x16xf32>
            scf.yield %next : vector<16x16xf32>
          }
          // CHECK: llvm.inline_asm {{.*}}vtmv.v.t v0, t0{{.*}}vse32.v
          vector.transfer_write %res, %arg2[%c0, %c0] : vector<16x16xf32>, memref<16x16xf32>
          return
        }
      }
    }
  }

  // CHECK-LABEL: hal.executable private @matmul_i8
  hal.executable private @matmul_i8 {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_i8
        func.func @matmul_i8(%arg0: memref<16x1xi8>, %arg1: memref<1x16xi8>, %arg2: memref<16x16xi32>) {
          %c0 = arith.constant 0 : index
          %cst_0 = arith.constant 0 : i32
          %cst_i8 = arith.constant 0 : i8

          %lhs = vector.transfer_read %arg0[%c0, %c0], %cst_i8 : memref<16x1xi8>, vector<16x1xi8>
          %rhs = vector.transfer_read %arg1[%c0, %c0], %cst_i8 : memref<1x16xi8>, vector<1x16xi8>
          %acc = vector.transfer_read %arg2[%c0, %c0], %cst_0 : memref<16x16xi32>, vector<16x16xi32>

          // vtzero runs under the f32 config (tiles are 32-bit); the i8
          // config then sets vtype.altfmt (bit 8 = 256) to make operand B
          // signed.
          // CHECK: llvm.inline_asm {{.*}}li t5, 16417{{.*}}li t6, 18{{.*}}msetmtype t5, t6{{.*}}vtzero mt0{{.*}}li t5, 16419{{.*}}li t6, 256{{.*}}msetmtype t5, t6{{.*}}vtmms.tvv mt0
          %res = vector.contract {
            indexing_maps = [
              affine_map<(d0, d1, d2) -> (d0, d2)>,
              affine_map<(d0, d1, d2) -> (d2, d1)>,
              affine_map<(d0, d1, d2) -> (d0, d1)>
            ],
            iterator_types = ["parallel", "parallel", "reduction"]
          } %lhs, %rhs, %acc : vector<16x1xi8>, vector<1x16xi8> into vector<16x16xi32>

          // Writeback drains mt0 row by row: vtmv.v.t + vse32.v, 16 times.
          // CHECK: llvm.inline_asm {{.*}}msetmtype t5, t6{{.*}}li t0, 0{{.*}}vtmv.v.t v0, t0{{.*}}vse32.v v0, (t4){{.*}}add t4, t4, t5{{.*}}addi t0, t0, 1{{.*}}bnez t6
          vector.transfer_write %res, %arg2[%c0, %c0] : vector<16x16xi32>, memref<16x16xi32>
          return
        }
      }
    }
  }

  // CHECK-LABEL: hal.executable private @matmul_i8_loop
  hal.executable private @matmul_i8_loop {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_i8_loop
        func.func @matmul_i8_loop(%arg0: memref<16x8xi8>, %arg1: memref<8x16xi8>, %arg2: memref<16x16xi32>) {
          // CHECK: llvm.inline_asm {{.*}}li t5, 16417{{.*}}msetmtype t5, t6{{.*}}vtzero mt0{{.*}}li t5, 16419{{.*}}li t6, 256{{.*}}msetmtype t5, t6{{.*}}vlse8.v{{.*}}vle8.v{{.*}}vtmms.tvv mt0{{.*}}vse32.v
          %c0 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c1 = arith.constant 1 : index
          %cst_0 = arith.constant 0 : i32
          %cst_i8 = arith.constant 0 : i8

          %acc_init = vector.transfer_read %arg2[%c0, %c0], %cst_0 : memref<16x16xi32>, vector<16x16xi32>
          %res = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %acc_init) -> (vector<16x16xi32>) {
            %lhs = vector.transfer_read %arg0[%c0, %iv], %cst_i8 : memref<16x8xi8>, vector<16x1xi8>
            %rhs = vector.transfer_read %arg1[%iv, %c0], %cst_i8 : memref<8x16xi8>, vector<1x16xi8>
            %next = vector.contract {
              indexing_maps = [
                affine_map<(d0, d1, d2) -> (d0, d2)>,
                affine_map<(d0, d1, d2) -> (d2, d1)>,
                affine_map<(d0, d1, d2) -> (d0, d1)>
              ],
              iterator_types = ["parallel", "parallel", "reduction"]
            } %lhs, %rhs, %acc : vector<16x1xi8>, vector<1x16xi8> into vector<16x16xi32>
            scf.yield %next : vector<16x16xi32>
          }
          vector.transfer_write %res, %arg2[%c0, %c0] : vector<16x16xi32>, memref<16x16xi32>
          return
        }
      }
    }
  }

  // CHECK-LABEL: hal.executable private @matmul_i8_multitile
  hal.executable private @matmul_i8_multitile {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_i8_multitile
        func.func @matmul_i8_multitile(%arg0: memref<16x8xi8>, %arg1: memref<8x32xi8>, %arg2: memref<16x32xi32>) {
          // CHECK: llvm.inline_asm {{.*}}msetmtype t5, t6{{.*}}vtzero mt0{{.*}}vtzero mt4{{.*}}li t5, 16419{{.*}}msetmtype t5, t6{{.*}}vtmms.tvv mt0{{.*}}vtmms.tvv mt4{{.*}}lui t0, 0x20000{{.*}}vtmv.v.t
          %c0 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c1 = arith.constant 1 : index
          %cst_0 = arith.constant 0 : i32
          %cst_i8 = arith.constant 0 : i8

          %acc_init = vector.transfer_read %arg2[%c0, %c0], %cst_0 : memref<16x32xi32>, vector<16x32xi32>
          %res = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %acc_init) -> (vector<16x32xi32>) {
            %lhs = vector.transfer_read %arg0[%c0, %iv], %cst_i8 : memref<16x8xi8>, vector<16x1xi8>
            %rhs = vector.transfer_read %arg1[%iv, %c0], %cst_i8 : memref<8x32xi8>, vector<1x32xi8>
            %next = vector.contract {
              indexing_maps = [
                affine_map<(d0, d1, d2) -> (d0, d2)>,
                affine_map<(d0, d1, d2) -> (d2, d1)>,
                affine_map<(d0, d1, d2) -> (d0, d1)>
              ],
              iterator_types = ["parallel", "parallel", "reduction"]
            } %lhs, %rhs, %acc : vector<16x1xi8>, vector<1x32xi8> into vector<16x32xi32>
            scf.yield %next : vector<16x32xi32>
          }
          vector.transfer_write %res, %arg2[%c0, %c0] : vector<16x32xi32>, memref<16x32xi32>
          return
        }
      }
    }
  }

  // CHECK-LABEL: hal.executable private @matmul_i8_transposed
  hal.executable private @matmul_i8_transposed {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_i8_transposed
        func.func @matmul_i8_transposed(%arg0: memref<8x16xi8>, %arg1: memref<8x32xi8>, %arg2: memref<16x32xi32>) {
          // CHECK: llvm.inline_asm {{.*}}vle8.v v4{{.*}}vle8.v v8{{.*}}vle8.v v12{{.*}}vtmms.tvv mt0{{.*}}vtmms.tvv mt4
          %c0 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c1 = arith.constant 1 : index
          %cst_0 = arith.constant 0 : i32
          %cst_i8 = arith.constant 0 : i8

          %acc_init = vector.transfer_read %arg2[%c0, %c0], %cst_0 : memref<16x32xi32>, vector<16x32xi32>
          %res = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %acc_init) -> (vector<16x32xi32>) {
            %lhs = vector.transfer_read %arg0[%iv, %c0], %cst_i8 : memref<8x16xi8>, vector<1x16xi8>
            %rhs = vector.transfer_read %arg1[%iv, %c0], %cst_i8 : memref<8x32xi8>, vector<1x32xi8>
            %next = vector.contract {
              indexing_maps = [
                affine_map<(d0, d1, d2) -> (d2, d0)>,
                affine_map<(d0, d1, d2) -> (d2, d1)>,
                affine_map<(d0, d1, d2) -> (d0, d1)>
              ],
              iterator_types = ["parallel", "parallel", "reduction"]
            } %lhs, %rhs, %acc : vector<1x16xi8>, vector<1x32xi8> into vector<16x32xi32>
            scf.yield %next : vector<16x32xi32>
          }
          vector.transfer_write %res, %arg2[%c0, %c0] : vector<16x32xi32>, memref<16x32xi32>
          return
        }
      }
    }
  }

  // CHECK-LABEL: hal.executable private @matmul_2x2_multitile
  hal.executable private @matmul_2x2_multitile {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_2x2_multitile
        func.func @matmul_2x2_multitile(%arg0: memref<8x32xf32>, %arg1: memref<8x32xf32>, %arg2: memref<32x32xf32>) {
          // CHECK: llvm.inline_asm {{.*}}msetmtype t5, t6{{.*}}vtzero mt0{{.*}}vtzero mt4{{.*}}vtzero mt8{{.*}}vtzero mt12{{.*}}vtfmm.tvv mt0{{.*}}vtfmm.tvv mt4{{.*}}vtfmm.tvv mt8{{.*}}vtfmm.tvv mt12{{.*}}lui t0, 0x20000{{.*}}lui t0, 0x40000{{.*}}lui t0, 0x60000{{.*}}"{a0},{a1},{a2},{a3},{a4},{a5},~{t0},~{t1},~{t2},~{t3},~{t4},~{t5},~{t6},~{v0},~{v4},~{v8},~{v12},~{v24},~{memory}"
          %c0 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c1 = arith.constant 1 : index
          %cst_0 = arith.constant 0.0 : f32

          %acc_init = vector.transfer_read %arg2[%c0, %c0], %cst_0 : memref<32x32xf32>, vector<32x32xf32>
          %res = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %acc_init) -> (vector<32x32xf32>) {
            %lhs = vector.transfer_read %arg0[%iv, %c0], %cst_0 : memref<8x32xf32>, vector<1x32xf32>
            %rhs = vector.transfer_read %arg1[%iv, %c0], %cst_0 : memref<8x32xf32>, vector<1x32xf32>
            %next = vector.contract {
              indexing_maps = [
                affine_map<(d0, d1, d2) -> (d2, d0)>,
                affine_map<(d0, d1, d2) -> (d2, d1)>,
                affine_map<(d0, d1, d2) -> (d0, d1)>
              ],
              iterator_types = ["parallel", "parallel", "reduction"]
            } %lhs, %rhs, %acc : vector<1x32xf32>, vector<1x32xf32> into vector<32x32xf32>
            scf.yield %next : vector<32x32xf32>
          }
          vector.transfer_write %res, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
          return
        }
      }
    }
  }

  // CHECK-LABEL: hal.executable private @matmul_mixed_f32_i8
  hal.executable private @matmul_mixed_f32_i8 {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_mixed_f32_i8
        func.func @matmul_mixed_f32_i8(%arg0: memref<16x1xf32>, %arg1: memref<1x16xf32>, %arg2: memref<16x16xf32>, %arg3: memref<16x1xi8>, %arg4: memref<1x16xi8>, %arg5: memref<16x16xi32>) {
          // Two element types in one function: each block carries its own
          // mtype/vtype, so no cross-chain bookkeeping is needed.
          // CHECK: llvm.inline_asm {{.*}}li t5, 16417{{.*}}li t6, 18{{.*}}msetmtype t5, t6{{.*}}vtzero mt0{{.*}}vtfmm.tvv mt0
          // CHECK: llvm.inline_asm {{.*}}li t5, 16419{{.*}}li t6, 256{{.*}}msetmtype t5, t6{{.*}}vtmms.tvv mt0
          %c0 = arith.constant 0 : index
          %cst_0 = arith.constant 0.0 : f32
          %cst_i32 = arith.constant 0 : i32
          %cst_i8 = arith.constant 0 : i8

          %lhs0 = vector.transfer_read %arg0[%c0, %c0], %cst_0 : memref<16x1xf32>, vector<16x1xf32>
          %rhs0 = vector.transfer_read %arg1[%c0, %c0], %cst_0 : memref<1x16xf32>, vector<1x16xf32>
          %acc0 = vector.transfer_read %arg2[%c0, %c0], %cst_0 : memref<16x16xf32>, vector<16x16xf32>
          %res0 = vector.contract {
            indexing_maps = [
              affine_map<(d0, d1, d2) -> (d0, d2)>,
              affine_map<(d0, d1, d2) -> (d2, d1)>,
              affine_map<(d0, d1, d2) -> (d0, d1)>
            ],
            iterator_types = ["parallel", "parallel", "reduction"]
          } %lhs0, %rhs0, %acc0 : vector<16x1xf32>, vector<1x16xf32> into vector<16x16xf32>
          vector.transfer_write %res0, %arg2[%c0, %c0] : vector<16x16xf32>, memref<16x16xf32>

          %lhs1 = vector.transfer_read %arg3[%c0, %c0], %cst_i8 : memref<16x1xi8>, vector<16x1xi8>
          %rhs1 = vector.transfer_read %arg4[%c0, %c0], %cst_i8 : memref<1x16xi8>, vector<1x16xi8>
          %acc1 = vector.transfer_read %arg5[%c0, %c0], %cst_i32 : memref<16x16xi32>, vector<16x16xi32>
          %res1 = vector.contract {
            indexing_maps = [
              affine_map<(d0, d1, d2) -> (d0, d2)>,
              affine_map<(d0, d1, d2) -> (d2, d1)>,
              affine_map<(d0, d1, d2) -> (d0, d1)>
            ],
            iterator_types = ["parallel", "parallel", "reduction"]
          } %lhs1, %rhs1, %acc1 : vector<16x1xi8>, vector<1x16xi8> into vector<16x16xi32>
          vector.transfer_write %res1, %arg5[%c0, %c0] : vector<16x16xi32>, memref<16x16xi32>
          return
        }
      }
    }
  }

  // A CoralNPU target without +zvtbase must be left alone.
  // CHECK-LABEL: hal.executable private @matmul_no_zvtbase
  hal.executable private @matmul_no_zvtbase {
    hal.executable.variant public @coralnpu_elf target(#target_no_zvt) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_no_zvtbase
        func.func @matmul_no_zvtbase(%arg0: memref<16x1xf32>, %arg1: memref<1x16xf32>, %arg2: memref<16x16xf32>) {
          // CHECK-NOT: llvm.inline_asm
          // CHECK-NOT: msetmtype
          // CHECK: vector.contract
          %c0 = arith.constant 0 : index
          %cst_0 = arith.constant 0.0 : f32
          %lhs = vector.transfer_read %arg0[%c0, %c0], %cst_0 : memref<16x1xf32>, vector<16x1xf32>
          %rhs = vector.transfer_read %arg1[%c0, %c0], %cst_0 : memref<1x16xf32>, vector<1x16xf32>
          %acc = vector.transfer_read %arg2[%c0, %c0], %cst_0 : memref<16x16xf32>, vector<16x16xf32>
          %res = vector.contract {
            indexing_maps = [
              affine_map<(d0, d1, d2) -> (d0, d2)>,
              affine_map<(d0, d1, d2) -> (d2, d1)>,
              affine_map<(d0, d1, d2) -> (d0, d1)>
            ],
            iterator_types = ["parallel", "parallel", "reduction"]
          } %lhs, %rhs, %acc : vector<16x1xf32>, vector<1x16xf32> into vector<16x16xf32>
          vector.transfer_write %res, %arg2[%c0, %c0] : vector<16x16xf32>, memref<16x16xf32>
          return
        }
      }
    }
  }

  // A non-CoralNPU backend must likewise be untouched.
  // CHECK-LABEL: hal.executable private @matmul_other_backend
  hal.executable private @matmul_other_backend {
    hal.executable.variant public @embedded_elf target(#target_llvmcpu) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_other_backend
        func.func @matmul_other_backend(%arg0: memref<16x1xf32>, %arg1: memref<1x16xf32>, %arg2: memref<16x16xf32>) {
          // CHECK-NOT: llvm.inline_asm
          // CHECK-NOT: msetmtype
          // CHECK: vector.contract
          %c0 = arith.constant 0 : index
          %cst_0 = arith.constant 0.0 : f32
          %lhs = vector.transfer_read %arg0[%c0, %c0], %cst_0 : memref<16x1xf32>, vector<16x1xf32>
          %rhs = vector.transfer_read %arg1[%c0, %c0], %cst_0 : memref<1x16xf32>, vector<1x16xf32>
          %acc = vector.transfer_read %arg2[%c0, %c0], %cst_0 : memref<16x16xf32>, vector<16x16xf32>
          %res = vector.contract {
            indexing_maps = [
              affine_map<(d0, d1, d2) -> (d0, d2)>,
              affine_map<(d0, d1, d2) -> (d2, d1)>,
              affine_map<(d0, d1, d2) -> (d0, d1)>
            ],
            iterator_types = ["parallel", "parallel", "reduction"]
          } %lhs, %rhs, %acc : vector<16x1xf32>, vector<1x16xf32> into vector<16x16xf32>
          vector.transfer_write %res, %arg2[%c0, %c0] : vector<16x16xf32>, memref<16x16xf32>
          return
        }
      }
    }
  }

  // Right target and feature, but 8x8 is not a supported tile shape, so this
  // must fall through to the generic lowering.
  // CHECK-LABEL: hal.executable private @matmul_unsupported_shape
  hal.executable private @matmul_unsupported_shape {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_unsupported_shape
        func.func @matmul_unsupported_shape(%arg0: memref<8x1xf32>, %arg1: memref<1x8xf32>, %arg2: memref<8x8xf32>) {
          // CHECK-NOT: llvm.inline_asm
          // CHECK-NOT: msetmtype
          %c0 = arith.constant 0 : index
          %cst_0 = arith.constant 0.0 : f32
          %lhs = vector.transfer_read %arg0[%c0, %c0], %cst_0 : memref<8x1xf32>, vector<8x1xf32>
          %rhs = vector.transfer_read %arg1[%c0, %c0], %cst_0 : memref<1x8xf32>, vector<1x8xf32>
          %acc = vector.transfer_read %arg2[%c0, %c0], %cst_0 : memref<8x8xf32>, vector<8x8xf32>
          %res = vector.contract {
            indexing_maps = [
              affine_map<(d0, d1, d2) -> (d0, d2)>,
              affine_map<(d0, d1, d2) -> (d2, d1)>,
              affine_map<(d0, d1, d2) -> (d0, d1)>
            ],
            iterator_types = ["parallel", "parallel", "reduction"]
          } %lhs, %rhs, %acc : vector<8x1xf32>, vector<1x8xf32> into vector<8x8xf32>
          vector.transfer_write %res, %arg2[%c0, %c0] : vector<8x8xf32>, memref<8x8xf32>
          return
        }
      }
    }
  }
}
