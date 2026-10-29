// RUN: iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(coralnpu-matrix-codegen)))))" %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(coralnpu-matrix-codegen)))))" %s | FileCheck %s --check-prefix=NOVSET
// RUN: iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-llvmcpu-virtual-vector-lowering)))))" %s | FileCheck %s --check-prefix=GATE

// Up to @vecmat_bf16_jammed (keep it last): any vset* clears mtype and
// vtype.altfmt (RvvFrontEnd.sv), so Zvt blocks must emit none.
// NOVSET-NOT: vset

#target = #hal.executable.target<"coralnpu", "coralnpu-elf", {
  cpu_features = "+zvtbase"
}>

// Same backend, but the Zvt feature is absent.
#target_no_zvt = #hal.executable.target<"coralnpu", "coralnpu-elf", {
  cpu_features = "+v"
}>

// Zvt + Zvfbfmin: truncf to bf16 fuses into the drain.
#target_bf16 = #hal.executable.target<"coralnpu", "coralnpu-elf", {
  cpu_features = "+zvtbase,+zvfbfmin"
}>

// Non-CoralNPU backends must be left untouched even with +zvtbase.
#target_llvmcpu = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  cpu_features = "+zvtbase"
}>

// Zvt + Zvfbfwma: jammed BF16 vecmat loops lower to vfwmaccbf16.vf.
#target_bf16_wma = #hal.executable.target<"coralnpu", "coralnpu-elf", {
  cpu_features = "+zvtbase,+zvfbfwma"
}>

// CHECK-LABEL: hal.executable private @matmul {
module {
  hal.executable private @matmul {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul(
        func.func @matmul(%arg0: memref<16x1xf32>, %arg1: memref<1x16xf32>, %arg2: memref<16x16xf32>) {
          %c0 = arith.constant 0 : index
          %cst_0 = arith.constant 0.0 : f32

          %lhs = vector.transfer_read %arg0[%c0, %c0], %cst_0 : memref<16x1xf32>, vector<16x1xf32>
          %rhs = vector.transfer_read %arg1[%c0, %c0], %cst_0 : memref<1x16xf32>, vector<1x16xf32>
          %acc = arith.constant dense<0.0> : vector<16x16xf32>

          // CHECK: llvm.inline_asm {{.*}}li t5, 16417{{.*}}li t6, 18{{.*}}msetmtype t5, t6{{.*}}msettn zero, t6{{.*}}vtzero mt0{{.*}}vtfmm.tvv mt0{{.*}}"{v0},{v4},~{t5},~{t6}"
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
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
          return
        }
      }
    }
  }

  // CHECK-LABEL: hal.executable private @matmul_loop {
  hal.executable private @matmul_loop {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_loop(
        func.func @matmul_loop(%arg0: memref<16x8xf32>, %arg1: memref<8x16xf32>, %arg2: memref<16x16xf32>) {
          // The init is read from %arg2, so the tiles are seeded from it.
          // CHECK: llvm.inline_asm {{.*}}li t5, 16417{{.*}}msetmtype t5, t6{{.*}}vle32.v v0, (t4){{.*}}vtmv.t.v t0, v0{{.*}}vlse32.v{{.*}}vtfmm.tvv mt0{{.*}}vse32.v
          %c0 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c1 = arith.constant 1 : index
          %cst_0 = arith.constant 0.0 : f32

          %acc_init = vector.transfer_read %arg2[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf32>, vector<16x16xf32>
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
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
          return
        }
      }
    }
  }

  // Writeback via per-row vector.extract + vector.store. Every row is stored,
  // as the tile drain that replaces them writes all rows.
  // CHECK-LABEL: hal.executable private @matmul_loop_extract_store {
  hal.executable private @matmul_loop_extract_store {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_loop_extract_store(
        func.func @matmul_loop_extract_store(%arg0: memref<16x8xf32>, %arg1: memref<8x16xf32>, %arg2: memref<16x16xf32>) {
          // CHECK: llvm.inline_asm {{.*}}vtzero mt0{{.*}}vtfmm.tvv mt0{{.*}}vtmv.v.t v0, t0{{.*}}vse32.v
          // CHECK-NOT: vector.extract
          // CHECK-NOT: vector.store
          // CHECK-NOT: scf.for
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c2 = arith.constant 2 : index
          %c3 = arith.constant 3 : index
          %c4 = arith.constant 4 : index
          %c5 = arith.constant 5 : index
          %c6 = arith.constant 6 : index
          %c7 = arith.constant 7 : index
          %c8 = arith.constant 8 : index
          %c9 = arith.constant 9 : index
          %c10 = arith.constant 10 : index
          %c11 = arith.constant 11 : index
          %c12 = arith.constant 12 : index
          %c13 = arith.constant 13 : index
          %c14 = arith.constant 14 : index
          %c15 = arith.constant 15 : index
          %cst_0 = arith.constant 0.0 : f32

          %acc_init = arith.constant dense<0.0> : vector<16x16xf32>
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
          %row2 = vector.extract %res[2] : vector<16xf32> from vector<16x16xf32>
          vector.store %row2, %arg2[%c2, %c0] : memref<16x16xf32>, vector<16xf32>
          %row3 = vector.extract %res[3] : vector<16xf32> from vector<16x16xf32>
          vector.store %row3, %arg2[%c3, %c0] : memref<16x16xf32>, vector<16xf32>
          %row4 = vector.extract %res[4] : vector<16xf32> from vector<16x16xf32>
          vector.store %row4, %arg2[%c4, %c0] : memref<16x16xf32>, vector<16xf32>
          %row5 = vector.extract %res[5] : vector<16xf32> from vector<16x16xf32>
          vector.store %row5, %arg2[%c5, %c0] : memref<16x16xf32>, vector<16xf32>
          %row6 = vector.extract %res[6] : vector<16xf32> from vector<16x16xf32>
          vector.store %row6, %arg2[%c6, %c0] : memref<16x16xf32>, vector<16xf32>
          %row7 = vector.extract %res[7] : vector<16xf32> from vector<16x16xf32>
          vector.store %row7, %arg2[%c7, %c0] : memref<16x16xf32>, vector<16xf32>
          %row8 = vector.extract %res[8] : vector<16xf32> from vector<16x16xf32>
          vector.store %row8, %arg2[%c8, %c0] : memref<16x16xf32>, vector<16xf32>
          %row9 = vector.extract %res[9] : vector<16xf32> from vector<16x16xf32>
          vector.store %row9, %arg2[%c9, %c0] : memref<16x16xf32>, vector<16xf32>
          %row10 = vector.extract %res[10] : vector<16xf32> from vector<16x16xf32>
          vector.store %row10, %arg2[%c10, %c0] : memref<16x16xf32>, vector<16xf32>
          %row11 = vector.extract %res[11] : vector<16xf32> from vector<16x16xf32>
          vector.store %row11, %arg2[%c11, %c0] : memref<16x16xf32>, vector<16xf32>
          %row12 = vector.extract %res[12] : vector<16xf32> from vector<16x16xf32>
          vector.store %row12, %arg2[%c12, %c0] : memref<16x16xf32>, vector<16xf32>
          %row13 = vector.extract %res[13] : vector<16xf32> from vector<16x16xf32>
          vector.store %row13, %arg2[%c13, %c0] : memref<16x16xf32>, vector<16xf32>
          %row14 = vector.extract %res[14] : vector<16xf32> from vector<16x16xf32>
          vector.store %row14, %arg2[%c14, %c0] : memref<16x16xf32>, vector<16xf32>
          %row15 = vector.extract %res[15] : vector<16xf32> from vector<16x16xf32>
          vector.store %row15, %arg2[%c15, %c0] : memref<16x16xf32>, vector<16xf32>
          return
        }
      }
    }
  }

  // Dynamic trip count: the loop stays and the tiles are zeroed once before it.
  // CHECK-LABEL: hal.executable private @matmul_loop_unfused {
  hal.executable private @matmul_loop_unfused {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_loop_unfused(
        func.func @matmul_loop_unfused(%arg0: memref<16x8xf32>, %arg1: memref<8x16xf32>, %arg2: memref<16x16xf32>, %n: index) {
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %cst_0 = arith.constant 0.0 : f32

          %acc_init = arith.constant dense<0.0> : vector<16x16xf32>
          // CHECK: llvm.inline_asm {{.*}}msetmtype t5, t6{{.*}}vtzero mt0
          // CHECK: scf.for
          %res = scf.for %iv = %c0 to %n step %c1 iter_args(%acc = %acc_init) -> (vector<16x16xf32>) {
            %lhs = vector.transfer_read %arg0[%c0, %iv], %cst_0 : memref<16x8xf32>, vector<16x1xf32>
            %rhs = vector.transfer_read %arg1[%iv, %c0], %cst_0 : memref<8x16xf32>, vector<1x16xf32>
            // The multiply block re-programs mtype: a vset* may sit between it
            // and the vtzero block.
            // CHECK: llvm.inline_asm {{.*}}msetmtype t5, t6
            // CHECK-NOT: vtzero
            // CHECK-SAME: vtfmm.tvv mt0
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
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
          return
        }
      }
    }
  }

  // Two N tiles with a dynamic trip count: the operands take slots 0-2.
  // CHECK-LABEL: hal.executable private @matmul_loop_unfused_2_tiles {
  hal.executable private @matmul_loop_unfused_2_tiles {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_loop_unfused_2_tiles(
        func.func @matmul_loop_unfused_2_tiles(%arg0: memref<16x8xf32>, %arg1: memref<8x32xf32>, %arg2: memref<16x32xf32>, %n: index) {
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %cst_0 = arith.constant 0.0 : f32
          %acc_init = arith.constant dense<0.0> : vector<16x32xf32>
          // CHECK: scf.for
          // CHECK: llvm.inline_asm {{.*}}vtfmm.tvv mt0, $0, $1{{.*}}vtfmm.tvv mt4, $0, $2{{.*}}"{v0},{v4},{v8},~{t5},~{t6}"
          %res = scf.for %iv = %c0 to %n step %c1 iter_args(%acc = %acc_init) -> (vector<16x32xf32>) {
            %lhs = vector.transfer_read %arg0[%c0, %iv], %cst_0 : memref<16x8xf32>, vector<16x1xf32>
            %rhs = vector.transfer_read %arg1[%iv, %c0], %cst_0 : memref<8x32xf32>, vector<1x32xf32>
            %next = vector.contract {
              indexing_maps = [
                affine_map<(d0, d1, d2) -> (d0, d2)>,
                affine_map<(d0, d1, d2) -> (d2, d1)>,
                affine_map<(d0, d1, d2) -> (d0, d1)>
              ],
              iterator_types = ["parallel", "parallel", "reduction"]
            } %lhs, %rhs, %acc : vector<16x1xf32>, vector<1x32xf32> into vector<16x32xf32>
            scf.yield %next : vector<16x32xf32>
          }
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x32xf32>, memref<16x32xf32>
          return
        }
      }
    }
  }

  // CHECK-LABEL: hal.executable private @matmul_i8 {
  hal.executable private @matmul_i8 {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_i8(
        func.func @matmul_i8(%arg0: memref<16x1xi8>, %arg1: memref<1x16xi8>, %arg2: memref<16x16xi32>) {
          %c0 = arith.constant 0 : index
          %cst_i8 = arith.constant 0 : i8

          %lhs = vector.transfer_read %arg0[%c0, %c0], %cst_i8 : memref<16x1xi8>, vector<16x1xi8>
          %rhs = vector.transfer_read %arg1[%c0, %c0], %cst_i8 : memref<1x16xi8>, vector<1x16xi8>
          %acc = arith.constant dense<0> : vector<16x16xi32>

          // vtzero runs under the f32 config; the i8 config sets altfmt (256)
          // to make operand B signed.
          // CHECK: llvm.inline_asm {{.*}}li t5, 16417{{.*}}li t6, 18{{.*}}msetmtype t5, t6{{.*}}vtzero mt0{{.*}}li t5, 16419{{.*}}li t6, 256{{.*}}msetmtype t5, t6{{.*}}vtmms.tvv mt0{{.*}}"{v0},{v1},~{t5},~{t6}"{{.*}}(vector<[8]xi8>, vector<[8]xi8>) -> ()
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
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xi32>, memref<16x16xi32>
          return
        }
      }
    }
  }

  // CHECK-LABEL: hal.executable private @matmul_i8_loop {
  hal.executable private @matmul_i8_loop {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_i8_loop(
        func.func @matmul_i8_loop(%arg0: memref<16x8xi8>, %arg1: memref<8x16xi8>, %arg2: memref<16x16xi32>) {
          // CHECK: llvm.inline_asm {{.*}}li t5, 16417{{.*}}msetmtype t5, t6{{.*}}vtzero mt0{{.*}}li t5, 16515{{.*}}li t6, 256{{.*}}msetmtype t5, t6{{.*}}vlse8.v{{.*}}vle8.v{{.*}}vtmms.tvv mt0{{.*}}vse32.v
          %c0 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c1 = arith.constant 1 : index
          %cst_i8 = arith.constant 0 : i8

          %acc_init = arith.constant dense<0> : vector<16x16xi32>
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
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xi32>, memref<16x16xi32>
          return
        }
      }
    }
  }

  // CHECK-LABEL: hal.executable private @matmul_bf16 {
  hal.executable private @matmul_bf16 {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_bf16(
        func.func @matmul_bf16(%arg0: memref<16x1xbf16>, %arg1: memref<1x16xbf16>, %arg2: memref<16x16xf32>) {
          %c0 = arith.constant 0 : index
          %cst_bf16 = arith.constant 0.0 : bf16

          %lhs16 = vector.transfer_read %arg0[%c0, %c0], %cst_bf16 : memref<16x1xbf16>, vector<16x1xbf16>
          %rhs16 = vector.transfer_read %arg1[%c0, %c0], %cst_bf16 : memref<1x16xbf16>, vector<1x16xbf16>
          %lhs = arith.extf %lhs16 : vector<16x1xbf16> to vector<16x1xf32>
          %rhs = arith.extf %rhs16 : vector<1x16xbf16> to vector<1x16xf32>
          %acc = arith.constant dense<0.0> : vector<16x16xf32>

          // bf16 config: mtwiden=2 (16418), SEW16/LMUL2/altfmt (265). The asm
          // takes the unextended bf16 values in LMUL2 registers.
          // CHECK: llvm.inline_asm {{.*}}li t5, 16417{{.*}}li t6, 18{{.*}}msetmtype t5, t6{{.*}}vtzero mt0{{.*}}li t5, 16418{{.*}}li t6, 265{{.*}}msetmtype t5, t6{{.*}}vtfmm.alt.tvv mt0{{.*}}"{v0},{v2},~{t5},~{t6}"{{.*}}(vector<[8]xbf16>, vector<[8]xbf16>) -> ()
          %res = vector.contract {
            indexing_maps = [
              affine_map<(d0, d1, d2) -> (d0, d2)>,
              affine_map<(d0, d1, d2) -> (d2, d1)>,
              affine_map<(d0, d1, d2) -> (d0, d1)>
            ],
            iterator_types = ["parallel", "parallel", "reduction"]
          } %lhs, %rhs, %acc : vector<16x1xf32>, vector<1x16xf32> into vector<16x16xf32>

          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
          return
        }
      }
    }
  }

  // CHECK-LABEL: hal.executable private @matmul_bf16_loop {
  hal.executable private @matmul_bf16_loop {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_bf16_loop(
        func.func @matmul_bf16_loop(%arg0: memref<16x8xbf16>, %arg1: memref<8x16xbf16>, %arg2: memref<16x16xf32>) {
          // CHECK: llvm.inline_asm {{.*}}li t5, 16417{{.*}}msetmtype t5, t6{{.*}}vtzero mt0{{.*}}li t5, 16450{{.*}}li t6, 265{{.*}}msetmtype t5, t6{{.*}}vlse16.v{{.*}}vle16.v{{.*}}vtfmm.alt.tvv mt0{{.*}}addi t0, t0, 2{{.*}}vse32.v
          %c0 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c1 = arith.constant 1 : index
          %cst_bf16 = arith.constant 0.0 : bf16

          %acc_init = arith.constant dense<0.0> : vector<16x16xf32>
          %res = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %acc_init) -> (vector<16x16xf32>) {
            %lhs = vector.transfer_read %arg0[%c0, %iv], %cst_bf16 : memref<16x8xbf16>, vector<16x1xbf16>
            %rhs = vector.transfer_read %arg1[%iv, %c0], %cst_bf16 : memref<8x16xbf16>, vector<1x16xbf16>
            %next = vector.contract {
              indexing_maps = [
                affine_map<(d0, d1, d2) -> (d0, d2)>,
                affine_map<(d0, d1, d2) -> (d2, d1)>,
                affine_map<(d0, d1, d2) -> (d0, d1)>
              ],
              iterator_types = ["parallel", "parallel", "reduction"]
            } %lhs, %rhs, %acc : vector<16x1xbf16>, vector<1x16xbf16> into vector<16x16xf32>
            scf.yield %next : vector<16x16xf32>
          }
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
          return
        }
      }
    }
  }

  // Transposed RHS (N x K) with fused truncf to bf16: B loads via vlse16.v and
  // the drain narrows via vfncvtbf16.f.f.w + vse16.v (t2/t3 configs).
  // CHECK-LABEL: hal.executable private @matmul_bf16_trunc_transposed_rhs {
  hal.executable private @matmul_bf16_trunc_transposed_rhs {
    hal.executable.variant public @coralnpu_elf target(#target_bf16) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_bf16_trunc_transposed_rhs(
        func.func @matmul_bf16_trunc_transposed_rhs(%arg0: memref<16x8xbf16>, %arg1: memref<32x8xbf16>, %arg2: memref<16x32xbf16>) {
          // CHECK: llvm.inline_asm {{.*}}vtzero mt0{{.*}}vtzero mt4{{.*}}li t5, 16450{{.*}}li t6, 265{{.*}}msetmtype t5, t6{{.*}}slli t4, t3, 4{{.*}}add t4, t2, t4{{.*}}vlse16.v v2, (t0), t1{{.*}}vlse16.v v8, (t2), t3{{.*}}vlse16.v v10, (t4), t3{{.*}}vlse16.v v6, (t0), t1{{.*}}vlse16.v v12, (t2), t3{{.*}}vlse16.v v14, (t4), t3{{.*}}vtfmm.alt.tvv mt0, v2, v8{{.*}}vtfmm.alt.tvv mt4, v2, v10{{.*}}addi t2, t2, 2{{.*}}addi t4, t4, 2{{.*}}li t2, 16417{{.*}}msetmtype t2, t3{{.*}}vtmv.v.t v8, t0{{.*}}li t2, 16418{{.*}}msetmtype t2, t3{{.*}}vfncvtbf16.f.f.w v12, v8{{.*}}vse16.v v12, (t4){{.*}}addi t4, t4, 32{{.*}}lui t0, 0x20000
          // CHECK-NOT: arith.truncf
          %c0 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c1 = arith.constant 1 : index
          %cst_bf16 = arith.constant 0.0 : bf16
          %acc_init = arith.constant dense<0.0> : vector<16x32xf32>
          %res = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %acc_init) -> (vector<16x32xf32>) {
            %lhs = vector.transfer_read %arg0[%c0, %iv], %cst_bf16 : memref<16x8xbf16>, vector<16x1xbf16>
            %rhs = vector.transfer_read %arg1[%c0, %iv], %cst_bf16 : memref<32x8xbf16>, vector<32x1xbf16>
            %next = vector.contract {
              indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
              iterator_types = ["parallel", "parallel", "reduction"]
            } %lhs, %rhs, %acc : vector<16x1xbf16>, vector<32x1xbf16> into vector<16x32xf32>
            scf.yield %next : vector<16x32xf32>
          }
          %trunc = arith.truncf %res : vector<16x32xf32> to vector<16x32xbf16>
          vector.transfer_write %trunc, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x32xbf16>, memref<16x32xbf16>
          return
        }
      }
    }
  }

  // Without Zvfbfmin the truncf is not fused: the f32 result drains to a
  // scratch tile and the truncf runs in the consumer row loop.
  // CHECK-LABEL: hal.executable private @matmul_bf16_trunc_no_zvfbfmin {
  hal.executable private @matmul_bf16_trunc_no_zvfbfmin {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_bf16_trunc_no_zvfbfmin(
        // CHECK: %[[SCRATCH:.*]] = memref.alloc() alignment = 64 : memref<16x32xf32, #iree_codegen.workgroup_local>
        // CHECK: llvm.inline_asm {{.*}}vtfmm.alt.tvv mt0{{.*}}vtmv.v.t v0, t0{{.*}}vse32.v v0, (t4)
        // CHECK-NOT: vfncvtbf16
        // CHECK: scf.for
        // CHECK:   %[[ROW:.*]] = vector.load %[[SCRATCH]]
        // CHECK:   %[[T:.*]] = arith.truncf %[[ROW]] : vector<32xf32> to vector<32xbf16>
        // CHECK:   vector.store %[[T]], %arg2
        func.func @matmul_bf16_trunc_no_zvfbfmin(%arg0: memref<16x8xbf16>, %arg1: memref<32x8xbf16>, %arg2: memref<16x32xbf16>) {
          %c0 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c1 = arith.constant 1 : index
          %cst_bf16 = arith.constant 0.0 : bf16
          %acc_init = arith.constant dense<0.0> : vector<16x32xf32>
          %res = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %acc_init) -> (vector<16x32xf32>) {
            %lhs = vector.transfer_read %arg0[%c0, %iv], %cst_bf16 : memref<16x8xbf16>, vector<16x1xbf16>
            %rhs = vector.transfer_read %arg1[%c0, %iv], %cst_bf16 : memref<32x8xbf16>, vector<32x1xbf16>
            %next = vector.contract {
              indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
              iterator_types = ["parallel", "parallel", "reduction"]
            } %lhs, %rhs, %acc : vector<16x1xbf16>, vector<32x1xbf16> into vector<16x32xf32>
            scf.yield %next : vector<16x32xf32>
          }
          %trunc = arith.truncf %res : vector<16x32xf32> to vector<16x32xbf16>
          vector.transfer_write %trunc, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x32xbf16>, memref<16x32xbf16>
          return
        }
      }
    }
  }

  // CHECK-LABEL: hal.executable private @matmul_i8_multitile {
  hal.executable private @matmul_i8_multitile {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_i8_multitile(
        func.func @matmul_i8_multitile(%arg0: memref<16x8xi8>, %arg1: memref<8x32xi8>, %arg2: memref<16x32xi32>) {
          // CHECK: llvm.inline_asm {{.*}}msetmtype t5, t6{{.*}}vtzero mt0{{.*}}vtzero mt4{{.*}}li t5, 16515{{.*}}msetmtype t5, t6{{.*}}vtmms.tvv mt0{{.*}}vtmms.tvv mt4{{.*}}lui t0, 0x20000{{.*}}vtmv.v.t
          %c0 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c1 = arith.constant 1 : index
          %cst_i8 = arith.constant 0 : i8

          %acc_init = arith.constant dense<0> : vector<16x32xi32>
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
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x32xi32>, memref<16x32xi32>
          return
        }
      }
    }
  }

  // CHECK-LABEL: hal.executable private @matmul_i8_transposed {
  hal.executable private @matmul_i8_transposed {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_i8_transposed(
        func.func @matmul_i8_transposed(%arg0: memref<8x16xi8>, %arg1: memref<8x32xi8>, %arg2: memref<16x32xi32>) {
          // CHECK: llvm.inline_asm {{.*}}vle8.v v1{{.*}}vl2re8.v v8{{.*}}vtmms.tvv mt0, v1, v8{{.*}}vtmms.tvv mt4, v1, v9
          %c0 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c1 = arith.constant 1 : index
          %cst_0 = arith.constant 0 : i32
          %cst_i8 = arith.constant 0 : i8

          %acc_init = vector.transfer_read %arg2[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x32xi32>, vector<16x32xi32>
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
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x32xi32>, memref<16x32xi32>
          return
        }
      }
    }
  }

  // CHECK-LABEL: hal.executable private @matmul_2x2_multitile {
  hal.executable private @matmul_2x2_multitile {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_2x2_multitile(
        func.func @matmul_2x2_multitile(%arg0: memref<8x32xf32>, %arg1: memref<8x32xf32>, %arg2: memref<32x32xf32>) {
          // CHECK: llvm.inline_asm {{.*}}msetmtype t5, t6{{.*}}vtzero mt0{{.*}}vtzero mt4{{.*}}vtzero mt8{{.*}}vtzero mt12{{.*}}vtfmm.tvv mt0{{.*}}vtfmm.tvv mt4{{.*}}vtfmm.tvv mt8{{.*}}vtfmm.tvv mt12{{.*}}lui t0, 0x20000{{.*}}lui t0, 0x40000{{.*}}lui t0, 0x60000{{.*}}"{a0},{a1},{a2},{a3},{a4},{a5},~{t0},~{t1},~{t2},~{t3},~{t4},~{t5},~{t6},~{v0},~{v1},~{v2},~{v3},~{v4},{{.*}},~{v15},{{.*}},~{v24},~{v25},~{v26},~{v27},{{.*}},~{memory}"
          %c0 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c1 = arith.constant 1 : index
          %cst_0 = arith.constant 0.0 : f32

          %acc_init = arith.constant dense<0.0> : vector<32x32xf32>
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
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<32x32xf32>, memref<32x32xf32>
          return
        }
      }
    }
  }

  // 1x4 strip (16x64): one LHS load feeds all four accumulators mt0..mt12.
  // CHECK-LABEL: hal.executable private @matmul_1x4_multitile {
  hal.executable private @matmul_1x4_multitile {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_1x4_multitile(
        func.func @matmul_1x4_multitile(%arg0: memref<16x8xbf16>, %arg1: memref<8x64xbf16>, %arg2: memref<16x64xf32>) {
          // CHECK: llvm.inline_asm {{.*}}vtzero mt0{{.*}}vtzero mt4{{.*}}vtzero mt8{{.*}}vtzero mt12{{.*}}vlse16.v v2, (t0), t1{{.*}}vl4re16.v v8, (t2){{.*}}addi t4, t2, 64{{.*}}vl4re16.v v16, (t4){{.*}}vtfmm.alt.tvv mt0, v2, v8{{.*}}vtfmm.alt.tvv mt4, v2, v10{{.*}}vtfmm.alt.tvv mt8, v2, v16{{.*}}vtfmm.alt.tvv mt12, v2, v18{{.*}}addi t0, t0, 2{{.*}}add t2, t2, t3{{.*}}addi t4, t4, 64{{.*}}lui t0, 0x20000{{.*}}addi t4, t4, 128{{.*}}lui t0, 0x40000{{.*}}addi t4, t4, 192{{.*}}lui t0, 0x60000
          %c0 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c1 = arith.constant 1 : index
          %cst_bf16 = arith.constant 0.0 : bf16
          %acc_init = arith.constant dense<0.0> : vector<16x64xf32>
          %res = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %acc_init) -> (vector<16x64xf32>) {
            %lhs = vector.transfer_read %arg0[%c0, %iv], %cst_bf16 : memref<16x8xbf16>, vector<16x1xbf16>
            %rhs = vector.transfer_read %arg1[%iv, %c0], %cst_bf16 : memref<8x64xbf16>, vector<1x64xbf16>
            %next = vector.contract {
              indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
              iterator_types = ["parallel", "parallel", "reduction"]
            } %lhs, %rhs, %acc : vector<16x1xbf16>, vector<1x64xbf16> into vector<16x64xf32>
            scf.yield %next : vector<16x64xf32>
          }
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x64xf32>, memref<16x64xf32>
          return
        }
      }
    }
  }

  // 4x1 strip (64x16) with K=16: four LHS tiles feed mt0..mt12 against one
  // RHS load, unrolled 4x with TK=2 (li t6, 2).
  // CHECK-LABEL: hal.executable private @matmul_4x1_multitile {
  hal.executable private @matmul_4x1_multitile {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_4x1_multitile(
        func.func @matmul_4x1_multitile(%arg0: memref<64x16xbf16>, %arg1: memref<16x16xbf16>, %arg2: memref<64x16xf32>) {
          // CHECK: llvm.inline_asm {{.*}}vtzero mt0{{.*}}vtzero mt4{{.*}}vtzero mt8{{.*}}vtzero mt12{{.*}}li t5, 16450{{.*}}slli t4, t1, 4{{.*}}li t6, 2{{.*}}vlse16.v v2, (t0), t1{{.*}}add t5, t0, t4{{.*}}vlse16.v v24, (t5), t1{{.*}}add t5, t5, t4{{.*}}vlse16.v v16, (t5), t1{{.*}}add t5, t5, t4{{.*}}vlse16.v v18, (t5), t1{{.*}}vle16.v v8, (t2){{.*}}vtfmm.alt.tvv mt0, v2, v8{{.*}}vtfmm.alt.tvv mt4, v24, v8{{.*}}vtfmm.alt.tvv mt8, v16, v8{{.*}}vtfmm.alt.tvv mt12, v18, v8{{.*}}mv t4, $4{{.*}}li t0, 0{{.*}}lui t0, 0x20000{{.*}}lui t0, 0x40000{{.*}}lui t0, 0x60000
          %c0 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1 = arith.constant 1 : index
          %cst_bf16 = arith.constant 0.0 : bf16
          %acc_init = arith.constant dense<0.0> : vector<64x16xf32>
          %res = scf.for %iv = %c0 to %c16 step %c1 iter_args(%acc = %acc_init) -> (vector<64x16xf32>) {
            %lhs = vector.transfer_read %arg0[%c0, %iv], %cst_bf16 : memref<64x16xbf16>, vector<64x1xbf16>
            %rhs = vector.transfer_read %arg1[%iv, %c0], %cst_bf16 : memref<16x16xbf16>, vector<1x16xbf16>
            %next = vector.contract {
              indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
              iterator_types = ["parallel", "parallel", "reduction"]
            } %lhs, %rhs, %acc : vector<64x1xbf16>, vector<1x16xbf16> into vector<64x16xf32>
            scf.yield %next : vector<64x16xf32>
          }
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<64x16xf32>, memref<64x16xf32>
          return
        }
      }
    }
  }

  // CHECK-LABEL: hal.executable private @matmul_mixed_f32_i8 {
  hal.executable private @matmul_mixed_f32_i8 {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_mixed_f32_i8(
        func.func @matmul_mixed_f32_i8(%arg0: memref<16x1xf32>, %arg1: memref<1x16xf32>, %arg2: memref<16x16xf32>, %arg3: memref<16x1xi8>, %arg4: memref<1x16xi8>, %arg5: memref<16x16xi32>) {
          // Two element types in one function: each block carries its own
          // mtype/vtype, so no cross-chain bookkeeping is needed.
          // CHECK: llvm.inline_asm {{.*}}li t5, 16417{{.*}}li t6, 18{{.*}}msetmtype t5, t6{{.*}}vtzero mt0{{.*}}vtfmm.tvv mt0
          // The i8 acc is read from %arg5, so a separate block seeds its tiles.
          // CHECK: llvm.inline_asm {{.*}}vle32.v v0, (t4){{.*}}vtmv.t.v t0, v0
          // CHECK: llvm.inline_asm {{.*}}li t5, 16419{{.*}}li t6, 256{{.*}}msetmtype t5, t6{{.*}}vtmms.tvv mt0
          %c0 = arith.constant 0 : index
          %cst_0 = arith.constant 0.0 : f32
          %cst_i32 = arith.constant 0 : i32
          %cst_i8 = arith.constant 0 : i8

          %lhs0 = vector.transfer_read %arg0[%c0, %c0], %cst_0 : memref<16x1xf32>, vector<16x1xf32>
          %rhs0 = vector.transfer_read %arg1[%c0, %c0], %cst_0 : memref<1x16xf32>, vector<1x16xf32>
          %acc0 = arith.constant dense<0.0> : vector<16x16xf32>
          %res0 = vector.contract {
            indexing_maps = [
              affine_map<(d0, d1, d2) -> (d0, d2)>,
              affine_map<(d0, d1, d2) -> (d2, d1)>,
              affine_map<(d0, d1, d2) -> (d0, d1)>
            ],
            iterator_types = ["parallel", "parallel", "reduction"]
          } %lhs0, %rhs0, %acc0 : vector<16x1xf32>, vector<1x16xf32> into vector<16x16xf32>
          vector.transfer_write %res0, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>

          %lhs1 = vector.transfer_read %arg3[%c0, %c0], %cst_i8 : memref<16x1xi8>, vector<16x1xi8>
          %rhs1 = vector.transfer_read %arg4[%c0, %c0], %cst_i8 : memref<1x16xi8>, vector<1x16xi8>
          %acc1 = vector.transfer_read %arg5[%c0, %c0], %cst_i32 {in_bounds = [true, true]} : memref<16x16xi32>, vector<16x16xi32>
          %res1 = vector.contract {
            indexing_maps = [
              affine_map<(d0, d1, d2) -> (d0, d2)>,
              affine_map<(d0, d1, d2) -> (d2, d1)>,
              affine_map<(d0, d1, d2) -> (d0, d1)>
            ],
            iterator_types = ["parallel", "parallel", "reduction"]
          } %lhs1, %rhs1, %acc1 : vector<16x1xi8>, vector<1x16xi8> into vector<16x16xi32>
          vector.transfer_write %res1, %arg5[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xi32>, memref<16x16xi32>
          return
        }
      }
    }
  }

  // A CoralNPU target without +zvtbase must be left alone.
  // CHECK-LABEL: hal.executable private @matmul_no_zvtbase {
  hal.executable private @matmul_no_zvtbase {
    hal.executable.variant public @coralnpu_elf target(#target_no_zvt) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_no_zvtbase(
        func.func @matmul_no_zvtbase(%arg0: memref<16x1xf32>, %arg1: memref<1x16xf32>, %arg2: memref<16x16xf32>) {
          // CHECK-NOT: llvm.inline_asm
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
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
          return
        }
      }
    }
  }

  // A non-CoralNPU backend must likewise be untouched.
  // CHECK-LABEL: hal.executable private @matmul_other_backend {
  hal.executable private @matmul_other_backend {
    hal.executable.variant public @embedded_elf target(#target_llvmcpu) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_other_backend(
        func.func @matmul_other_backend(%arg0: memref<16x1xf32>, %arg1: memref<1x16xf32>, %arg2: memref<16x16xf32>) {
          // CHECK-NOT: llvm.inline_asm
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
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
          return
        }
      }
    }
  }

  // 8x4 is not a supported tile shape, so this takes the generic lowering.
  // CHECK-LABEL: hal.executable private @matmul_unsupported_shape {
  hal.executable private @matmul_unsupported_shape {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_unsupported_shape(
        func.func @matmul_unsupported_shape(%arg0: memref<8x1xf32>, %arg1: memref<1x4xf32>, %arg2: memref<8x4xf32>) {
          // CHECK-NOT: llvm.inline_asm
          // CHECK-NOT: vector.contract
          %c0 = arith.constant 0 : index
          %cst_0 = arith.constant 0.0 : f32
          %lhs = vector.transfer_read %arg0[%c0, %c0], %cst_0 : memref<8x1xf32>, vector<8x1xf32>
          %rhs = vector.transfer_read %arg1[%c0, %c0], %cst_0 : memref<1x4xf32>, vector<1x4xf32>
          %acc = vector.transfer_read %arg2[%c0, %c0], %cst_0 : memref<8x4xf32>, vector<8x4xf32>
          %res = vector.contract {
            indexing_maps = [
              affine_map<(d0, d1, d2) -> (d0, d2)>,
              affine_map<(d0, d1, d2) -> (d2, d1)>,
              affine_map<(d0, d1, d2) -> (d0, d1)>
            ],
            iterator_types = ["parallel", "parallel", "reduction"]
          } %lhs, %rhs, %acc : vector<8x1xf32>, vector<1x4xf32> into vector<8x4xf32>
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<8x4xf32>, memref<8x4xf32>
          return
        }
      }
    }
  }

  // Fused elementwise consumer (bias + ReLU) drains to an entry-block scratch
  // tile and rolls the 2D consumer DAG into a 1D row loop.
  // CHECK-LABEL: hal.executable private @matmul_fused_consumer_row_loop {
  hal.executable private @matmul_fused_consumer_row_loop {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_fused_consumer_row_loop(
        func.func @matmul_fused_consumer_row_loop(%arg0: memref<16x8xf32>, %arg1: memref<8x16xf32>, %arg2: memref<16xf32>, %arg3: memref<16x16xf32>) {
          // CHECK: %[[SCRATCH:.*]] = memref.alloc() alignment = 64 : memref<16x16xf32, #iree_codegen.workgroup_local>
          // CHECK: llvm.inline_asm {{.*}}vtzero mt0{{.*}}vtfmm.tvv mt0{{.*}}vtmv.v.t v0, t0{{.*}}vse32.v v0, (t4)
          // CHECK: scf.for %[[R:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
          // CHECK:   %[[ROW:.*]] = vector.load %[[SCRATCH]][%[[R]], %{{.*}}] : memref<16x16xf32, #iree_codegen.workgroup_local>, vector<16xf32>
          // CHECK:   %[[ADD:.*]] = arith.addf %[[ROW]], %{{.*}} : vector<16xf32>
          // CHECK:   %[[MAX:.*]] = arith.maximumf %[[ADD]], %{{.*}} : vector<16xf32>
          // CHECK:   %[[IDX:.*]] = arith.addi %{{.*}}, %[[R]] : index
          // CHECK:   vector.store %[[MAX]], %arg3[%[[IDX]], %{{.*}}] : memref<16x16xf32>, vector<16xf32>
          // CHECK: }
          %c0 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c1 = arith.constant 1 : index
          %cst_0 = arith.constant 0.0 : f32
          %zero_2d = arith.constant dense<0.0> : vector<16x16xf32>
          %res = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %zero_2d) -> (vector<16x16xf32>) {
            %lhs = vector.transfer_read %arg0[%c0, %iv], %cst_0 : memref<16x8xf32>, vector<16x1xf32>
            %rhs = vector.transfer_read %arg1[%iv, %c0], %cst_0 : memref<8x16xf32>, vector<1x16xf32>
            %next = vector.contract {
              indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
              iterator_types = ["parallel", "parallel", "reduction"]
            } %lhs, %rhs, %acc : vector<16x1xf32>, vector<1x16xf32> into vector<16x16xf32>
            scf.yield %next : vector<16x16xf32>
          }
          %bias_1d = vector.load %arg2[%c0] : memref<16xf32>, vector<16xf32>
          %bias_2d = vector.broadcast %bias_1d : vector<16xf32> to vector<16x16xf32>
          %biased = arith.addf %res, %bias_2d : vector<16x16xf32>
          %relu = arith.maximumf %biased, %zero_2d : vector<16x16xf32>
          vector.transfer_write %relu, %arg3[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
          return
        }
      }
    }
  }

  // Two scratch-fused chains share one scratch tile: the first chain must read
  // scratch right after its own drain before the second drain overwrites it.
  // CHECK-LABEL: hal.executable private @matmul_shared_scratch {
  hal.executable private @matmul_shared_scratch {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_shared_scratch(
        func.func @matmul_shared_scratch(%arg0: memref<16x1xf32>, %arg1: memref<1x16xf32>, %arg2: memref<16x16xf32>, %arg3: memref<16x16xf32>) {
          // CHECK: %[[SCRATCH:.*]] = memref.alloc() alignment = 64 : memref<16x16xf32, #iree_codegen.workgroup_local>
          // CHECK: llvm.inline_asm {{.*}}vtzero mt0{{.*}}vtfmm.tvv mt0
          // CHECK: llvm.inline_asm {{.*}}vse32.v v0, (t4)
          // CHECK: vector.transfer_read %[[SCRATCH]]
          // CHECK: llvm.inline_asm {{.*}}vtzero mt0{{.*}}vtfmm.tvv mt0
          // CHECK: llvm.inline_asm {{.*}}vse32.v v0, (t4)
          // CHECK: vector.transfer_read %[[SCRATCH]]
          // CHECK-NOT: scf.for
          %c0 = arith.constant 0 : index
          %cst_0 = arith.constant 0.0 : f32
          %acc = arith.constant dense<0.0> : vector<16x16xf32>
          %bias = arith.constant dense<1.0> : vector<16x16xf32>
          %lhs = vector.transfer_read %arg0[%c0, %c0], %cst_0 : memref<16x1xf32>, vector<16x1xf32>
          %rhs = vector.transfer_read %arg1[%c0, %c0], %cst_0 : memref<1x16xf32>, vector<1x16xf32>
          %r1 = vector.contract {
            indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
            iterator_types = ["parallel", "parallel", "reduction"]
          } %lhs, %rhs, %acc : vector<16x1xf32>, vector<1x16xf32> into vector<16x16xf32>
          %r2 = vector.contract {
            indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
            iterator_types = ["parallel", "parallel", "reduction"]
          } %lhs, %rhs, %acc : vector<16x1xf32>, vector<1x16xf32> into vector<16x16xf32>
          %s1 = arith.addf %r1, %bias : vector<16x16xf32>
          vector.transfer_write %s1, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
          %s2 = arith.addf %r2, %bias : vector<16x16xf32>
          vector.transfer_write %s2, %arg3[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
          return
        }
      }
    }
  }

  // Transposed output C[N, M], k = 3..11: operands swap (B rows become 2x1 LHS
  // tiles mt0/mt4, A is strided RHS), addresses start at k = 3, 8 asm steps.
  // CHECK-LABEL: hal.executable private @matmul_transposed_out {
  hal.executable private @matmul_transposed_out {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_transposed_out(
        func.func @matmul_transposed_out(%arg0: memref<16x11xf32>, %arg1: memref<11x32xf32>, %arg2: memref<32x16xf32>) {
          // CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
          // CHECK-DAG: arith.addi %{{.*}}, %[[C3]] : index
          // CHECK-DAG: arith.muli %[[C3]], %{{.*}} : index
          // CHECK: llvm.inline_asm {{.*}}vtzero mt0{{.*}}vtzero mt4{{.*}}li t6, 8{{.*}}vl8re32.v v24, (t0){{.*}}vlse32.v v8, (t2), t3{{.*}}vtfmm.tvv mt0, v24, v8{{.*}}vtfmm.tvv mt4, v28, v8{{.*}}addi t2, t2, 4
          // CHECK-NOT: scf.for
          %c0 = arith.constant 0 : index
          %c3 = arith.constant 3 : index
          %c11 = arith.constant 11 : index
          %c1 = arith.constant 1 : index
          %cst_0 = arith.constant 0.0 : f32
          %acc_init = arith.constant dense<0.0> : vector<32x16xf32>
          %res = scf.for %iv = %c3 to %c11 step %c1 iter_args(%acc = %acc_init) -> (vector<32x16xf32>) {
            %lhs = vector.transfer_read %arg0[%c0, %iv], %cst_0 : memref<16x11xf32>, vector<16x1xf32>
            %rhs = vector.transfer_read %arg1[%iv, %c0], %cst_0 : memref<11x32xf32>, vector<1x32xf32>
            %next = vector.contract {
              indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d1, d0)>],
              iterator_types = ["parallel", "parallel", "reduction"]
            } %lhs, %rhs, %acc : vector<16x1xf32>, vector<1x32xf32> into vector<32x16xf32>
            scf.yield %next : vector<32x16xf32>
          }
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<32x16xf32>, memref<32x16xf32>
          return
        }
      }
    }
  }

  // INT8 result converted to f32 and [1, 0]-transposed: the f32 rows cannot
  // be staged in the i32 scratch tile, so the consumer reads the tile back
  // instead of running as a row loop.
  // CHECK-LABEL: hal.executable private @matmul_i8_sitofp_transposed_out {
  hal.executable private @matmul_i8_sitofp_transposed_out {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_i8_sitofp_transposed_out(
        func.func @matmul_i8_sitofp_transposed_out(%arg0: memref<16x8xi8>, %arg1: memref<8x16xi8>, %arg2: memref<16x16xf32>) {
          // CHECK: llvm.inline_asm {{.*}}vtmms.tvv mt0
          // CHECK: %[[ACC:.*]] = vector.transfer_read
          // CHECK: %[[F:.*]] = arith.sitofp %[[ACC]] : vector<16x16xi32> to vector<16x16xf32>
          // CHECK: vector.transpose %[[F]], [1, 0]
          %c0 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c1 = arith.constant 1 : index
          %cst_i8 = arith.constant 0 : i8
          %acc_init = arith.constant dense<0> : vector<16x16xi32>
          %res = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %acc_init) -> (vector<16x16xi32>) {
            %lhs = vector.transfer_read %arg0[%c0, %iv], %cst_i8 : memref<16x8xi8>, vector<16x1xi8>
            %rhs = vector.transfer_read %arg1[%iv, %c0], %cst_i8 : memref<8x16xi8>, vector<1x16xi8>
            %next = vector.contract {
              indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
              iterator_types = ["parallel", "parallel", "reduction"]
            } %lhs, %rhs, %acc : vector<16x1xi8>, vector<1x16xi8> into vector<16x16xi32>
            scf.yield %next : vector<16x16xi32>
          }
          %f = arith.sitofp %res : vector<16x16xi32> to vector<16x16xf32>
          %t = vector.transpose %f, [1, 0] : vector<16x16xf32> to vector<16x16xf32>
          vector.transfer_write %t, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
          return
        }
      }
    }
  }

  // Partial M < 16 tile (8x16) with fused consumer: drains only 8 rows into
  // the 16x16 scratch tile and loops %r = 0..8 over the consumer.
  // CHECK-LABEL: hal.executable private @matmul_partial_m_rows {
  hal.executable private @matmul_partial_m_rows {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_partial_m_rows(
        func.func @matmul_partial_m_rows(%arg0: memref<8x4xf32>, %arg1: memref<4x16xf32>, %arg2: memref<16xf32>, %arg3: memref<8x16xf32>) {
          // CHECK: %[[SCRATCH:.*]] = memref.alloc() alignment = 64 : memref<16x16xf32, #iree_codegen.workgroup_local>
          // CHECK: llvm.inline_asm {{.*}}vtzero mt0{{.*}}vid.v v8\0A\09li t6, 8\0A\09vmsltu.vx v0, v8, t6{{.*}}vlse32.v v4, (t0), t1, v0.t\0A\09vle32.v v8, (t2)\0A\09vtfmm.tvv mt0, v4, v8{{.*}}li t6, 2\0A\092:\0A\09vtmv.v.t v0, t0\0A\09vse32.v v0, (t4)
          // CHECK: %[[C8:.*]] = arith.constant 8 : index
          // CHECK: scf.for %[[R:.*]] = %{{.*}} to %[[C8]] step %{{.*}} {
          // CHECK:   %[[ROW:.*]] = vector.load %[[SCRATCH]][%[[R]], %{{.*}}] : memref<16x16xf32, #iree_codegen.workgroup_local>, vector<16xf32>
          // CHECK:   %[[ADD:.*]] = arith.addf %[[ROW]], %{{.*}} : vector<16xf32>
          // CHECK:   %[[IDX:.*]] = arith.addi %{{.*}}, %[[R]] : index
          // CHECK:   vector.store %[[ADD]], %arg3[%[[IDX]], %{{.*}}] : memref<8x16xf32>, vector<16xf32>
          // CHECK: }
          %c0 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1 = arith.constant 1 : index
          %cst_0 = arith.constant 0.0 : f32
          %zero_2d = arith.constant dense<0.0> : vector<8x16xf32>
          %res = scf.for %iv = %c0 to %c4 step %c1 iter_args(%acc = %zero_2d) -> (vector<8x16xf32>) {
            %lhs = vector.transfer_read %arg0[%c0, %iv], %cst_0 : memref<8x4xf32>, vector<8x1xf32>
            %rhs = vector.transfer_read %arg1[%iv, %c0], %cst_0 : memref<4x16xf32>, vector<1x16xf32>
            %next = vector.contract {
              indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
              iterator_types = ["parallel", "parallel", "reduction"]
            } %lhs, %rhs, %acc : vector<8x1xf32>, vector<1x16xf32> into vector<8x16xf32>
            scf.yield %next : vector<8x16xf32>
          }
          %bias_1d = vector.load %arg2[%c0] : memref<16xf32>, vector<16xf32>
          %bias_2d = vector.broadcast %bias_1d : vector<16xf32> to vector<8x16xf32>
          %biased = arith.addf %res, %bias_2d : vector<8x16xf32>
          vector.transfer_write %biased, %arg3[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
          return
        }
      }
    }
  }

  // Partial M and N tile (8x12): masks the A loads to 8 rows and the B loads to
  // 12 columns, copying each mask to v0 before its loads, and drains 12
  // columns via msettn zero, 12.
  // CHECK-LABEL: hal.executable private @matmul_partial_n_cols {
  hal.executable private @matmul_partial_n_cols {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_partial_n_cols(
        func.func @matmul_partial_n_cols(%arg0: memref<8x4xf32>, %arg1: memref<4x12xf32>, %arg2: memref<8x12xf32>) {
          // CHECK: llvm.inline_asm {{.*}}vtzero mt0{{.*}}vid.v v8\0A\09li t6, 8\0A\09vmsltu.vx v1, v8, t6\0A\09li t6, 12\0A\09vmsltu.vx v2, v8, t6\0A\09vmv1r.v v0, v1\0A\09vlse32.v v4, (t0), t1, v0.t\0A\09vmv1r.v v0, v2\0A\09vle32.v v8, (t2), v0.t\0A\09vtfmm.tvv mt0, v4, v8{{.*}}li t6, 12\0A\09msettn zero, t6{{.*}}li t6, 2\0A\092:\0A\09vtmv.v.t v0, t0\0A\09vse32.v v0, (t4)
          // CHECK-NOT: scf.for
          %c0 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1 = arith.constant 1 : index
          %cst_0 = arith.constant 0.0 : f32
          %zero_2d = arith.constant dense<0.0> : vector<8x12xf32>
          %res = scf.for %iv = %c0 to %c4 step %c1 iter_args(%acc = %zero_2d) -> (vector<8x12xf32>) {
            %lhs = vector.transfer_read %arg0[%c0, %iv], %cst_0 : memref<8x4xf32>, vector<8x1xf32>
            %rhs = vector.transfer_read %arg1[%iv, %c0], %cst_0 : memref<4x12xf32>, vector<1x12xf32>
            %next = vector.contract {
              indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
              iterator_types = ["parallel", "parallel", "reduction"]
            } %lhs, %rhs, %acc : vector<8x1xf32>, vector<1x12xf32> into vector<8x12xf32>
            scf.yield %next : vector<8x12xf32>
          }
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<8x12xf32>, memref<8x12xf32>
          return
        }
      }
    }
  }

  // BF16 partial M and N tile (8x12): the masks wait in v1 and v4, clear of
  // the A operand in v2-v3.
  // CHECK-LABEL: hal.executable private @matmul_partial_mn_bf16 {
  hal.executable private @matmul_partial_mn_bf16 {
    hal.executable.variant public @coralnpu_elf target(#target_bf16) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_partial_mn_bf16(
        func.func @matmul_partial_mn_bf16(%arg0: memref<8x4xbf16>, %arg1: memref<4x12xbf16>, %arg2: memref<8x12xf32>) {
          // CHECK: llvm.inline_asm {{.*}}vid.v v8\0A\09li t6, 8\0A\09vmsltu.vx v1, v8, t6\0A\09li t6, 12\0A\09vmsltu.vx v4, v8, t6\0A\09vmv1r.v v0, v1\0A\09vlse16.v v2, (t0), t1, v0.t\0A\09vmv1r.v v0, v4\0A\09vle16.v v8, (t2), v0.t\0A\09vtfmm.alt.tvv mt0, v2, v8
          %c0 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1 = arith.constant 1 : index
          %cst_0 = arith.constant 0.0 : bf16
          %zero_2d = arith.constant dense<0.0> : vector<8x12xf32>
          %res = scf.for %iv = %c0 to %c4 step %c1 iter_args(%acc = %zero_2d) -> (vector<8x12xf32>) {
            %lhs0 = vector.transfer_read %arg0[%c0, %iv], %cst_0 : memref<8x4xbf16>, vector<8x1xbf16>
            %rhs0 = vector.transfer_read %arg1[%iv, %c0], %cst_0 : memref<4x12xbf16>, vector<1x12xbf16>
            %lhs = arith.extf %lhs0 : vector<8x1xbf16> to vector<8x1xf32>
            %rhs = arith.extf %rhs0 : vector<1x12xbf16> to vector<1x12xf32>
            %next = vector.contract {
              indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
              iterator_types = ["parallel", "parallel", "reduction"]
            } %lhs, %rhs, %acc : vector<8x1xf32>, vector<1x12xf32> into vector<8x12xf32>
            scf.yield %next : vector<8x12xf32>
          }
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<8x12xf32>, memref<8x12xf32>
          return
        }
      }
    }
  }

  // INT8 partial M and N tile (8x12): the masks wait in v2 and v4, clear of
  // the A operand in v1 (v1/v3/v5/v7 at tk = 4).
  // CHECK-LABEL: hal.executable private @matmul_partial_mn_i8 {
  hal.executable private @matmul_partial_mn_i8 {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_partial_mn_i8(
        func.func @matmul_partial_mn_i8(%arg0: memref<8x4xi8>, %arg1: memref<4x12xi8>, %arg2: memref<8x12xi32>) {
          // CHECK: llvm.inline_asm {{.*}}vid.v v8\0A\09li t6, 8\0A\09vmsltu.vx v2, v8, t6\0A\09li t6, 12\0A\09vmsltu.vx v4, v8, t6\0A\09vmv1r.v v0, v2\0A\09vlse8.v v1, (t0), t1, v0.t\0A\09vmv1r.v v0, v4\0A\09vle8.v v8, (t2), v0.t\0A\09vtmms.tvv mt0, v1, v8
          %c0 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1 = arith.constant 1 : index
          %cst_0 = arith.constant 0 : i8
          %zero_2d = arith.constant dense<0> : vector<8x12xi32>
          %res = scf.for %iv = %c0 to %c4 step %c1 iter_args(%acc = %zero_2d) -> (vector<8x12xi32>) {
            %lhs = vector.transfer_read %arg0[%c0, %iv], %cst_0 : memref<8x4xi8>, vector<8x1xi8>
            %rhs = vector.transfer_read %arg1[%iv, %c0], %cst_0 : memref<4x12xi8>, vector<1x12xi8>
            %next = vector.contract {
              indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
              iterator_types = ["parallel", "parallel", "reduction"]
            } %lhs, %rhs, %acc : vector<8x1xi8>, vector<1x12xi8> into vector<8x12xi32>
            scf.yield %next : vector<8x12xi32>
          }
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<8x12xi32>, memref<8x12xi32>
          return
        }
      }
    }
  }

  // Partial N tile (16x12) with an N x K RHS: B loads via vlse32.v, the drain
  // writes 12 columns, and iree-v3.12.0-0006 leaves the contraction to this
  // pass (GATE).
  // CHECK-LABEL: hal.executable private @matmul_partial_n_transposed_rhs {
  hal.executable private @matmul_partial_n_transposed_rhs {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_partial_n_transposed_rhs(
        // GATE-LABEL: func.func @matmul_partial_n_transposed_rhs(
        func.func @matmul_partial_n_transposed_rhs(%arg0: memref<16x4xf32>, %arg1: memref<12x4xf32>, %arg2: memref<16x12xf32>) {
          // CHECK: llvm.inline_asm {{.*}}vtzero mt0{{.*}}vid.v v8\0A\09li t6, 12\0A\09vmsltu.vx v0, v8, t6{{.*}}vlse32.v v4, (t0), t1\0A\09vlse32.v v8, (t2), t3, v0.t\0A\09vtfmm.tvv mt0, v4, v8{{.*}}li t6, 12\0A\09msettn zero, t6
          // CHECK-NOT: scf.for
          // GATE-NOT: func.func
          // GATE: vector.contract
          %c0 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1 = arith.constant 1 : index
          %cst_0 = arith.constant 0.0 : f32
          %zero_2d = arith.constant dense<0.0> : vector<16x12xf32>
          %res = scf.for %iv = %c0 to %c4 step %c1 iter_args(%acc = %zero_2d) -> (vector<16x12xf32>) {
            %lhs = vector.transfer_read %arg0[%c0, %iv], %cst_0 : memref<16x4xf32>, vector<16x1xf32>
            %rhs = vector.transfer_read %arg1[%c0, %iv], %cst_0 : memref<12x4xf32>, vector<12x1xf32>
            %next = vector.contract {
              indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
              iterator_types = ["parallel", "parallel", "reduction"]
            } %lhs, %rhs, %acc : vector<16x1xf32>, vector<12x1xf32> into vector<16x12xf32>
            scf.yield %next : vector<16x12xf32>
          }
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x12xf32>, memref<16x12xf32>
          return
        }
      }
    }
  }

  // Peeled M = 1 tail (scalar LHS via from_elements) with bf16 truncf output:
  // rowsM = 1 skips the 4-row BF16 drain, so 1 f32 row drains to scratch.
  // CHECK-LABEL: hal.executable private @matmul_partial_m1 {
  hal.executable private @matmul_partial_m1 {
    hal.executable.variant public @coralnpu_elf target(#target_bf16) {
      builtin.module {
        // CHECK-LABEL: func.func @matmul_partial_m1(
        func.func @matmul_partial_m1(%arg0: memref<1x4xbf16>, %arg1: memref<4x16xbf16>, %arg2: memref<1x16xbf16>) {
          // CHECK: %[[SCRATCH:.*]] = memref.alloc() alignment = 64 : memref<16x16xf32, #iree_codegen.workgroup_local>
          // CHECK: llvm.inline_asm {{.*}}vtzero mt0{{.*}}li t5, 16450{{.*}}vid.v v8\0A\09li t6, 1\0A\09vmsltu.vx v0, v8, t6{{.*}}vlse16.v v2, (t0), t1, v0.t\0A\09vle16.v v8, (t2)\0A\09{{.*}}vtfmm.alt.tvv mt0, v2, v8{{.*}}li t0, 0\0A\09vtmv.v.t v0, t0\0A\09vse32.v v0, (t4)
          // CHECK-NOT: vfncvtbf16
          %c0 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1 = arith.constant 1 : index
          %cst_0 = arith.constant 0.0 : bf16
          %zero_2d = arith.constant dense<0.0> : vector<1x16xf32>
          %res = scf.for %iv = %c0 to %c4 step %c1 iter_args(%acc = %zero_2d) -> (vector<1x16xf32>) {
            %s = memref.load %arg0[%c0, %iv] : memref<1x4xbf16>
            %lhs0 = vector.from_elements %s : vector<1x1xbf16>
            %rhs0 = vector.transfer_read %arg1[%iv, %c0], %cst_0 : memref<4x16xbf16>, vector<1x16xbf16>
            %lhs = arith.extf %lhs0 : vector<1x1xbf16> to vector<1x1xf32>
            %rhs = arith.extf %rhs0 : vector<1x16xbf16> to vector<1x16xf32>
            %next = vector.contract {
              indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
              iterator_types = ["parallel", "parallel", "reduction"]
            } %lhs, %rhs, %acc : vector<1x1xf32>, vector<1x16xf32> into vector<1x16xf32>
            scf.yield %next : vector<1x16xf32>
          }
          %trunc = arith.truncf %res : vector<1x16xf32> to vector<1x16xbf16>
          vector.transfer_write %trunc, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<1x16xbf16>, memref<1x16xbf16>
          return
        }
      }
    }
  }

  // 3x3 conv chain (kh, kw, ic): vtzero once before kh, the ic loop fuses into
  // a GEMM K-loop accumulating across kh/kw, and one drain after kh.
  // CHECK-LABEL: hal.executable private @conv_multi_reduction_loop_chain {
  hal.executable private @conv_multi_reduction_loop_chain {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @conv_multi_reduction_loop_chain(
        func.func @conv_multi_reduction_loop_chain(%arg0: memref<3x3x16x8xf32>, %arg1: memref<3x3x8x16xf32>, %arg2: memref<16x16xf32>) {
          // CHECK: llvm.inline_asm {{.*}}vtzero mt0
          // CHECK: scf.for %[[KH:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
          // CHECK:   scf.for %[[KW:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
          // CHECK:     llvm.inline_asm {{.*}}li t6, 8{{.*}}vtfmm.tvv mt0, v4, v8
          // CHECK-NOT: vtmv.v.t
          // CHECK:   }
          // CHECK: }
          // CHECK: llvm.inline_asm {{.*}}vtmv.v.t v0, t0{{.*}}vse32.v v0, (t4)
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c3 = arith.constant 3 : index
          %c8 = arith.constant 8 : index
          %cst_0 = arith.constant 0.0 : f32
          %zero_2d = arith.constant dense<0.0> : vector<16x16xf32>
          %res = scf.for %kh = %c0 to %c3 step %c1 iter_args(%acc_kh = %zero_2d) -> (vector<16x16xf32>) {
            %res_kw = scf.for %kw = %c0 to %c3 step %c1 iter_args(%acc_kw = %acc_kh) -> (vector<16x16xf32>) {
              %res_ic = scf.for %ic = %c0 to %c8 step %c1 iter_args(%acc_ic = %acc_kw) -> (vector<16x16xf32>) {
                %lhs = vector.transfer_read %arg0[%kh, %kw, %c0, %ic], %cst_0 : memref<3x3x16x8xf32>, vector<16x1xf32>
                %rhs = vector.transfer_read %arg1[%kh, %kw, %ic, %c0], %cst_0 : memref<3x3x8x16xf32>, vector<1x16xf32>
                %next = vector.contract {
                  indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                  iterator_types = ["parallel", "parallel", "reduction"]
                } %lhs, %rhs, %acc_ic : vector<16x1xf32>, vector<1x16xf32> into vector<16x16xf32>
                scf.yield %next : vector<16x16xf32>
              }
              scf.yield %res_ic : vector<16x16xf32>
            }
            scf.yield %res_kw : vector<16x16xf32>
          }
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
          return
        }
      }
    }
  }

  // Strided conv outer-product loop (M=3 scalar loads at ow*2 + kw + r*2
  // broadcast and FMA'd into a 3x16 accumulator): reassembled into a 3x16
  // Zvt contraction with doubled row stride.
  // CHECK-LABEL: hal.executable private @conv_strided_outer_product {
  hal.executable private @conv_strided_outer_product {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @conv_strided_outer_product(
        func.func @conv_strided_outer_product(%arg0: memref<1x8x4xf32>, %arg1: memref<4x16xf32>, %arg2: memref<3x16xf32>, %ow: index, %kw: index) {
          // CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
          // CHECK-DAG: arith.muli %{{.*}}, %[[C8]] : index
          // CHECK: llvm.inline_asm {{.*}}vtzero mt0{{.*}}vid.v v8\0A\09li t6, 3\0A\09vmsltu.vx v0, v8, t6{{.*}}vlse32.v v4, (t0), t1, v0.t\0A\09vle32.v v8, (t2)\0A\09vtfmm.tvv mt0, v4, v8
          // CHECK-NOT: scf.for
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c2 = arith.constant 2 : index
          %c4 = arith.constant 4 : index
          %cst_0 = arith.constant 0.0 : f32
          %zero = arith.constant dense<0.0> : vector<3x16xf32>
          %res = scf.for %ic = %c0 to %c4 step %c1 iter_args(%acc = %zero) -> (vector<3x16xf32>) {
            %rhs = vector.transfer_read %arg1[%ic, %c0], %cst_0 {in_bounds = [true]} : memref<4x16xf32>, vector<16xf32>
            %r0 = vector.extract %acc[0] : vector<16xf32> from vector<3x16xf32>
            %r1 = vector.extract %acc[1] : vector<16xf32> from vector<3x16xf32>
            %r2 = vector.extract %acc[2] : vector<16xf32> from vector<3x16xf32>
            %m0 = arith.muli %ow, %c2 : index
            %i0 = arith.addi %m0, %kw : index
            %s0 = memref.load %arg0[%c0, %i0, %ic] : memref<1x8x4xf32>
            %b0 = vector.broadcast %s0 : f32 to vector<16xf32>
            %f0 = vector.fma %rhs, %b0, %r0 : vector<16xf32>
            %m1 = arith.muli %ow, %c2 : index
            %a1 = arith.addi %m1, %kw : index
            %i1 = arith.addi %a1, %c2 : index
            %s1 = memref.load %arg0[%c0, %i1, %ic] : memref<1x8x4xf32>
            %b1 = vector.broadcast %s1 : f32 to vector<16xf32>
            %f1 = vector.fma %rhs, %b1, %r1 : vector<16xf32>
            %m2 = arith.muli %ow, %c2 : index
            %a2 = arith.addi %m2, %kw : index
            %i2 = arith.addi %a2, %c4 : index
            %s2 = memref.load %arg0[%c0, %i2, %ic] : memref<1x8x4xf32>
            %b2 = vector.broadcast %s2 : f32 to vector<16xf32>
            %f2 = vector.fma %rhs, %b2, %r2 : vector<16xf32>
            %u0 = vector.insert_strided_slice %f0, %acc offsets = [0, 0], strides = [1] : vector<16xf32> into vector<3x16xf32>
            %u1 = vector.insert_strided_slice %f1, %u0 offsets = [1, 0], strides = [1] : vector<16xf32> into vector<3x16xf32>
            %u2 = vector.insert_strided_slice %f2, %u1 offsets = [2, 0], strides = [1] : vector<16xf32> into vector<3x16xf32>
            scf.yield %u2 : vector<3x16xf32>
          }
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<3x16xf32>, memref<3x16xf32>
          return
        }
      }
    }
  }

  // 1-channel conv kernel-column loop (16 pixels from ow + kw, the channel dim
  // dropped): kw offsets the pixel index, whose stride is the unit channel
  // stride, so the loop fuses into a K loop of contiguous A rows.
  // CHECK-LABEL: hal.executable private @conv_1ch_kw_loop {
  hal.executable private @conv_1ch_kw_loop {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @conv_1ch_kw_loop(
        func.func @conv_1ch_kw_loop(%arg0: memref<1x4x34x1xi8>, %arg1: memref<3x1x16xi8>, %arg2: memref<16x16xi32>, %ow: index) {
          // CHECK: llvm.inline_asm {{.*}}vle8.v v1, (t0){{.*}}vle8.v v8, (t2){{.*}}vtmms.tvv mt0, v1, v8{{.*}}add t0, t0, t1
          // CHECK-NOT: scf.for
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c3 = arith.constant 3 : index
          %zero = arith.constant dense<0> : vector<16x16xi32>
          %img = memref.subview %arg0[0, 0, 0, 0] [1, 4, 34, 1] [1, 1, 1, 1] : memref<1x4x34x1xi8> to memref<1x4x34xi8, strided<[136, 34, 1]>>
          %res = scf.for %kw = %c0 to %c3 step %c1 iter_args(%acc = %zero) -> (vector<16x16xi32>) {
            %w = arith.addi %kw, %ow : index
            %x8 = vector.load %img[%c0, %c0, %w] : memref<1x4x34xi8, strided<[136, 34, 1]>>, vector<16xi8>
            %x = arith.extsi %x8 : vector<16xi8> to vector<16xi32>
            %e0 = vector.extract %x[0] : i32 from vector<16xi32>
            %e1 = vector.extract %x[1] : i32 from vector<16xi32>
            %e2 = vector.extract %x[2] : i32 from vector<16xi32>
            %e3 = vector.extract %x[3] : i32 from vector<16xi32>
            %e4 = vector.extract %x[4] : i32 from vector<16xi32>
            %e5 = vector.extract %x[5] : i32 from vector<16xi32>
            %e6 = vector.extract %x[6] : i32 from vector<16xi32>
            %e7 = vector.extract %x[7] : i32 from vector<16xi32>
            %e8 = vector.extract %x[8] : i32 from vector<16xi32>
            %e9 = vector.extract %x[9] : i32 from vector<16xi32>
            %e10 = vector.extract %x[10] : i32 from vector<16xi32>
            %e11 = vector.extract %x[11] : i32 from vector<16xi32>
            %e12 = vector.extract %x[12] : i32 from vector<16xi32>
            %e13 = vector.extract %x[13] : i32 from vector<16xi32>
            %e14 = vector.extract %x[14] : i32 from vector<16xi32>
            %e15 = vector.extract %x[15] : i32 from vector<16xi32>
            %lhs = vector.from_elements %e0, %e1, %e2, %e3, %e4, %e5, %e6, %e7, %e8, %e9, %e10, %e11, %e12, %e13, %e14, %e15 : vector<16x1xi32>
            %b8 = vector.load %arg1[%kw, %c0, %c0] : memref<3x1x16xi8>, vector<16xi8>
            %b = arith.extsi %b8 : vector<16xi8> to vector<16xi32>
            %rhs = vector.shape_cast %b : vector<16xi32> to vector<1x16xi32>
            %next = vector.contract {
              indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
              iterator_types = ["parallel", "parallel", "reduction"]
            } %lhs, %rhs, %acc : vector<16x1xi32>, vector<1x16xi32> into vector<16x16xi32>
            scf.yield %next : vector<16x16xi32>
          }
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xi32>, memref<16x16xi32>
          return
        }
      }
    }
  }

  // Strided 1-channel conv outer-product loop (M=3 pixels 2 apart in one load
  // from ow2 + kw, the channel dim dropped): reassembled into a contraction,
  // whose loop fuses into a K loop of A columns with a 2-byte row stride.
  // CHECK-LABEL: hal.executable private @conv_1ch_strided_outer_product {
  hal.executable private @conv_1ch_strided_outer_product {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK-LABEL: func.func @conv_1ch_strided_outer_product(
        func.func @conv_1ch_strided_outer_product(%arg0: memref<1x4x16x1xi8>, %arg1: memref<3x1x16xi8>, %arg2: memref<3x16xi32>, %ow2: index) {
          // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
          // CHECK-DAG: arith.muli %{{.*}}, %[[C2]] : index
          // CHECK: llvm.inline_asm {{.*}}vlse8.v v1, (t0), t1, v0.t{{.*}}vle8.v v8, (t2){{.*}}vtmms.tvv mt0, v1, v8{{.*}}addi t0, t0, 1
          // CHECK-NOT: scf.for
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c3 = arith.constant 3 : index
          %zero = arith.constant dense<0> : vector<3x16xi32>
          %img = memref.subview %arg0[0, 0, 0, 0] [1, 4, 16, 1] [1, 1, 1, 1] : memref<1x4x16x1xi8> to memref<1x4x16xi8, strided<[64, 16, 1]>>
          %res = scf.for %kw = %c0 to %c3 step %c1 iter_args(%acc = %zero) -> (vector<3x16xi32>) {
            %w = arith.addi %ow2, %kw : index
            %x = vector.load %img[%ow2, %ow2, %w] : memref<1x4x16xi8, strided<[64, 16, 1]>>, vector<5xi8>
            %b8 = vector.load %arg1[%kw, %c0, %c0] : memref<3x1x16xi8>, vector<16xi8>
            %b = arith.extsi %b8 : vector<16xi8> to vector<16xi32>
            %r0 = vector.extract %acc[0] : vector<16xi32> from vector<3x16xi32>
            %r1 = vector.extract %acc[1] : vector<16xi32> from vector<3x16xi32>
            %r2 = vector.extract %acc[2] : vector<16xi32> from vector<3x16xi32>
            %s0 = vector.extract %x[0] : i8 from vector<5xi8>
            %t0 = arith.extsi %s0 : i8 to i32
            %v0 = vector.broadcast %t0 : i32 to vector<16xi32>
            %m0 = arith.muli %b, %v0 : vector<16xi32>
            %f0 = arith.addi %m0, %r0 : vector<16xi32>
            %s1 = vector.extract %x[2] : i8 from vector<5xi8>
            %t1 = arith.extsi %s1 : i8 to i32
            %v1 = vector.broadcast %t1 : i32 to vector<16xi32>
            %m1 = arith.muli %b, %v1 : vector<16xi32>
            %f1 = arith.addi %m1, %r1 : vector<16xi32>
            %s2 = vector.extract %x[4] : i8 from vector<5xi8>
            %t2 = arith.extsi %s2 : i8 to i32
            %v2 = vector.broadcast %t2 : i32 to vector<16xi32>
            %m2 = arith.muli %b, %v2 : vector<16xi32>
            %f2 = arith.addi %m2, %r2 : vector<16xi32>
            %u0 = vector.insert_strided_slice %f0, %acc offsets = [0, 0], strides = [1] : vector<16xi32> into vector<3x16xi32>
            %u1 = vector.insert_strided_slice %f1, %u0 offsets = [1, 0], strides = [1] : vector<16xi32> into vector<3x16xi32>
            %u2 = vector.insert_strided_slice %f2, %u1 offsets = [2, 0], strides = [1] : vector<16xi32> into vector<3x16xi32>
            scf.yield %u2 : vector<3x16xi32>
          }
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<3x16xi32>, memref<3x16xi32>
          return
        }
      }
    }
  }

  // A 32-element LHS that slides with k is not one broadcast element, so the
  // loop does not lower to vfwmaccbf16.vf.
  // CHECK-LABEL: hal.executable private @vecmat_bf16_vector_lhs {
  hal.executable private @vecmat_bf16_vector_lhs {
    hal.executable.variant public @coralnpu_elf target(#target_bf16_wma) {
      builtin.module {
        // CHECK-LABEL: func.func @vecmat_bf16_vector_lhs(
        func.func @vecmat_bf16_vector_lhs(%arg0: memref<1x40xbf16>, %arg1: memref<8x32xbf16>, %arg2: memref<1x32xf32>) {
          // CHECK-NOT: llvm.inline_asm
          // CHECK: scf.for
          // CHECK: vector.fma
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c8 = arith.constant 8 : index
          %cst_0 = arith.constant 0.0 : bf16
          %zero = arith.constant dense<0.0> : vector<32xf32>
          %res = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %zero) -> (vector<32xf32>) {
            %x16 = vector.transfer_read %arg0[%c0, %iv], %cst_0 {in_bounds = [true]} : memref<1x40xbf16>, vector<32xbf16>
            %w16 = vector.transfer_read %arg1[%iv, %c0], %cst_0 {in_bounds = [true]} : memref<8x32xbf16>, vector<32xbf16>
            %x = arith.extf %x16 : vector<32xbf16> to vector<32xf32>
            %w = arith.extf %w16 : vector<32xbf16> to vector<32xf32>
            %next = vector.fma %x, %w, %acc : vector<32xf32>
            scf.yield %next : vector<32xf32>
          }
          vector.transfer_write %res, %arg2[%c0, %c0] {in_bounds = [true]} : vector<32xf32>, memref<1x32xf32>
          return
        }
      }
    }
  }

  // Jammed 2x32 BF16 vecmat K-loop: flh scalar LHS, two vl8re16.v RHS loads,
  // vfwmaccbf16.vf into v16/v24, both reloaded from a scratch tile.
  // CHECK-LABEL: hal.executable private @vecmat_bf16_jammed {
  hal.executable private @vecmat_bf16_jammed {
    hal.executable.variant public @coralnpu_elf target(#target_bf16_wma) {
      builtin.module {
        // CHECK-LABEL: func.func @vecmat_bf16_jammed(
        // NOVSET-LABEL: func.func @vecmat_bf16_jammed(
        func.func @vecmat_bf16_jammed(%arg0: memref<1x8xbf16>, %arg1: memref<8x64xbf16>, %arg2: memref<1x64xbf16>) {
          // CHECK: %[[SCRATCH:.*]] = memref.alloc() alignment = 64 : memref<16x32xf32, #iree_codegen.workgroup_local>
          // CHECK: llvm.inline_asm {{.*}}vmv.v.i v16, 0{{.*}}vmv.v.i v24, 0{{.*}}flh fa0, 0(t2){{.*}}vl8re16.v v0, (t0){{.*}}vfwmaccbf16.vf v16, fa0, v0{{.*}}vfwmaccbf16.vf v24, fa0, v4{{.*}}flh fa1, 14(t2){{.*}}vse32.v v16, ($3){{.*}}addi t0, $3, 128{{.*}}vse32.v v24, (t0)
          // CHECK: vector.load %[[SCRATCH]]
          // CHECK: vector.load %[[SCRATCH]]
          // CHECK-NOT: scf.for
          // NOVSET: llvm.inline_asm {{.*}}vsetvli{{.*}}vsetvli{{.*}}vsetvli
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c8 = arith.constant 8 : index
          %c32 = arith.constant 32 : index
          %cst_0 = arith.constant 0.0 : bf16
          %zero = arith.constant dense<0.0> : vector<32xf32>
          %res:2 = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc0 = %zero, %acc1 = %zero) -> (vector<32xf32>, vector<32xf32>) {
            %s = memref.load %arg0[%c0, %iv] : memref<1x8xbf16>
            %b = vector.broadcast %s : bf16 to vector<32xbf16>
            %lhs = arith.extf %b : vector<32xbf16> to vector<32xf32>
            %r0_16 = vector.transfer_read %arg1[%iv, %c0], %cst_0 {in_bounds = [true]} : memref<8x64xbf16>, vector<32xbf16>
            %r1_16 = vector.transfer_read %arg1[%iv, %c32], %cst_0 {in_bounds = [true]} : memref<8x64xbf16>, vector<32xbf16>
            %r0 = arith.extf %r0_16 : vector<32xbf16> to vector<32xf32>
            %r1 = arith.extf %r1_16 : vector<32xbf16> to vector<32xf32>
            %next0 = vector.fma %lhs, %r0, %acc0 : vector<32xf32>
            %next1 = vector.fma %lhs, %r1, %acc1 : vector<32xf32>
            scf.yield %next0, %next1 : vector<32xf32>, vector<32xf32>
          }
          %t0 = arith.truncf %res#0 : vector<32xf32> to vector<32xbf16>
          %t1 = arith.truncf %res#1 : vector<32xf32> to vector<32xbf16>
          vector.transfer_write %t0, %arg2[%c0, %c0] {in_bounds = [true]} : vector<32xbf16>, memref<1x64xbf16>
          vector.transfer_write %t1, %arg2[%c0, %c32] {in_bounds = [true]} : vector<32xbf16>, memref<1x64xbf16>
          return
        }
      }
    }
  }
}
