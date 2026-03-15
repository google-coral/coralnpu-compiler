/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "runtime/sim/simulator_inline.h"

#include <string.h>

#include "iree/base/api.h"
#include "runtime/sim/simulator_api.h"
#include "runtime/sim/simulator_elf_loader.h"

static uint32_t iree_rv32_encode_auipc(int rd, uint32_t imm20_upper) {
  return (imm20_upper & 0xFFFFF000u) | ((uint32_t)rd << 7) | 0x17u;
}

static uint32_t iree_rv32_encode_addi(int rd, int rs1, int32_t imm12) {
  uint32_t imm = (uint32_t)imm12 & 0xFFFu;
  return (imm << 20) | ((uint32_t)rs1 << 15) | (0u << 12) |
         ((uint32_t)rd << 7) | 0x13u;
}

static uint32_t iree_rv32_encode_lw(int rd, int rs1, int32_t imm12) {
  uint32_t imm = (uint32_t)imm12 & 0xFFFu;
  return (imm << 20) | ((uint32_t)rs1 << 15) | (2u << 12) |
         ((uint32_t)rd << 7) | 0x03u;
}

static uint32_t iree_rv32_encode_jal(int rd, int32_t rel) {
  uint32_t urel = (uint32_t)rel;
  uint32_t imm20 = (urel >> 20) & 0x1u;
  uint32_t imm10_1 = (urel >> 1) & 0x3ffu;
  uint32_t imm11 = (urel >> 11) & 0x1u;
  uint32_t imm19_12 = (urel >> 12) & 0xffu;
  return (imm20 << 31) | (imm19_12 << 12) | (imm11 << 20) | (imm10_1 << 21) |
         ((uint32_t)rd << 7) | 0x6fu;
}

static int32_t iree_rv32_pcrel_lo12(uint32_t insn_pc, uint32_t literal_pc) {
  return (int32_t)literal_pc - (int32_t)insn_pc;
}

static iree_status_t iree_hal_coralnpu_simulator_install_trampoline(
    uint32_t entry_pc, uint32_t dispatch_state_addr, uint32_t *out_start_pc) {
  IREE_ASSERT_ARGUMENT(out_start_pc);

  const uint32_t tramp_pc = 0x00001000u;
  const uint32_t stack_top_addr = 0x00018000u;

  enum {
    W_AUIPC_RA = 0,
    W_ADDI_RA,
    W_AUIPC_SP,
    W_LW_SP,
    W_AUIPC_A1,
    W_LW_A1,
    W_JAL_ENTRY,
    W_HALT,
    W_LIT_SP,
    W_LIT_A1,
    W_COUNT
  };

  const uint32_t halt_pc = tramp_pc + W_HALT * 4u;
  const uint32_t tramp_size = W_COUNT * 4u;

  if (tramp_pc < coralnpu_itcm_start ||
      tramp_pc + tramp_size > coralnpu_itcm_start + coralnpu_itcm_size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "trampoline does not fit in ITCM");
  }

  const int32_t rel =
      (int32_t)entry_pc - (int32_t)(tramp_pc + W_JAL_ENTRY * 4u);
  if ((rel & 0x1) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "entry PC is not 2-byte aligned");
  }
  if (rel < -(1 << 20) || rel >= (1 << 20)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "entry PC too far for trampoline jal");
  }

  uint32_t tramp[W_COUNT];
  memset(tramp, 0, sizeof(tramp));

  // ra = halt_pc
  tramp[W_AUIPC_RA] = 0x00000097u; // auipc ra, 0
  tramp[W_ADDI_RA] = iree_rv32_encode_addi(
      /*rd=*/1, /*rs1=*/1, (int32_t)(halt_pc - (tramp_pc + W_AUIPC_RA * 4u)));

  // sp = literal(stack_top_addr)
  tramp[W_AUIPC_SP] = iree_rv32_encode_auipc(/*rd=*/2, 0);
  tramp[W_LW_SP] =
      iree_rv32_encode_lw(/*rd=*/2, /*rs1=*/2,
                          iree_rv32_pcrel_lo12(tramp_pc + W_AUIPC_SP * 4u,
                                               tramp_pc + W_LIT_SP * 4u));

  // a1 = literal(dispatch_state_addr)
  tramp[W_AUIPC_A1] = iree_rv32_encode_auipc(/*rd=*/11, 0);
  tramp[W_LW_A1] =
      iree_rv32_encode_lw(/*rd=*/11, /*rs1=*/11,
                          iree_rv32_pcrel_lo12(tramp_pc + W_AUIPC_A1 * 4u,
                                               tramp_pc + W_LIT_A1 * 4u));

  // jump to entry_pc
  tramp[W_JAL_ENTRY] = iree_rv32_encode_jal(/*rd=*/0, rel);

  // simulator halt instruction
  tramp[W_HALT] = 0x08000073u;

  // literal pool
  tramp[W_LIT_SP] = stack_top_addr;
  tramp[W_LIT_A1] = dispatch_state_addr;

  simulator_load_itcm(tramp_pc - coralnpu_itcm_start, tramp, sizeof(tramp));
  *out_start_pc = tramp_pc;

  fprintf(stderr,
          "[SIM DEBUG] trampoline installed: tramp_pc=0x%08x "
          "entry_pc=0x%08x halt_pc=0x%08x jal_rel=%d\n",
          tramp_pc, entry_pc, halt_pc, rel);
  fprintf(stderr, "[SIM DEBUG] ABI init: sp=0x%08x a1=0x%08x\n", stack_top_addr,
          dispatch_state_addr);
  fflush(stderr);

  return iree_ok_status();
}

static uint32_t iree_align_u32(uint32_t value, uint32_t alignment) {
  return (value + alignment - 1u) & ~(alignment - 1u);
}

static void iree_hal_coralnpu_simulator_write_dtcm_u32(uint32_t addr,
                                                       uint32_t value) {
  uint8_t bytes[4];
  bytes[0] = (uint8_t)(value & 0xFFu);
  bytes[1] = (uint8_t)((value >> 8) & 0xFFu);
  bytes[2] = (uint8_t)((value >> 16) & 0xFFu);
  bytes[3] = (uint8_t)((value >> 24) & 0xFFu);
  simulator_load_dtcm(addr, bytes, sizeof(bytes));
}

iree_status_t iree_hal_simulator_issue_dispatch_inline(
    iree_const_byte_span_t dispatch_image,
    const iree_hal_executable_dispatch_state_v0_t *dispatch_state,
    iree_host_size_t ordinal, iree_byte_span_t local_memory) {
  IREE_ASSERT_ARGUMENT(dispatch_state);
  (void)ordinal;

  if (local_memory.data_length != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "simulator runtime patch does not support local_memory");
  }

  if (dispatch_state->constant_count != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "simulator runtime patch does not support push constants yet");
  }

  if (dispatch_state->binding_count != 0 &&
      (!dispatch_state->binding_ptrs || !dispatch_state->binding_lengths)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dispatch bindings are not initialized");
  }

  if (dispatch_state->binding_count > 64) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "too many bindings for temporary array");
  }

  // Load the ELF and get the real ELF entry PC.
  uint32_t entry_pc = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_coralnpu_simulator_load_elf(dispatch_image, &entry_pc));

  // Pack runtime bindings into DTCM in binding order.
  uint32_t binding_dtcm_offsets[64];
  uint32_t dtcm_offset = 0;
  for (uint32_t i = 0; i < dispatch_state->binding_count; ++i) {
    void *binding_ptr = dispatch_state->binding_ptrs[i];
    size_t binding_length = dispatch_state->binding_lengths[i];
    binding_dtcm_offsets[i] = dtcm_offset;

    if (binding_length == 0)
      continue;
    if (!binding_ptr) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "binding pointer is null");
    }
    if (dtcm_offset + binding_length > coralnpu_dtcm_size) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "bindings exceed DTCM capacity");
    }

    simulator_load_dtcm(dtcm_offset, binding_ptr, binding_length);
    dtcm_offset += (uint32_t)binding_length;
  }

  // Build minimal simulator-visible dispatch ABI in DTCM.
  // Layout:
  //   bindings_table_dtcm: binding_count * 4 bytes
  //   dispatch_state_dtcm: at least 32 bytes, with offset 28 used
  dtcm_offset = iree_align_u32(dtcm_offset, 16);

  const uint32_t bindings_table_dtcm = dtcm_offset;
  const uint32_t bindings_table_size = dispatch_state->binding_count * 4u;
  dtcm_offset += bindings_table_size;

  dtcm_offset = iree_align_u32(dtcm_offset, 16);

  const uint32_t dispatch_state_dtcm = dtcm_offset;
  const uint32_t dispatch_state_size = 32u; // enough for byte offset 28
  dtcm_offset += dispatch_state_size;

  if (dtcm_offset > coralnpu_dtcm_size) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "dispatch ABI exceeds DTCM capacity");
  }

  // Zero the dispatch_state area.
  for (uint32_t off = 0; off < dispatch_state_size; off += 4) {
    iree_hal_coralnpu_simulator_write_dtcm_u32(dispatch_state_dtcm + off, 0);
  }

  // bindings_table[i] = absolute simulator address of binding i
  for (uint32_t i = 0; i < dispatch_state->binding_count; ++i) {
    iree_hal_coralnpu_simulator_write_dtcm_u32(bindings_table_dtcm + i * 4u,
                                               coralnpu_dtcm_start +
                                                   binding_dtcm_offsets[i]);
  }

  // dispatch_state[7] = absolute simulator address of bindings table
  // (byte offset 28)
  iree_hal_coralnpu_simulator_write_dtcm_u32(
      dispatch_state_dtcm + 28u, coralnpu_dtcm_start + bindings_table_dtcm);

  fprintf(
      stderr,
      "[SIM DEBUG] ABI: dispatch_state_dtcm=0x%08x dispatch_state_abs=0x%08x "
      "bindings_table_dtcm=0x%08x bindings_table_abs=0x%08x\n",
      dispatch_state_dtcm, coralnpu_dtcm_start + dispatch_state_dtcm,
      bindings_table_dtcm, coralnpu_dtcm_start + bindings_table_dtcm);
  for (uint32_t i = 0; i < dispatch_state->binding_count; ++i) {
    fprintf(stderr,
            "[SIM DEBUG] ABI: binding[%u] -> dtcm_off=0x%08x abs=0x%08x\n", i,
            binding_dtcm_offsets[i],
            coralnpu_dtcm_start + binding_dtcm_offsets[i]);
  }
  fflush(stderr);

  // Install generic trampoline: only sets ra/sp/a1 and jumps to entry.
  uint32_t start_pc = 0;
  IREE_RETURN_IF_ERROR(iree_hal_coralnpu_simulator_install_trampoline(
      entry_pc, coralnpu_dtcm_start + dispatch_state_dtcm, &start_pc));

  fprintf(
      stderr,
      "[SIM DEBUG] ELF loaded, entry_pc=0x%08x start_pc=0x%08x bindings=%u\n",
      entry_pc, start_pc, dispatch_state->binding_count);
  fflush(stderr);

  // Optional debug dump before execution.
  for (uint32_t i = 0; i < dispatch_state->binding_count; ++i) {
    size_t binding_length = dispatch_state->binding_lengths[i];
    if (binding_length == 0)
      continue;

    int32_t temp[32];
    memset(temp, 0, sizeof(temp));
    simulator_read_dtcm(binding_dtcm_offsets[i], temp, binding_length);
  }

  simulator_run(start_pc);

  // Read results back from DTCM to the bound host buffers.
  for (uint32_t i = 0; i < dispatch_state->binding_count; ++i) {
    void *binding_ptr = dispatch_state->binding_ptrs[i];
    size_t binding_length = dispatch_state->binding_lengths[i];
    if (binding_length == 0)
      continue;

    simulator_read_dtcm(binding_dtcm_offsets[i], binding_ptr, binding_length);
  }

  return iree_ok_status();
}
