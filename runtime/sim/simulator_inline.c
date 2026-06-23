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

#include <inttypes.h>
#include <limits.h>
#include <string.h>

#ifdef IREE_CORALNPU_SIMULATOR_DEBUG
#include <stdio.h>
#endif

#include "crt/coralnpu_dispatch.h"
#include "iree/base/api.h"
#include "runtime/sim/simulator_api.h"
#include "runtime/sim/simulator_elf_loader.h"

static iree_status_t iree_hal_coralnpu_allocate_dtcm(uint32_t* cursor,
                                                     uint32_t heap_end,
                                                     uint32_t alignment,
                                                     size_t size,
                                                     uint32_t* out_address) {
  uint64_t aligned =
      ((uint64_t)*cursor + alignment - 1u) & ~((uint64_t)alignment - 1u);
  uint64_t allocation_end = aligned + size;

  if (allocation_end < aligned || allocation_end > heap_end) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "dispatch data exceeds the firmware heap");
  }

  *out_address = (uint32_t)aligned;
  *cursor = (uint32_t)allocation_end;
  return iree_ok_status();
}

static void iree_hal_coralnpu_write_dtcm_u32(uint32_t address, uint32_t value) {
  uint8_t bytes[4] = {
      (uint8_t)(value & 0xFFu),
      (uint8_t)((value >> 8) & 0xFFu),
      (uint8_t)((value >> 16) & 0xFFu),
      (uint8_t)((value >> 24) & 0xFFu),
  };

  simulator_load_dtcm(address - coralnpu_dtcm_start, bytes, sizeof(bytes));
}

static void iree_hal_coralnpu_zero_dtcm(uint32_t address, size_t size) {
  uint8_t zeros[256] = {0};

  while (size != 0) {
    size_t chunk = size < sizeof(zeros) ? size : sizeof(zeros);

    simulator_load_dtcm(address - coralnpu_dtcm_start, zeros, chunk);

    address += (uint32_t)chunk;
    size -= chunk;
  }
}

iree_status_t iree_hal_simulator_issue_dispatch_inline(
    iree_const_byte_span_t dispatch_image,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    iree_host_size_t ordinal, iree_byte_span_t local_memory) {
  IREE_ASSERT_ARGUMENT(dispatch_state);

#ifdef IREE_CORALNPU_SIMULATOR_DEBUG
  fprintf(stderr, "[CoralNPU simulator] dispatch entered\n");
  fflush(stderr);
#endif

  if (ordinal > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "dispatch ordinal is too large");
  }

  if (dispatch_state->constant_count != 0 &&
      dispatch_state->constants == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "push constants are not initialized");
  }

  if (dispatch_state->binding_count != 0 &&
      (dispatch_state->binding_ptrs == NULL ||
       dispatch_state->binding_lengths == NULL)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dispatch bindings are not initialized");
  }

  iree_hal_coralnpu_simulator_elf_layout_t elf_layout;

  IREE_RETURN_IF_ERROR(iree_hal_coralnpu_simulator_load_elf_with_layout(
      dispatch_image, &elf_layout));

  coralnpu_dispatch_request_t request;
  memset(&request, 0, sizeof(request));

#ifdef IREE_CORALNPU_SIMULATOR_DEBUG
  fprintf(stderr,
          "[CoralNPU simulator] request size: ELF=%" PRIu32
          " runtime=%zu, address=0x%08" PRIx32 "\n",
          elf_layout.dispatch_request_size, sizeof(request),
          elf_layout.dispatch_request_addr);
  fflush(stderr);
#endif

  if (elf_layout.dispatch_request_size != sizeof(request)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "CoralNPU dispatch ABI size mismatch: ELF=%" PRIu32
                            " runtime=%zu",
                            elf_layout.dispatch_request_size, sizeof(request));
  }

  request.magic = CORALNPU_DISPATCH_MAGIC;
  request.version = CORALNPU_DISPATCH_VERSION;
  request.status = CORALNPU_DISPATCH_STATUS_READY;
  request.return_code = 0;

  request.workgroup_size_x = dispatch_state->workgroup_size_x;
  request.workgroup_size_y = dispatch_state->workgroup_size_y;
  request.workgroup_size_z = dispatch_state->workgroup_size_z;

  request.workgroup_count_x = dispatch_state->workgroup_count_x;
  request.workgroup_count_y = dispatch_state->workgroup_count_y;
  request.workgroup_count_z = dispatch_state->workgroup_count_z;

  request.max_concurrency = dispatch_state->max_concurrency;
  request.push_constant_count = dispatch_state->constant_count;
  request.binding_count = dispatch_state->binding_count;
  request.ordinal = (uint32_t)ordinal;

  uint32_t heap_cursor = elf_layout.heap_start_addr;

  if (dispatch_state->constant_count != 0) {
    IREE_RETURN_IF_ERROR(iree_hal_coralnpu_allocate_dtcm(
        &heap_cursor, elf_layout.heap_end_addr, 4,
        (size_t)dispatch_state->constant_count * sizeof(uint32_t),
        &request.push_constants_addr));

    for (uint32_t i = 0; i < dispatch_state->constant_count; ++i) {
      iree_hal_coralnpu_write_dtcm_u32(
          request.push_constants_addr + i * sizeof(uint32_t),
          dispatch_state->constants[i]);
    }
  }

  if (dispatch_state->binding_count != 0) {
    IREE_RETURN_IF_ERROR(iree_hal_coralnpu_allocate_dtcm(
        &heap_cursor, elf_layout.heap_end_addr, 4,
        (size_t)dispatch_state->binding_count * sizeof(uint32_t),
        &request.binding_ptrs_addr));

    IREE_RETURN_IF_ERROR(iree_hal_coralnpu_allocate_dtcm(
        &heap_cursor, elf_layout.heap_end_addr, 4,
        (size_t)dispatch_state->binding_count * sizeof(uint32_t),
        &request.binding_lengths_addr));
  }

  if (local_memory.data_length != 0) {
    IREE_RETURN_IF_ERROR(iree_hal_coralnpu_allocate_dtcm(
        &heap_cursor, elf_layout.heap_end_addr, 64, local_memory.data_length,
        &request.local_memory_addr));

    iree_hal_coralnpu_zero_dtcm(request.local_memory_addr,
                                local_memory.data_length);
  }

  const uint32_t binding_data_start = heap_cursor;
  uint32_t binding_cursor = binding_data_start;

  for (uint32_t i = 0; i < dispatch_state->binding_count; ++i) {
    void* binding_ptr = dispatch_state->binding_ptrs[i];
    size_t binding_length = dispatch_state->binding_lengths[i];

    if (binding_length > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "binding %u is too large", i);
    }

    if (binding_length != 0 && binding_ptr == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "binding %u has a null pointer", i);
    }

    uint32_t binding_address = 0;

    IREE_RETURN_IF_ERROR(iree_hal_coralnpu_allocate_dtcm(
        &binding_cursor, elf_layout.heap_end_addr, 64, binding_length,
        &binding_address));

    iree_hal_coralnpu_write_dtcm_u32(
        request.binding_ptrs_addr + i * sizeof(uint32_t), binding_address);

    iree_hal_coralnpu_write_dtcm_u32(
        request.binding_lengths_addr + i * sizeof(uint32_t),
        (uint32_t)binding_length);

    if (binding_length != 0) {
      simulator_load_dtcm(binding_address - coralnpu_dtcm_start, binding_ptr,
                          binding_length);
    }
  }

  uint32_t dispatch_request_dtcm_offset =
      elf_layout.dispatch_request_addr - coralnpu_dtcm_start;

  simulator_load_dtcm(dispatch_request_dtcm_offset, &request, sizeof(request));

#ifdef IREE_CORALNPU_SIMULATOR_DEBUG
  fprintf(stderr,
          "[CoralNPU simulator] running: "
          "entry_pc=0x%08" PRIx32 ", ordinal=%zu, bindings=%u\n",
          elf_layout.start_pc, (size_t)ordinal,
          (unsigned)dispatch_state->binding_count);
  fflush(stderr);
#endif

  simulator_run(elf_layout.start_pc);

#ifdef IREE_CORALNPU_SIMULATOR_DEBUG
  fprintf(stderr, "[CoralNPU simulator] execution returned\n");
  fflush(stderr);
#endif

  simulator_read_dtcm(dispatch_request_dtcm_offset, &request, sizeof(request));

  if (request.magic != CORALNPU_DISPATCH_MAGIC ||
      request.version != CORALNPU_DISPATCH_VERSION) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "firmware corrupted the CoralNPU dispatch request");
  }

  if (request.status != CORALNPU_DISPATCH_STATUS_COMPLETE) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "firmware dispatch did not complete: status=%u return_code=%" PRId32,
        (unsigned)request.status, request.return_code);
  }

  if (request.return_code != 0) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "firmware dispatch failed with return code %" PRId32,
        request.return_code);
  }

  binding_cursor = binding_data_start;

  for (uint32_t i = 0; i < dispatch_state->binding_count; ++i) {
    void* binding_ptr = dispatch_state->binding_ptrs[i];
    size_t binding_length = dispatch_state->binding_lengths[i];
    uint32_t binding_address = 0;

    IREE_RETURN_IF_ERROR(iree_hal_coralnpu_allocate_dtcm(
        &binding_cursor, elf_layout.heap_end_addr, 64, binding_length,
        &binding_address));

    if (binding_length != 0) {
      simulator_read_dtcm(binding_address - coralnpu_dtcm_start, binding_ptr,
                          binding_length);
    }
  }

  return iree_ok_status();
}
