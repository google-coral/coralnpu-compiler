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

#include "crt/coralnpu_dispatch.h"
#include "crt/coralnpu_dma.h"
#include "iree/base/api.h"
#include "runtime/sim/simulator_api.h"
#include "runtime/sim/simulator_elf_loader.h"
#include "runtime/sim/simulator_stats.h"

// Bump-allocates |size| bytes aligned to |alignment| below |limit|.
static iree_status_t iree_hal_coralnpu_allocate(uint32_t* cursor,
                                                uint64_t limit,
                                                uint32_t alignment, size_t size,
                                                const char* region,
                                                uint32_t* out_address) {
  uint64_t aligned =
      ((uint64_t)*cursor + alignment - 1u) & ~((uint64_t)alignment - 1u);
  uint64_t allocation_end = aligned + size;

  if (allocation_end < aligned || allocation_end > limit) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "dispatch data exceeds the %s", region);
  }

  *out_address = (uint32_t)aligned;
  *cursor = (uint32_t)allocation_end;
  return iree_ok_status();
}

// Simulated 1 GiB DDR window [0x80000000, 0xC0000000): writeable staging grows
// up from IREE_HAL_CORALNPU_DDR_START (above the guest DMA trace when enabled);
// cached read-only bindings grow down from IREE_HAL_CORALNPU_DDR_END.
#define IREE_HAL_CORALNPU_DDR_BASE 0x80000000u
#define IREE_HAL_CORALNPU_DDR_END 0xC0000000u
#define IREE_HAL_CORALNPU_STAGING_RESERVE (128u * 1024 * 1024)

#if CORALNPU_DMA_TRACE_CAPACITY > 0
// Computed in 64-bit so large ADDR/CAPACITY values cannot wrap 32-bit before
// the static_assert below.
#define IREE_HAL_CORALNPU_DMA_TRACE_END                   \
  ((uint64_t)CORALNPU_DMA_TRACE_ADDR + sizeof(uint32_t) + \
   (uint64_t)CORALNPU_DMA_TRACE_CAPACITY * sizeof(coralnpu_dma_trace_t))
_Static_assert(CORALNPU_DMA_TRACE_ADDR >= IREE_HAL_CORALNPU_DDR_BASE &&
                   ((IREE_HAL_CORALNPU_DMA_TRACE_END + 63u) & ~63ull) <=
                       IREE_HAL_CORALNPU_DDR_END -
                           IREE_HAL_CORALNPU_STAGING_RESERVE,
               "DMA trace region must stay in DDR below the staging reserve");
#define IREE_HAL_CORALNPU_DDR_START \
  ((uint32_t)((IREE_HAL_CORALNPU_DMA_TRACE_END + 63u) & ~63ull))
#else
#define IREE_HAL_CORALNPU_DDR_START IREE_HAL_CORALNPU_DDR_BASE
#endif  // CORALNPU_DMA_TRACE_CAPACITY > 0

// Guarded by |iree_hal_coralnpu_simulator_mutex| in simulator_backend.c.
static coralnpu_simulator_t* iree_hal_coralnpu_ro_cache_sim = NULL;

void iree_hal_simulator_inline_forget(coralnpu_simulator_t* sim) {
  if (iree_hal_coralnpu_ro_cache_sim == sim) {
    iree_hal_coralnpu_ro_cache_sim = NULL;
  }
}

// FNV-1a over <=4 KiB buffers or 256 evenly spaced 8B samples of larger ones,
// so sparse changes to a large buffer at the same address go undetected.
static uint64_t iree_hal_coralnpu_sample_hash(const uint8_t* bytes,
                                              size_t length) {
  uint64_t h = 14695981039346656037ull ^ length, w = 0;
  if (length <= 4096) {
    size_t i = 0;
    for (; i + 8 <= length; i += 8) {
      memcpy(&w, bytes + i, 8);
      h = (h ^ w) * 1099511628211ull;
    }
    for (; i < length; ++i) h = (h ^ bytes[i]) * 1099511628211ull;
    return h;
  }
  for (size_t k = 0; k < 256; ++k) {
    memcpy(&w, bytes + (length - 8) * k / 255, 8);
    h = (h ^ w) * 1099511628211ull;
  }
  return h;
}

iree_status_t iree_hal_simulator_issue_dispatch_inline(
    coralnpu_simulator_t* sim, iree_const_byte_span_t dispatch_image,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const bool* binding_writeable, iree_host_size_t ordinal,
    iree_host_size_t local_memory_size,
    iree_hal_coralnpu_dispatch_stats_t* out_stats) {
  IREE_ASSERT_ARGUMENT(dispatch_state);
  IREE_ASSERT_LE(dispatch_state->binding_count,
                 IREE_HAL_EXECUTABLE_MAX_BINDING_COUNT);

  iree_hal_coralnpu_simulator_elf_layout_t elf_layout;

  IREE_RETURN_IF_ERROR(iree_hal_coralnpu_simulator_load_elf_with_layout(
      sim, dispatch_image, &elf_layout));

  iree_hal_coralnpu_stats_collector_t collector;
  iree_hal_coralnpu_stats_collector_initialize(sim, out_stats, &collector);

  coralnpu_dispatch_request_t request = {0};

  if (elf_layout.dispatch_request_size != sizeof(request)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "CoralNPU dispatch ABI size mismatch: ELF=%" PRIu32
                            " runtime=%zu",
                            elf_layout.dispatch_request_size, sizeof(request));
  }

  request.magic = CORALNPU_DISPATCH_MAGIC;
  request.version = CORALNPU_DISPATCH_VERSION;
  request.status = CORALNPU_DISPATCH_STATUS_READY;

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

  const size_t constants_size =
      (size_t)dispatch_state->constant_count * sizeof(uint32_t);
  if (dispatch_state->constant_count != 0) {
    IREE_RETURN_IF_ERROR(iree_hal_coralnpu_allocate(
        &heap_cursor, elf_layout.heap_end_addr, 4, constants_size,
        "firmware heap", &request.push_constants_addr));
    coralnpu_simulator_write_mem(sim, request.push_constants_addr,
                                 dispatch_state->constants, constants_size);
  }

  const size_t binding_table_size =
      (size_t)dispatch_state->binding_count * sizeof(uint32_t);
  if (dispatch_state->binding_count != 0) {
    IREE_RETURN_IF_ERROR(iree_hal_coralnpu_allocate(
        &heap_cursor, elf_layout.heap_end_addr, 4, binding_table_size,
        "firmware heap", &request.binding_ptrs_addr));

    IREE_RETURN_IF_ERROR(iree_hal_coralnpu_allocate(
        &heap_cursor, elf_layout.heap_end_addr, 4, binding_table_size,
        "firmware heap", &request.binding_lengths_addr));
  }

  // Without local memory, the address and the size in the request stay 0.
  if (local_memory_size != 0) {
    IREE_RETURN_IF_ERROR(iree_hal_coralnpu_allocate(
        &heap_cursor, elf_layout.heap_end_addr, 64, local_memory_size,
        "firmware heap", &request.local_memory_addr));
    // The local memory fits in the firmware heap, so its size fits in 32 bits.
    request.local_memory_size = (uint32_t)local_memory_size;

    iree_hal_coralnpu_simulator_zero_mem(sim, request.local_memory_addr,
                                         local_memory_size);
  }

#if CORALNPU_DMA_TRACE_CAPACITY > 0
  // DDR outlives the dispatch, so reset the trace count.
  iree_hal_coralnpu_simulator_zero_mem(sim, CORALNPU_DMA_TRACE_ADDR,
                                       sizeof(uint32_t));
#endif  // CORALNPU_DMA_TRACE_CAPACITY > 0

  // Cache read-only bindings in simulated DDR across dispatches on |sim|.
  // Guarded by |iree_hal_coralnpu_simulator_mutex| in simulator_backend.c.
  enum { kRoCacheCapacity = 2048 };
  static struct {
    const void* ptr;
    size_t length;
    uint32_t addr;
    uint64_t hash;
  } ro_cache[kRoCacheCapacity];
  static uint32_t ro_count = 0;
  // Lowest address held by the cache.
  static uint32_t ro_floor = IREE_HAL_CORALNPU_DDR_END;
  if (sim != iree_hal_coralnpu_ro_cache_sim) {
    iree_hal_coralnpu_ro_cache_sim = sim;
    ro_count = 0;
    ro_floor = IREE_HAL_CORALNPU_DDR_END;
  }

  uint32_t ddr_cursor = IREE_HAL_CORALNPU_DDR_START;
  uint32_t binding_addresses[IREE_HAL_EXECUTABLE_MAX_BINDING_COUNT] = {0};
  uint32_t binding_sizes[IREE_HAL_EXECUTABLE_MAX_BINDING_COUNT] = {0};
  bool use_ro_cache = true;

  for (uint32_t i = 0; i < dispatch_state->binding_count; ++i) {
    void* binding_ptr = dispatch_state->binding_ptrs[i];
    size_t binding_length = dispatch_state->binding_lengths[i];

    uint32_t binding_address = 0;
    for (uint32_t j = 0; j < i; ++j) {
      // Reuse a slot only if it covers this binding.
      if (dispatch_state->binding_ptrs[j] == binding_ptr &&
          dispatch_state->binding_lengths[j] >= binding_length) {
        binding_address = binding_addresses[j];
        break;
      }
    }

    if (binding_address == 0 && use_ro_cache && !binding_writeable[i] &&
        binding_length > 0) {
      uint64_t h = iree_hal_coralnpu_sample_hash(binding_ptr, binding_length);
      for (uint32_t c = 0; c < ro_count; ++c) {
        if (ro_cache[c].ptr == binding_ptr &&
            ro_cache[c].length == binding_length) {
          if (ro_cache[c].hash != h) {
            coralnpu_simulator_write_mem(sim, ro_cache[c].addr, binding_ptr,
                                         binding_length);
            ro_cache[c].hash = h;
          }
          binding_address = ro_cache[c].addr;
          break;
        }
      }
      uint64_t floor = ((uint64_t)ro_floor - binding_length) & ~63ull;
      uint64_t staging_end = (uint64_t)IREE_HAL_CORALNPU_DDR_START +
                             IREE_HAL_CORALNPU_STAGING_RESERVE;
      if (ddr_cursor > staging_end) staging_end = ddr_cursor;
      if (binding_address == 0 && ro_count < kRoCacheCapacity &&
          binding_length < ro_floor && floor >= staging_end) {
        ro_floor = binding_address = (uint32_t)floor;
        coralnpu_simulator_write_mem(sim, binding_address, binding_ptr,
                                     binding_length);
        ro_cache[ro_count].ptr = binding_ptr;
        ro_cache[ro_count].length = binding_length;
        ro_cache[ro_count].addr = binding_address;
        ro_cache[ro_count].hash = h;
        ++ro_count;
      }
    }

    if (binding_address == 0) {
      iree_status_t status =
          iree_hal_coralnpu_allocate(&ddr_cursor, ro_floor, 64, binding_length,
                                     "DDR region", &binding_address);
      if (!iree_status_is_ok(status) && ro_count != 0) {
        // Cached bindings leave too little room for this dispatch: empty the
        // cache and restage the dispatch in the whole window without it.
        iree_status_ignore(status);
        ro_count = 0;
        ro_floor = IREE_HAL_CORALNPU_DDR_END;
        use_ro_cache = false;
        ddr_cursor = IREE_HAL_CORALNPU_DDR_START;
        i = UINT32_MAX;  // Restart from binding 0.
        continue;
      }
      IREE_RETURN_IF_ERROR(status);
      if (binding_length != 0) {
        coralnpu_simulator_write_mem(sim, binding_address, binding_ptr,
                                     binding_length);
      }
    }

    binding_addresses[i] = binding_address;
    binding_sizes[i] = (uint32_t)binding_length;
  }

  if (dispatch_state->binding_count != 0) {
    coralnpu_simulator_write_mem(sim, request.binding_ptrs_addr,
                                 binding_addresses, binding_table_size);
    coralnpu_simulator_write_mem(sim, request.binding_lengths_addr,
                                 binding_sizes, binding_table_size);
  }

  coralnpu_simulator_write_mem(sim, elf_layout.dispatch_request_addr, &request,
                               sizeof(request));

  iree_hal_coralnpu_stats_collector_begin_run(&collector);
  const bool halted = coralnpu_simulator_run(sim, elf_layout.start_pc);
  iree_hal_coralnpu_stats_collector_end_run(&collector);
  if (!halted) {
    return iree_make_status(IREE_STATUS_INTERNAL, "CoralNPU core did not halt");
  }

  coralnpu_simulator_read_mem(sim, elf_layout.dispatch_request_addr, &request,
                              sizeof(request));

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

  for (uint32_t i = 0; i < dispatch_state->binding_count; ++i) {
    if (!binding_writeable[i] || binding_sizes[i] == 0) {
      continue;
    }
    // Skip if an earlier writeable binding already read back this range.
    bool already_read = false;
    for (uint32_t j = 0; j < i; ++j) {
      if (binding_writeable[j] && binding_sizes[j] >= binding_sizes[i] &&
          binding_addresses[j] == binding_addresses[i]) {
        already_read = true;
        break;
      }
    }
    if (!already_read) {
      coralnpu_simulator_read_mem(sim, binding_addresses[i],
                                  dispatch_state->binding_ptrs[i],
                                  binding_sizes[i]);
    }
  }

  return iree_ok_status();
}
