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

// Tests the stack peak measurement on a fake core whose memory is an array.

#include "metrics/runtime/stack_peak.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

// The memory of the fake core, at addresses [FAKE_BASE, FAKE_BASE + size).
#define FAKE_BASE 0x10000u
static uint8_t fake_memory[0x3000];

// Reads the memory of the fake core.
static void fake_read_mem(void *self, uint32_t address, void *data,
                          size_t size) {
  (void)self;
  memcpy(data, &fake_memory[address - FAKE_BASE], size);
}

// Writes the memory of the fake core.
static void fake_write_mem(void *self, uint32_t address, const void *data,
                           size_t size) {
  (void)self;
  memcpy(&fake_memory[address - FAKE_BASE], data, size);
}

// The fake core has no cycle counter.
static uint64_t fake_cycle_count(void *self) {
  (void)self;
  return 0;
}

static const iree_hal_coralnpu_core_access_t fake_core = {
    .self = NULL,
    .cycle_count = fake_cycle_count,
    .read_mem = fake_read_mem,
    .write_mem = fake_write_mem,
};

// A test case: the stack follows the heap unless |gap| is set, and the
// firmware writes |stack_use| bytes below the end of the stack.
typedef struct test_case_t {
  const char *name;
  bool gap;
  // Bytes of dispatch data at the start of the heap.
  uint32_t heap_used;
  uint32_t stack_use;
  uint32_t expected_peak;
  bool expected_overflow;
} test_case_t;

// Heap [0x10000, 0x11000) and stack [0x11000, 0x12000); with a gap, the stack
// starts 0x100 bytes after the end of the heap.
static iree_hal_coralnpu_memory_stats_t make_memory(bool gap) {
  iree_hal_coralnpu_memory_stats_t memory;
  memset(&memory, 0, sizeof(memory));
  memory.heap.start = FAKE_BASE;
  memory.heap.size = 0x1000;
  memory.stack.start = FAKE_BASE + 0x1000 + (gap ? 0x100 : 0);
  memory.stack.size = 0x1000;
  return memory;
}

// Runs |test| and returns whether it passed.
static bool run_test(const test_case_t *test) {
  // The firmware writes zeros, which differ from the paint.
  memset(fake_memory, 0, sizeof(fake_memory));
  const iree_hal_coralnpu_memory_stats_t memory = make_memory(test->gap);
  iree_hal_coralnpu_stack_paint_t paint;
  iree_hal_coralnpu_stack_paint(&fake_core, &memory,
                                memory.heap.start + test->heap_used, &paint);
  const uint32_t stack_end = memory.stack.start + memory.stack.size;
  memset(&fake_memory[stack_end - test->stack_use - FAKE_BASE], 0,
         test->stack_use);
  uint32_t peak = 0;
  bool overflow = false;
  iree_hal_coralnpu_stack_scan(&fake_core, &paint, &peak, &overflow);
  if (peak == test->expected_peak && overflow == test->expected_overflow) {
    return true;
  }
  fprintf(stderr,
          "%s: peak=%" PRIu32 " overflow=%d, expected peak=%" PRIu32
          " overflow=%d\n",
          test->name, peak, overflow, test->expected_peak,
          test->expected_overflow);
  return false;
}

int main(void) {
  static const test_case_t tests[] = {
      {"unused", false, 0, 0, 0, false},
      {"half", false, 0, 0x800, 0x800, false},
      {"unaligned use", false, 0, 0x802, 0x804, false},
      {"full with guard", false, 0, 0x1000, 0x1000, false},
      {"overflow into guard", false, 0, 0x1080, 0x1080, true},
      // The guard is the free heap, so a full heap leaves no guard and a
      // fully used stack may have overflowed.
      {"full heap", false, 0x1000, 0x1000, 0x1000, true},
      {"guard limited by heap", false, 0x1000 - 8, 0x1004, 0x1004, true},
      // Only free heap right below the stack can be the guard area.
      {"gap", true, 0, 0x800, 0x800, false},
      {"full with gap", true, 0, 0x1000, 0x1000, true},
  };
  int failures = 0;
  for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); ++i) {
    if (!run_test(&tests[i])) ++failures;
  }
  if (failures) {
    fprintf(stderr, "%d stack peak tests failed\n", failures);
    return 1;
  }
  printf("All stack peak tests passed\n");
  return 0;
}
