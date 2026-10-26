// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Runs dma_dispatch_test through iree_hal_simulator_issue_dispatch_inline with
// a 16 KB writeable binding, more than the 12 KB below the DMA trace, and
// checks the output and the trace. Against the old layout the copy overwrites
// the trace count, so the trace check is what catches a collision. Runs twice
// on one simulator to check the trace reset.
//
// Usage:
//   dma_dispatch_runner <elf>

#include "crt/coralnpu_dma.h"
#include "runtime/sim/simulator_api.h"
#include "runtime/sim/simulator_inline.h"

// IREE:
#include "iree/base/api.h"
#include "iree/hal/local/executable_library.h"

// Standard library:
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

// Each binding alone overruns the 12 KB below the trace.
constexpr size_t kBindingSize = 16 * 1024;
constexpr int kRuns = 2;

// Reads a file into memory, or exits on error.
std::vector<uint8_t> ReadFile(const char* path) {
  FILE* f = std::fopen(path, "rb");
  if (f == nullptr) {
    std::perror(path);
    std::exit(2);
  }
  if (std::fseek(f, 0, SEEK_END) != 0) {
    std::perror(path);
    std::exit(2);
  }
  long n = std::ftell(f);
  if (n < 0 || std::fseek(f, 0, SEEK_SET) != 0) {
    std::perror(path);
    std::exit(2);
  }
  std::vector<uint8_t> buf(static_cast<size_t>(n));
  if (std::fread(buf.data(), 1, buf.size(), f) != buf.size()) {
    std::fprintf(stderr, "short read: %s\n", path);
    std::exit(2);
  }
  std::fclose(f);
  return buf;
}

// Returns true if the trace holds one START and one DONE of |bytes| bytes at
// seq=1.
bool CheckTrace(coralnpu_simulator_t* sim, uint32_t bytes) {
  uint32_t count = 0;
  coralnpu_simulator_read_mem(sim, CORALNPU_DMA_TRACE_ADDR, &count,
                              sizeof(count));
  if (count != 2) {
    std::fprintf(stderr, "FAIL: trace count is %u, expected 2\n", count);
    return false;
  }
  coralnpu_dma_trace_t records[2];
  coralnpu_simulator_read_mem(sim, CORALNPU_DMA_TRACE_ADDR + sizeof(count),
                              records, sizeof(records));
  const uint32_t phases[2] = {CORALNPU_DMA_EVENT_START,
                              CORALNPU_DMA_EVENT_DONE};
  for (int i = 0; i < 2; ++i) {
    // .bss is re-zeroed before each dispatch, so ++g_dma.seq starts at 1.
    if (records[i].phase != phases[i] || records[i].bytes != bytes ||
        records[i].seq != 1u) {
      std::fprintf(stderr,
                   "FAIL: trace record %d is seq=%u phase=%u bytes=%u "
                   "(expected seq=1 phase=%u bytes=%u)\n",
                   i, records[i].seq, records[i].phase, records[i].bytes,
                   phases[i], bytes);
      return false;
    }
  }
  return true;
}

// Runs the dispatch once on |sim| and checks its output and the trace.
bool RunOnce(coralnpu_simulator_t* sim, const std::vector<uint8_t>& image,
             int run) {
  std::vector<uint32_t> input(kBindingSize / sizeof(uint32_t));
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = 0xA5000000u + static_cast<uint32_t>(i);
  }
  std::vector<uint32_t> output(input.size(), 0);

  void* binding_ptrs[2] = {input.data(), output.data()};
  const size_t binding_lengths[2] = {kBindingSize, kBindingSize};
  // The output is staged at the bottom of DDR, next to the trace.
  const bool binding_writeable[2] = {false, true};

  iree_hal_executable_dispatch_state_v0_t dispatch_state = {};
  dispatch_state.workgroup_size_x = 1;
  dispatch_state.workgroup_size_y = 1;
  dispatch_state.workgroup_size_z = 1;
  dispatch_state.workgroup_count_x = 1;
  dispatch_state.workgroup_count_y = 1;
  dispatch_state.workgroup_count_z = 1;
  dispatch_state.max_concurrency = 1;
  dispatch_state.binding_count = 2;
  dispatch_state.binding_ptrs = binding_ptrs;
  dispatch_state.binding_lengths = binding_lengths;

  iree_status_t status = iree_hal_simulator_issue_dispatch_inline(
      sim, iree_make_const_byte_span(image.data(), image.size()),
      &dispatch_state, binding_writeable, /*ordinal=*/0,
      /*local_memory_size=*/0, /*out_stats=*/nullptr);
  if (!iree_status_is_ok(status)) {
    std::fprintf(stderr, "FAIL: run %d: ", run);
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return false;
  }

  for (size_t i = 0; i < output.size(); ++i) {
    if (output[i] != input[i]) {
      std::fprintf(stderr,
                   "FAIL: run %d: output word %zu is 0x%08x, expected 0x%08x\n",
                   run, i, output[i], input[i]);
      return false;
    }
  }
  if (!CheckTrace(sim, kBindingSize)) {
    std::fprintf(stderr, "  (run %d)\n", run);
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <elf>\n", argv[0]);
    return 2;
  }
  std::vector<uint8_t> image = ReadFile(argv[1]);

  coralnpu_simulator_t* sim = coralnpu_simulator_mpact_create();
  if (sim == nullptr) {
    std::fprintf(stderr, "failed to create MPACT simulator\n");
    return 2;
  }

  bool ok = true;
  for (int run = 0; run < kRuns && ok; ++run) {
    ok = RunOnce(sim, image, run);
  }

  iree_hal_simulator_inline_forget(sim);
  coralnpu_simulator_destroy(sim);
  return ok ? 0 : 1;
}
