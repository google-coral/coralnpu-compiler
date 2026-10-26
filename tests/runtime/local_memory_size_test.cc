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

// Tests that a dispatch receives the allocated local memory size.
//
// Usage:
//   local_memory_size_test <elf>

#include "runtime/sim/simulator_api.h"
#include "runtime/sim/simulator_inline.h"

// IREE:
#include "iree/base/api.h"
#include "iree/hal/local/executable_library.h"

// Standard library:
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

// Matches LOCAL_MEMORY_MARKER in local_memory_size_firmware.c.
constexpr uint32_t kMarker = 0x10CA1u;

// DTCM range from //crt:coralnpu_tcm_ld.
constexpr uint32_t kDtcmStart = 0x00010000u;
constexpr uint32_t kDtcmEnd = 0x00018000u;

constexpr uint32_t kPageSize =
    IREE_HAL_EXECUTABLE_WORKGROUP_LOCAL_MEMORY_PAGE_SIZE;
static_assert(kPageSize == 4096,
              "6-page max / 7-page overflow assumes 4 KB pages in 32 KB DTCM");

// Reads a file into memory, or exits on error.
std::vector<uint8_t> ReadFile(const char* path) {
  FILE* f = std::fopen(path, "rb");
  if (f == nullptr) {
    std::perror(path);
    std::exit(2);
  }
  std::fseek(f, 0, SEEK_END);
  long n = std::ftell(f);
  if (n < 0) {
    std::perror(path);
    std::exit(2);
  }
  std::fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> buf(static_cast<size_t>(n));
  if (std::fread(buf.data(), 1, buf.size(), f) != buf.size()) {
    std::fprintf(stderr, "short read: %s\n", path);
    std::exit(2);
  }
  std::fclose(f);
  return buf;
}

// Runs export 0 on |sim| with |local_memory_size| bytes of local memory.
iree_status_t IssueDispatch(coralnpu_simulator_t* sim,
                            const std::vector<uint8_t>& image,
                            uint32_t local_memory_size, uint32_t out[3]) {
  void* binding_ptrs[1] = {out};
  size_t binding_lengths[1] = {3 * sizeof(uint32_t)};
  const bool binding_writeable[1] = {true};

  iree_hal_executable_dispatch_state_v0_t dispatch_state = {};
  dispatch_state.workgroup_size_x = 1;
  dispatch_state.workgroup_size_y = 1;
  dispatch_state.workgroup_size_z = 1;
  dispatch_state.workgroup_count_x = 1;
  dispatch_state.workgroup_count_y = 1;
  dispatch_state.workgroup_count_z = 1;
  dispatch_state.max_concurrency = 1;
  dispatch_state.binding_count = 1;
  dispatch_state.binding_ptrs = binding_ptrs;
  dispatch_state.binding_lengths = binding_lengths;

  return iree_hal_simulator_issue_dispatch_inline(
      sim, iree_make_const_byte_span(image.data(), image.size()),
      &dispatch_state, binding_writeable, /*ordinal=*/0, local_memory_size,
      /*hooks=*/nullptr);
}

// Runs one dispatch and checks the local memory size and address.
bool RunCase(coralnpu_simulator_t* sim, const std::vector<uint8_t>& image,
             uint32_t local_memory_size) {
  uint32_t out[3] = {0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu};
  iree_status_t status = IssueDispatch(sim, image, local_memory_size, out);
  if (!iree_status_is_ok(status)) {
    std::fprintf(stderr,
                 "FAIL size=%" PRIu32 ": dispatch failed: ", local_memory_size);
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    return false;
  }

  std::fprintf(stderr,
               "size=%" PRIu32 ": guest saw local_memory_size=%" PRIu32
               " local_memory=0x%08" PRIx32 " last_word=0x%08" PRIx32 "\n",
               local_memory_size, out[0], out[1], out[2]);

  bool ok = true;
  if (out[0] != local_memory_size) {
    std::fprintf(stderr, "FAIL: expected local_memory_size=%" PRIu32 "\n",
                 local_memory_size);
    ok = false;
  }
  if (local_memory_size == 0) {
    if (out[1] != 0) {
      std::fprintf(stderr, "FAIL: expected a null local_memory\n");
      ok = false;
    }
  } else {
    if (out[1] < kDtcmStart ||
        static_cast<uint64_t>(out[1]) + local_memory_size > kDtcmEnd) {
      std::fprintf(stderr, "FAIL: local memory is outside DTCM\n");
      ok = false;
    }
    if (out[1] % 64 != 0) {
      std::fprintf(stderr, "FAIL: local memory is not 64-byte aligned\n");
      ok = false;
    }
    if (out[2] != kMarker) {
      std::fprintf(stderr, "FAIL: last word of local memory not writable\n");
      ok = false;
    }
  }
  return ok;
}

// Checks that an oversized local memory request fails with RESOURCE_EXHAUSTED.
bool RunOverflowCase(coralnpu_simulator_t* sim,
                     const std::vector<uint8_t>& image,
                     uint32_t local_memory_size) {
  uint32_t out[3] = {0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu};
  iree_status_t status = IssueDispatch(sim, image, local_memory_size, out);
  const iree_status_code_t code = iree_status_code(status);
  iree_status_ignore(status);
  if (code != IREE_STATUS_RESOURCE_EXHAUSTED) {
    std::fprintf(stderr,
                 "FAIL size=%" PRIu32
                 ": expected RESOURCE_EXHAUSTED, got status code %d\n",
                 local_memory_size, static_cast<int>(code));
    return false;
  }
  std::fprintf(stderr,
               "size=%" PRIu32
               ": rejected with RESOURCE_EXHAUSTED as expected\n",
               local_memory_size);
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

  // Run the 6-page max first as well so all 24 KB of local memory is dirtied
  // before verifying re-zeroing on 0 bytes, 1 page, 1 page + unaligned tail,
  // and 6 pages again.
  const uint32_t kSizes[] = {
      6 * kPageSize, 0, kPageSize, kPageSize + 259, 6 * kPageSize,
  };
  bool ok = true;
  for (uint32_t size : kSizes) {
    ok &= RunCase(sim, image, size);
  }
  // 7 pages is too big for the heap.
  ok &= RunOverflowCase(sim, image, 7 * kPageSize);

  iree_hal_simulator_inline_forget(sim);
  coralnpu_simulator_destroy(sim);
  return ok ? 0 : 1;
}
