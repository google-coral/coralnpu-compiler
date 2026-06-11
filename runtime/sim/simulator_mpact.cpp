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

#include <cstdio>

#include "runtime/sim/simulator_api.h"

struct CoralNPUMailbox {
  uint32_t message[4] = {0, 0, 0, 0};
};

class CoralNPUSimulator {
 public:
  static CoralNPUSimulator *Create();

  virtual ~CoralNPUSimulator() = default;

  // Functions for reading/writing TCMs and Mailbox.
  virtual void ReadTCM(uint32_t addr, size_t size, char *data) = 0;
  virtual const CoralNPUMailbox &ReadMailbox() = 0;
  virtual void WriteTCM(uint32_t addr, size_t size, const char *data) = 0;
  virtual void WriteMailbox(const CoralNPUMailbox &mailbox) = 0;

  // Wait for interrupt
  virtual bool WaitForTermination(int timeout) = 0;

  // Begin executing starting with the PC set to the specified address. Returns
  // when the core halts.
  virtual void Run(uint32_t start_addr) = 0;
};

static CoralNPUSimulator *sim = NULL;

void simulator_create(void) { sim = CoralNPUSimulator::Create(); }

void simulator_load_itcm(uint32_t offset, const void *data, size_t size) {
  sim->WriteTCM(coralnpu_itcm_start + offset, size,
                static_cast<const char *>(data));
}

void simulator_load_dtcm(uint32_t offset, const void *data, size_t size) {
  sim->WriteTCM(coralnpu_dtcm_start + offset, size,
                static_cast<const char *>(data));
}

void simulator_read_itcm(uint32_t offset, void *data, size_t size) {
  sim->ReadTCM(coralnpu_itcm_start + offset, size, static_cast<char *>(data));
}

void simulator_read_dtcm(uint32_t offset, void *data, size_t size) {
  sim->ReadTCM(coralnpu_dtcm_start + offset, size, static_cast<char *>(data));
}

void simulator_run(uint32_t pc) {
  sim->Run(coralnpu_itcm_start + pc);
  if (sim->WaitForTermination(500000)) {
    printf("Halted\n");
  } else {
    printf("Didn't halt\n");
  }
}
