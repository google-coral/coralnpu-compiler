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

#include "hw_sim/coralnpu_simulator.h"
#include "runtime/sim/simulator_api.h"

static CoralNPUSimulator* sim = NULL;

void simulator_create(void) {
  if (!sim) {
    sim = CoralNPUSimulator::Create();
  }
}

void simulator_reset(void) {
  if (sim) {
    delete sim;
  }
  sim = CoralNPUSimulator::Create();
}

void simulator_write_mem(uint32_t addr, const void* data, size_t size) {
  sim->WriteMem(addr, size, static_cast<const char*>(data));
}

void simulator_read_mem(uint32_t addr, void* data, size_t size) {
  sim->ReadMem(addr, size, static_cast<char*>(data));
}

void simulator_run(uint32_t start_pc) {
  sim->Run(start_pc);
  sim->WaitForTermination(1000000);
}

uint64_t simulator_get_cycle_count(void) {
  if (sim) {
    return sim->GetCycleCount();
  }
  return 0;
}
