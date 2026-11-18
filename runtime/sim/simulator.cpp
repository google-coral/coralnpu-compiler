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

#include "hw_sim/coralnpu_simulator.h"
#include "runtime/sim/simulator_api.h"

void coralnpu_simulator_destroy(coralnpu_simulator_t* sim) { delete sim; }

void coralnpu_simulator_write_mem(coralnpu_simulator_t* sim, uint32_t addr,
                                  const void* data, size_t size) {
  sim->WriteMem(addr, size, static_cast<const char*>(data));
}

void coralnpu_simulator_read_mem(coralnpu_simulator_t* sim, uint32_t addr,
                                 void* data, size_t size) {
  sim->ReadMem(addr, size, static_cast<char*>(data));
}

namespace {

// Cycle budget per dispatch for backends that honor WaitForTermination's step
// limit (Verilator); large model dispatches exceed 1M RTL cycles.
constexpr int kMaxDispatchCycles = 100000000;

}  // namespace

bool coralnpu_simulator_run(coralnpu_simulator_t* sim, uint32_t start_pc) {
  sim->Run(start_pc);
  return sim->WaitForTermination(kMaxDispatchCycles);
}

uint64_t coralnpu_simulator_get_cycle_count(coralnpu_simulator_t* sim) {
  return sim->GetCycleCount();
}
