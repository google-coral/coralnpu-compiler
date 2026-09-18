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

#ifndef RUNTIME_SIM_SIMULATOR_API_H_
#define RUNTIME_SIM_SIMULATOR_API_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
class CoralNPUSimulator;
typedef CoralNPUSimulator coralnpu_simulator_t;
extern "C" {
#else
typedef struct coralnpu_simulator_t coralnpu_simulator_t;
#endif  // __cplusplus

typedef coralnpu_simulator_t *(*coralnpu_simulator_create_fn_t)(void);

// Creates an MPACT instruction-set simulator instance (always linked in).
coralnpu_simulator_t *coralnpu_simulator_mpact_create(void);

// Creates a Nexus FPGA hardware backend instance (linked in when
// CORALNPU_STATIC_FPGA is enabled; otherwise loaded via dlopen).
coralnpu_simulator_t *coralnpu_simulator_fpga_create(void);

void coralnpu_simulator_destroy(coralnpu_simulator_t *sim);
void coralnpu_simulator_write_mem(coralnpu_simulator_t *sim, uint32_t addr,
                                  const void *data, size_t size);
void coralnpu_simulator_read_mem(coralnpu_simulator_t *sim, uint32_t addr,
                                 void *data, size_t size);
// Returns false if the core did not halt.
bool coralnpu_simulator_run(coralnpu_simulator_t *sim, uint32_t start_pc);
uint64_t coralnpu_simulator_get_cycle_count(coralnpu_simulator_t *sim);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // RUNTIME_SIM_SIMULATOR_API_H_
