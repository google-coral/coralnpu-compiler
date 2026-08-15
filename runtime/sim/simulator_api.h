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

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

static const uint32_t coralnpu_itcm_start = 0x00000000u;
extern uint32_t coralnpu_itcm_size;
extern uint32_t coralnpu_dtcm_start;
extern uint32_t coralnpu_dtcm_size;

void simulator_create(void);
void simulator_reset(void);
void simulator_write_mem(uint32_t addr, const void *data, size_t size);
void simulator_read_mem(uint32_t addr, void *data, size_t size);
void simulator_run(uint32_t start_pc);
uint64_t simulator_get_cycle_count(void);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // RUNTIME_SIM_SIMULATOR_API_H_
