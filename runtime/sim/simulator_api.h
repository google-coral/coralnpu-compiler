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
static const uint32_t coralnpu_itcm_size = 0x00004000u;  // 8 KB
static const uint32_t coralnpu_dtcm_start = 0x00010000u;
static const uint32_t coralnpu_dtcm_size = 0x00008000u;  // 32 KB

void simulator_create(void);
void simulator_load_itcm(uint32_t offset, const void *data, size_t size);
void simulator_load_dtcm(uint32_t offset, const void *data, size_t size);
void simulator_read_itcm(uint32_t offset, void *data, size_t size);
void simulator_read_dtcm(uint32_t offset, void *data, size_t size);
void simulator_run(uint32_t start_pc);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // RUNTIME_SIM_SIMULATOR_API_H_
