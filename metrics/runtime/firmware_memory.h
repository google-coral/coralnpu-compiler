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

// Reads the memory layout of the firmware of a dispatch, the ELF that the core
// runs (crt/coralnpu_dispatch.h), from the symbols of the crt linker script
// and the PT_LOAD segments of the ELF.

#ifndef METRICS_RUNTIME_FIRMWARE_MEMORY_H_
#define METRICS_RUNTIME_FIRMWARE_MEMORY_H_

#include <stdbool.h>

#include "iree/base/api.h"
#include "metrics/runtime/dispatch_stats.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Reads the layout of the validated firmware ELF |image| into |out_memory|:
// the start and size of ITCM, DTCM, the heap and the stack, and the bytes of
// the PT_LOAD segments in ITCM, and in DTCM without the heap and the stack.
// Zeroes the other fields. Returns false if |image| lacks a layout symbol, or
// if the heap or the stack is not in DTCM.
bool iree_hal_coralnpu_firmware_memory_read(
    iree_const_byte_span_t image, iree_hal_coralnpu_memory_stats_t *out_memory);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // METRICS_RUNTIME_FIRMWARE_MEMORY_H_
