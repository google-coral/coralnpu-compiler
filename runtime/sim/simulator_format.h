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

#ifndef RUNTIME_SIM_SIMULATOR_FORMAT_H_
#define RUNTIME_SIM_SIMULATOR_FORMAT_H_

#include "iree/base/api.h"

static inline bool
iree_hal_coralnpu_is_simulator_format(iree_string_view_t executable_format) {
  return iree_string_view_equal(executable_format,
                                IREE_SV("embedded-elf-riscv_32"));
}

#endif // RUNTIME_CORALNPU_SIM_SIMULATOR_FORMAT_H_