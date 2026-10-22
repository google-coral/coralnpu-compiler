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

// The --coralnpu_stats_* flags of the IREE tools, which configure the
// statistics writer (stats_writer.h).

#ifndef METRICS_RUNTIME_STATS_FLAGS_H_
#define METRICS_RUNTIME_STATS_FLAGS_H_

#include "iree/base/api.h"
#include "runtime/driver/coralnpu_dispatch_observer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Sets |out_observer| to a statistics writer configured by the
// --coralnpu_stats_* flags or CORALNPU_STATS_*, unless the format is none.
iree_status_t iree_hal_coralnpu_stats_writer_from_flags(
    iree_hal_coralnpu_dispatch_observer_t *out_observer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // METRICS_RUNTIME_STATS_FLAGS_H_
