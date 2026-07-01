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

#include "iree_pjrt/common/dylib_platform.h"
#include "pjrt_plugin/client.h"

// Provides the shared library exports.
#include "iree_pjrt/common/dylib_entry_point.cc.inc"

namespace iree::pjrt {
namespace {

// Declared but not implemented by the include file.
void InitializeAPI(PJRT_Api* api) {
  BindApi<DylibPlatform, coralnpu::CORALNPUClientInstance>(api);
}

}  // namespace
}  // namespace iree::pjrt
