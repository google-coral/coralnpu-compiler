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

#ifndef PJRT_PLUGIN_CLIENT_H_
#define PJRT_PLUGIN_CLIENT_H_

#include "iree/hal/local/loaders/registration/init.h"
#include "iree_pjrt/common/api_impl.h"

namespace iree::pjrt::coralnpu {

class CoralNPUClientInstance final : public ClientInstance {
 public:
  CoralNPUClientInstance(std::unique_ptr<Platform> platform);
  ~CoralNPUClientInstance();
  iree_status_t CreateDriver(iree_hal_driver_t **out_driver) override;
  iree_status_t PopulateVMModules(
      std::vector<iree::vm::ref<iree_vm_module_t>> &modules,
      iree_hal_device_t *hal_device,
      iree::vm::ref<iree_vm_module_t> &main_module) override;
  bool SetDefaultCompilerFlags(CompilerJob *compiler_job) override;
  bool SetCompilerFlags(CompilerJob *compiler_job,
                        const xla::CompileOptionsProto &options) override;

 private:
  iree_status_t InitializeDeps();

  // Deps.
  iree_hal_executable_plugin_manager_t *plugin_manager_ = nullptr;
  iree_hal_executable_loader_t *loaders_[8] = {nullptr};
  iree_host_size_t loader_count_ = 0;
  iree_hal_allocator_t *device_allocator_ = nullptr;
};

}  // namespace iree::pjrt::coralnpu

#endif  // PJRT_PLUGIN_CLIENT_H_
