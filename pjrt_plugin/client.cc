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

#include "pjrt_plugin/client.h"

#include "iree/hal/local/loaders/embedded_elf_loader.h"
#include "iree/hal/local/plugins/registration/init.h"
#include "runtime/driver/coralnpu_driver.h"

namespace iree::pjrt::coralnpu {

CORALNPUClientInstance::CORALNPUClientInstance(
    std::unique_ptr<Platform> platform)
    : ClientInstance(std::move(platform)) {
  // Seems that it must match how registered. Action at a distance not
  // great.
  // TODO: Get this when constructing the client so it is guaranteed to
  // match.
  cached_platform_name_ = "iree_coralnpu";
}

CORALNPUClientInstance::~CORALNPUClientInstance() {
  iree_hal_allocator_release(device_allocator_);
  iree_hal_executable_loader_release(loader_[0]);
  if (plugin_manager_)
    iree_hal_executable_plugin_manager_release(plugin_manager_);
}

iree_status_t CORALNPUClientInstance::InitializeDeps() {
  // plugin_manager_
  IREE_RETURN_IF_ERROR(iree_hal_executable_plugin_manager_create(
      /*capacity=*/0, host_allocator_, &plugin_manager_));

  // loader
  IREE_RETURN_IF_ERROR(iree_hal_embedded_elf_loader_create(
      plugin_manager_, host_allocator_, &loader_[0]));

  // device_allocator_
  IREE_RETURN_IF_ERROR(iree_hal_allocator_create_heap(
      iree_make_cstring_view("local"), host_allocator_, host_allocator_,
      &device_allocator_));
  return iree_ok_status();
}

iree_status_t CORALNPUClientInstance::CreateDriver(
    iree_hal_driver_t** out_driver) {
  // TODO: There is substantial configuration available.
  // We choose to use explicit instantiation (vs registration) because
  // it is assumed that for server-library oriented cases, we are going to
  // want non-default control.
  IREE_RETURN_IF_ERROR(InitializeDeps());

  // driver
  logger().debug("Creating single threaded CoralNPU driver");
  iree_hal_coralnpu_device_params_t coralnpu_params;
  iree_hal_coralnpu_device_params_initialize(&coralnpu_params);
  IREE_RETURN_IF_ERROR(iree_hal_coralnpu_driver_create(
      IREE_SV("coralnpu"), &coralnpu_params, /* loader_count*/ 1, loader_,
      device_allocator_, host_allocator_, out_driver));

  logger().debug("CoralNPU driver created");
  return iree_ok_status();
}

bool CORALNPUClientInstance::SetDefaultCompilerFlags(
    CompilerJob* compiler_job) {
  return compiler_job->SetFlag("--iree-hal-target-device=coralnpu") &&
         compiler_job->SetFlag("--coralnpu-target-abi=ilp32") &&
         // TODO(b/507532766): Make this compiler flag configurable.
         compiler_job->SetFlag(
             "--coralnpu-target-cpu-features=+m,+f,+zvl128b,+zve32x");
}

}  // namespace iree::pjrt::coralnpu
