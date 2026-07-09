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

#include <inttypes.h>

#include "iree/hal/api.h"
#include "iree/hal/drivers/local_sync/sync_driver.h"
#include "iree/hal/local/loaders/registration/init.h"
#include "iree/hal/local/plugins/registration/init.h"
#include "runtime/driver/coralnpu_driver.h"

namespace iree::pjrt::coralnpu {
namespace {

// Define the composite driver structure.
typedef struct iree_hal_composite_driver_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_hal_driver_t* cpu_driver;
  iree_hal_driver_t* npu_driver;
} iree_hal_composite_driver_t;

extern const iree_hal_driver_vtable_t iree_hal_composite_driver_vtable;

static iree_hal_composite_driver_t* iree_hal_composite_driver_cast(
    iree_hal_driver_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_composite_driver_vtable);
  return (iree_hal_composite_driver_t*)base_value;
}

static void iree_hal_composite_driver_destroy(iree_hal_driver_t* base_driver) {
  iree_hal_composite_driver_t* driver =
      iree_hal_composite_driver_cast(base_driver);
  iree_allocator_t host_allocator = driver->host_allocator;

  iree_hal_driver_release(driver->cpu_driver);
  iree_hal_driver_release(driver->npu_driver);

  iree_allocator_free(host_allocator, driver);
}

static iree_status_t iree_hal_composite_driver_query_available_devices(
    iree_hal_driver_t* base_driver, iree_allocator_t host_allocator,
    iree_host_size_t* out_device_info_count,
    iree_hal_device_info_t** out_device_infos) {
  // We expose exactly two devices:
  // Index 0: CPU device
  // Index 1: CoralNPU device
  iree_hal_device_info_t* device_infos = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, 2 * sizeof(iree_hal_device_info_t),
                            (void**)&device_infos);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  // Device 0: CPU
  device_infos[0].device_id = 0;
  device_infos[0].name = iree_make_cstring_view("cpu");
  device_infos[0].path = iree_make_cstring_view("cpu");

  // Device 1: CoralNPU
  device_infos[1].device_id = 1;
  device_infos[1].name = iree_make_cstring_view("coralnpu");
  device_infos[1].path = iree_make_cstring_view("coralnpu");

  *out_device_info_count = 2;
  *out_device_infos = device_infos;
  return iree_ok_status();
}

static iree_status_t iree_hal_composite_driver_dump_device_info(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_string_builder_t* builder) {
  return iree_ok_status();
}

static iree_status_t iree_hal_composite_driver_create_device_by_id(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_composite_driver_t* driver =
      iree_hal_composite_driver_cast(base_driver);

  if (device_id == 0) {
    // Create CPU device.
    return iree_hal_driver_create_device_by_id(
        driver->cpu_driver, 0, param_count, params, host_allocator, out_device);
  } else if (device_id == 1) {
    // Create CoralNPU device.
    return iree_hal_driver_create_device_by_id(
        driver->npu_driver, 0, param_count, params, host_allocator, out_device);
  }

  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "device_id %" PRIuPTR " not found", device_id);
}

static iree_status_t iree_hal_composite_driver_create_device_by_path(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name,
    iree_string_view_t device_path, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  iree_hal_composite_driver_t* driver =
      iree_hal_composite_driver_cast(base_driver);
  if (iree_string_view_equal(device_path, IREE_SV("cpu")) ||
      iree_string_view_equal(device_path, IREE_SV("0"))) {
    return iree_hal_driver_create_device_by_id(
        driver->cpu_driver, 0, param_count, params, host_allocator, out_device);
  } else if (iree_string_view_equal(device_path, IREE_SV("coralnpu")) ||
             iree_string_view_equal(device_path, IREE_SV("1"))) {
    return iree_hal_driver_create_device_by_id(
        driver->npu_driver, 0, param_count, params, host_allocator, out_device);
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND, "device_path %.*s not found",
                          (int)device_path.size, device_path.data);
}

const iree_hal_driver_vtable_t iree_hal_composite_driver_vtable = {
    /*.destroy=*/iree_hal_composite_driver_destroy,
    /*.query_available_devices=*/
    iree_hal_composite_driver_query_available_devices,
    /*.dump_device_info=*/iree_hal_composite_driver_dump_device_info,
    /*.create_device_by_id=*/iree_hal_composite_driver_create_device_by_id,
    /*.create_device_by_path=*/iree_hal_composite_driver_create_device_by_path,
};

}  // namespace

CoralNPUClientInstance::CoralNPUClientInstance(
    std::unique_ptr<Platform> platform)
    : ClientInstance(std::move(platform)) {
  cached_platform_name_ = "iree_coralnpu";
}

CoralNPUClientInstance::~CoralNPUClientInstance() {
  iree_hal_allocator_release(device_allocator_);
  for (iree_host_size_t i = 0; i < loader_count_; ++i) {
    iree_hal_executable_loader_release(loaders_[i]);
  }
  if (plugin_manager_)
    iree_hal_executable_plugin_manager_release(plugin_manager_);
}

iree_status_t CoralNPUClientInstance::InitializeDeps() {
  // plugin_manager_
  IREE_RETURN_IF_ERROR(iree_hal_executable_plugin_manager_create(
      /*capacity=*/0, host_allocator_, &plugin_manager_));

  // loaders
  IREE_RETURN_IF_ERROR(iree_hal_create_all_available_executable_loaders(
      plugin_manager_, IREE_ARRAYSIZE(loaders_), &loader_count_, loaders_,
      host_allocator_));

  // device_allocator_
  IREE_RETURN_IF_ERROR(iree_hal_allocator_create_heap(
      iree_make_cstring_view("local"), host_allocator_, host_allocator_,
      &device_allocator_));
  return iree_ok_status();
}

iree_status_t CoralNPUClientInstance::CreateDriver(
    iree_hal_driver_t** out_driver) {
  IREE_RETURN_IF_ERROR(InitializeDeps());

  // 1. Create CoralNPU driver.
  iree_hal_driver_t* npu_driver = nullptr;
  iree_hal_coralnpu_device_params_t coralnpu_params;
  iree_hal_coralnpu_device_params_initialize(&coralnpu_params);
  IREE_RETURN_IF_ERROR(iree_hal_coralnpu_driver_create(
      IREE_SV("coralnpu"), &coralnpu_params, loader_count_, loaders_,
      device_allocator_, host_allocator_, &npu_driver));

  // 2. Create CPU driver.
  iree_hal_driver_t* cpu_driver = nullptr;
  iree_hal_sync_device_params_t sync_params;
  iree_hal_sync_device_params_initialize(&sync_params);
  IREE_RETURN_IF_ERROR(iree_hal_sync_driver_create(
      IREE_SV("local-sync"), &sync_params, loader_count_, loaders_,
      device_allocator_, host_allocator_, &cpu_driver));

  // 3. Create the composite wrapper driver.
  iree_hal_composite_driver_t* driver = nullptr;
  iree_status_t status =
      iree_allocator_malloc(host_allocator_, sizeof(*driver), (void**)&driver);
  if (!iree_status_is_ok(status)) {
    iree_hal_driver_release(npu_driver);
    iree_hal_driver_release(cpu_driver);
    return status;
  }

  iree_hal_resource_initialize(&iree_hal_composite_driver_vtable,
                               &driver->resource);
  driver->host_allocator = host_allocator_;
  driver->cpu_driver = cpu_driver;
  driver->npu_driver = npu_driver;

  *out_driver = (iree_hal_driver_t*)driver;
  return iree_ok_status();
}

iree_status_t CoralNPUClientInstance::PopulateVMModules(
    std::vector<iree::vm::ref<iree_vm_module_t>>& modules,
    iree_hal_device_t* hal_device,
    iree::vm::ref<iree_vm_module_t>& main_module) {
  iree_hal_device_group_builder_t builder;
  iree_hal_device_group_builder_initialize(&builder);
  for (DeviceInstance* dev_inst : addressable_devices()) {
    iree_hal_device_t* dev = nullptr;
    IREE_RETURN_IF_ERROR(dev_inst->GetHalDevice(&dev));
    IREE_RETURN_IF_ERROR(
        iree_hal_device_group_builder_add_device(&builder, dev));
  }
  iree_hal_device_group_t* device_group = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_device_group_builder_finalize(
      &builder, host_allocator_, &device_group));

  modules.push_back({});
  iree_status_t status = iree_hal_module_create(
      vm_instance(), iree_hal_module_device_policy_default(), device_group,
      IREE_HAL_MODULE_FLAG_NONE, iree_hal_module_debug_sink_stdio(stderr),
      host_allocator_, &modules.back());
  iree_hal_device_group_release(device_group);
  IREE_RETURN_IF_ERROR(status);

  modules.push_back(main_module);
  return iree_ok_status();
}

bool CoralNPUClientInstance::SetCompilerFlags(
    CompilerJob* compiler_job, const xla::CompileOptionsProto& options) {
  int target_dev = -1;
  const auto& build_options = options.executable_build_options();
  if (build_options.has_device_assignment() &&
      build_options.device_assignment().computation_devices_size() > 0 &&
      build_options.device_assignment()
              .computation_devices(0)
              .replica_device_ids_size() > 0) {
    target_dev = build_options.device_assignment()
                     .computation_devices(0)
                     .replica_device_ids(0);
  }

  bool is_uncommitted =
      build_options.allow_spmd_sharding_propagation_to_parameters_size() > 0 &&
      build_options.allow_spmd_sharding_propagation_to_parameters(0);

  auto set_cpu_flags = [&]() {
    return compiler_job->SetFlag("--iree-hal-target-device=local") &&
           compiler_job->SetFlag(
               "--iree-hal-local-target-device-backends=llvm-cpu") &&
           compiler_job->SetFlag("--iree-llvmcpu-target-cpu=host");
  };

  auto set_coralnpu_flags = [&]() {
    return compiler_job->SetFlag("--iree-hal-target-device=coralnpu") &&
           compiler_job->SetFlag("--coralnpu-target-abi=ilp32") &&
           compiler_job->SetFlag(
               "--coralnpu-target-cpu-features=+m,+f,+zvl128b,+zve32f") &&
           compiler_job->SetFlag("--iree-execution-model=async-internal");
  };

  if (!is_uncommitted && target_dev == 0) {
    // Single-device CPU execution: compile exclusively for CPU.
    return set_cpu_flags();
  } else if (!is_uncommitted && target_dev == 1) {
    // Single-device CoralNPU execution: compile exclusively for CoralNPU.
    return set_coralnpu_flags();
  }

  // Multi-device execution: enable both CPU and CoralNPU targets so
  // CoralNPUAffinityAnnotation partitions ops between CPU and CoralNPU.
  return set_cpu_flags() && set_coralnpu_flags();
}

bool CoralNPUClientInstance::SetDefaultCompilerFlags(
    CompilerJob* compiler_job) {
  return SetCompilerFlags(compiler_job, xla::CompileOptionsProto());
}

}  // namespace iree::pjrt::coralnpu
