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

#include "runtime/driver/registration/driver_module.h"

#include <stddef.h>
#include <stdlib.h>

#include "iree/base/api.h"
#include "iree/base/internal/dynamic_library.h"
#include "iree/base/tooling/flags.h"
#include "runtime/driver/coralnpu_driver.h"
#include "runtime/sim/simulator_backend.h"

IREE_FLAG(string, simulator, "mpact",
          "Execution backend to run CoralNPU dispatches on (mpact, verilator, "
          "fpga).");

// Factory function for the MPACT functional simulator.
coralnpu_simulator_t* coralnpu_simulator_mpact_create(void);

static iree_status_t iree_hal_coralnpu_simulator_load_dylib(
    const char* library_name, const char* symbol_name,
    const char* unavailable_message,
    iree_hal_coralnpu_exec_backend_t* out_exec_backend) {
  // Never released: the loaded code must outlive the driver using it.
  iree_dynamic_library_t* library = NULL;
  iree_status_t status = iree_dynamic_library_load_from_file(
      library_name, IREE_DYNAMIC_LIBRARY_FLAG_NONE, iree_allocator_system(),
      &library);
  if (iree_status_is_not_found(status)) {
    iree_status_ignore(status);
    return iree_make_status(IREE_STATUS_UNAVAILABLE, "%s", unavailable_message);
  }
  IREE_RETURN_IF_ERROR(status);

  coralnpu_simulator_create_fn_t factory = NULL;
  IREE_RETURN_IF_ERROR(iree_dynamic_library_lookup_symbol(library, symbol_name,
                                                          (void**)&factory));
  *out_exec_backend = iree_hal_coralnpu_simulator_backend_make(factory);
  return iree_ok_status();
}

static iree_status_t iree_hal_coralnpu_simulator_load(
    iree_string_view_t name,
    iree_hal_coralnpu_exec_backend_t* out_exec_backend) {
  if (iree_string_view_is_empty(name) ||
      iree_string_view_equal(name, IREE_SV("mpact"))) {
    *out_exec_backend = iree_hal_coralnpu_simulator_backend_make(
        coralnpu_simulator_mpact_create);
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("verilator"))) {
    iree_status_t status = iree_hal_coralnpu_simulator_load_dylib(
        "libcoralnpu_simulator_vme.so", "coralnpu_simulator_verilator_create",
        "", out_exec_backend);
    if (iree_status_is_ok(status)) {
      return status;
    }
    iree_status_ignore(status);
    return iree_hal_coralnpu_simulator_load_dylib(
        "libcoralnpu_simulator_rvv.so", "coralnpu_simulator_verilator_create",
        "Verilator simulator library not available; ensure "
        "libcoralnpu_simulator_vme.so or libcoralnpu_simulator_rvv.so is in "
        "LD_LIBRARY_PATH",
        out_exec_backend);
  }
  if (iree_string_view_equal(name, IREE_SV("fpga")) ||
      iree_string_view_equal(name, IREE_SV("hw"))) {
    return iree_hal_coralnpu_simulator_load_dylib(
        "libcoralnpu_simulator_fpga.so", "coralnpu_simulator_fpga_create",
        "FPGA simulator library not available; ensure "
        "libcoralnpu_simulator_fpga.so is in LD_LIBRARY_PATH",
        out_exec_backend);
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "unknown simulator '%.*s' (expected 'mpact', 'verilator', or 'fpga')",
      (int)name.size, name.data);
}

// Set by iree_hal_coralnpu_driver_module_set_exec_backend; left zeroed (and
// thus resolved on demand below) unless a tool overrides it.
static iree_hal_coralnpu_exec_backend_t iree_hal_coralnpu_exec_backend_override;

void iree_hal_coralnpu_driver_module_set_exec_backend(
    const iree_hal_coralnpu_exec_backend_t* exec_backend) {
  iree_hal_coralnpu_exec_backend_override = *exec_backend;
}

static iree_status_t iree_hal_coralnpu_driver_factory_enumerate(
    void* self, iree_host_size_t* out_driver_info_count,
    const iree_hal_driver_info_t** out_driver_infos) {
  static const iree_hal_driver_info_t driver_infos[] = {
      {
          .driver_name = IREE_SVL("coralnpu"),
          .full_name = IREE_SVL("Coral NPU (RISC-V 32)"),
      },
  };
  *out_driver_info_count = IREE_ARRAYSIZE(driver_infos);
  *out_driver_infos = driver_infos;
  return iree_ok_status();
}

static iree_status_t iree_hal_coralnpu_driver_factory_try_create(
    void* self, iree_string_view_t driver_name, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  if (!iree_string_view_equal(driver_name, IREE_SV("coralnpu"))) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no driver '%.*s' is provided by this factory",
                            (int)driver_name.size, driver_name.data);
  }

  // Loading the simulator is deferred until here: driver registration happens
  // in every process linking the HAL and must never fail.
  iree_hal_coralnpu_exec_backend_t exec_backend =
      iree_hal_coralnpu_exec_backend_override;
  if (!exec_backend.create) {
    const char* sim_name = getenv("CORALNPU_SIMULATOR");
    if (!sim_name || !*sim_name) {
      sim_name = FLAG_simulator;
    }
    IREE_RETURN_IF_ERROR(iree_hal_coralnpu_simulator_load(
        iree_make_cstring_view(sim_name), &exec_backend));
  }

  iree_hal_coralnpu_device_params_t default_params;
  iree_hal_coralnpu_device_params_initialize(&default_params);

  // NOTE: no executable loaders are registered as CoralNPU only runs riscv_32
  // executables that the device loads itself.

  iree_hal_allocator_t* device_allocator = NULL;
  iree_status_t status = iree_hal_allocator_create_heap(
      iree_make_cstring_view("local"), host_allocator, host_allocator,
      &device_allocator);

  if (iree_status_is_ok(status)) {
    status = iree_hal_coralnpu_driver_create(driver_name, &default_params,
                                             &exec_backend, device_allocator,
                                             host_allocator, out_driver);
  }

  iree_hal_allocator_release(device_allocator);
  return status;
}

iree_status_t iree_hal_coralnpu_driver_module_register(
    iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = iree_hal_coralnpu_driver_factory_enumerate,
      .try_create = iree_hal_coralnpu_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
