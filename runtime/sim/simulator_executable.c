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

#include "runtime/sim/simulator_executable.h"

#include <string.h>

#include "iree/base/api.h"
#include "runtime/driver/coralnpu_executable.h"

typedef struct iree_hal_simulator_executable_t {
  iree_hal_coralnpu_executable_t base;
  iree_const_byte_span_t dispatch_image;
  iree_host_size_t export_count;
} iree_hal_simulator_executable_t;

static iree_hal_simulator_executable_t *iree_hal_simulator_executable_cast(
    iree_hal_executable_t *base_executable) {
  return (iree_hal_simulator_executable_t *)base_executable;
}

static const iree_hal_coralnpu_executable_vtable_t
    iree_hal_simulator_executable_vtable;

bool iree_hal_simulator_executable_isa(iree_hal_executable_t *base_executable) {
  if (!base_executable) return false;

  iree_hal_coralnpu_executable_t *coralnpu_executable =
      iree_hal_coralnpu_executable_cast(base_executable);

  return coralnpu_executable->resource.vtable ==
         &iree_hal_simulator_executable_vtable.base;
}

iree_const_byte_span_t iree_hal_simulator_executable_dispatch_image(
    iree_hal_executable_t *base_executable) {
  iree_hal_simulator_executable_t *executable =
      iree_hal_simulator_executable_cast(base_executable);
  return executable->dispatch_image;
}

static void iree_hal_simulator_executable_destroy(
    iree_hal_executable_t *base_executable) {
  iree_hal_simulator_executable_t *executable =
      iree_hal_simulator_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->base.host_allocator;
  iree_hal_coralnpu_executable_deinitialize(&executable->base);
  iree_allocator_free(host_allocator, executable);
}

static iree_host_size_t iree_hal_simulator_executable_export_count(
    iree_hal_executable_t *base_executable) {
  iree_hal_simulator_executable_t *executable =
      iree_hal_simulator_executable_cast(base_executable);
  return executable->export_count;
}

static iree_status_t iree_hal_simulator_executable_export_info(
    iree_hal_executable_t *base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_hal_executable_export_info_t *out_info) {
  iree_hal_simulator_executable_t *executable =
      iree_hal_simulator_executable_cast(base_executable);

  if (!out_info) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "out_info is null");
  }
  if (export_ordinal >= executable->export_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "export ordinal out of range");
  }

  memset(out_info, 0, sizeof(*out_info));
  return iree_ok_status();
}

static iree_status_t iree_hal_simulator_executable_export_parameters(
    iree_hal_executable_t *base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_host_size_t capacity,
    iree_hal_executable_export_parameter_t *out_parameters) {
  iree_hal_simulator_executable_t *executable =
      iree_hal_simulator_executable_cast(base_executable);

  if (export_ordinal >= executable->export_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "export ordinal out of range");
  }
  if (capacity > 0 && !out_parameters) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_parameters is null");
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_simulator_executable_lookup_export_by_name(
    iree_hal_executable_t *base_executable, iree_string_view_t name,
    iree_hal_executable_export_ordinal_t *out_export_ordinal) {
  (void)base_executable;
  (void)name;
  if (out_export_ordinal) *out_export_ordinal = 0;
  return iree_make_status(IREE_STATUS_NOT_FOUND, "export lookup unsupported");
}

static iree_status_t iree_hal_simulator_executable_issue_call(
    iree_hal_coralnpu_executable_t *executable, iree_host_size_t ordinal,
    const iree_hal_executable_dispatch_state_v0_t *dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t *workgroup_state,
    uint32_t worker_id) {
  (void)executable;
  (void)ordinal;
  (void)dispatch_state;
  (void)workgroup_state;
  (void)worker_id;
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "simulator executable does not support per-workgroup issue_call");
}

static const iree_hal_coralnpu_executable_vtable_t
    iree_hal_simulator_executable_vtable = {
        .base =
            {
                .destroy = iree_hal_simulator_executable_destroy,
                .export_count = iree_hal_simulator_executable_export_count,
                .export_info = iree_hal_simulator_executable_export_info,
                .export_parameters =
                    iree_hal_simulator_executable_export_parameters,
                .lookup_export_by_name =
                    iree_hal_simulator_executable_lookup_export_by_name,
            },
        .issue_call = iree_hal_simulator_executable_issue_call,
};

iree_status_t iree_hal_simulator_executable_create(
    const iree_hal_executable_params_t *executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t **out_executable) {
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;

  if (!executable_params->executable_data.data ||
      executable_params->executable_data.data_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "empty executable_data");
  }

  const iree_host_size_t image_size =
      executable_params->executable_data.data_length;
  const iree_host_size_t total_size =
      sizeof(iree_hal_simulator_executable_t) + image_size;

  iree_hal_simulator_executable_t *executable = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void **)&executable));
  memset(executable, 0, total_size);

  iree_hal_coralnpu_executable_initialize(&iree_hal_simulator_executable_vtable,
                                          host_allocator, &executable->base);

  uint8_t *image_storage = (uint8_t *)executable + sizeof(*executable);
  memcpy(image_storage, executable_params->executable_data.data, image_size);

  executable->dispatch_image =
      iree_make_const_byte_span(image_storage, image_size);

  // Current simulator patch supports one export per executable image.
  executable->export_count = 1;

  *out_executable = (iree_hal_executable_t *)&executable->base;
  return iree_ok_status();
}
