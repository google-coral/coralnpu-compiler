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

#include "runtime/driver/coralnpu_command_buffer.h"

#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "runtime/driver/coralnpu_device.h"
#include "runtime/driver/coralnpu_dispatch_observer.h"
#include "runtime/driver/coralnpu_executable.h"

//===----------------------------------------------------------------------===//
// iree_hal_coralnpu_command_buffer_t
//===----------------------------------------------------------------------===//

// Inline synchronous one-shot command "buffer".
typedef struct iree_hal_coralnpu_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_hal_device_t *device;
  iree_allocator_t host_allocator;
} iree_hal_coralnpu_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
    iree_hal_coralnpu_command_buffer_vtable;

static iree_hal_coralnpu_command_buffer_t *
iree_hal_coralnpu_command_buffer_cast(iree_hal_command_buffer_t *base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_coralnpu_command_buffer_vtable);
  return (iree_hal_coralnpu_command_buffer_t *)base_value;
}

iree_host_size_t iree_hal_coralnpu_command_buffer_size(
    iree_hal_command_buffer_mode_t mode, iree_host_size_t binding_capacity) {
  return sizeof(iree_hal_coralnpu_command_buffer_t) +
         iree_hal_command_buffer_validation_state_size(mode, binding_capacity);
}

iree_status_t iree_hal_coralnpu_command_buffer_initialize(
    iree_hal_device_t *device, iree_hal_allocator_t *device_allocator,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_allocator_t host_allocator, iree_byte_span_t storage,
    iree_hal_command_buffer_t **out_command_buffer) {
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;

  if (!iree_all_bits_set(
          mode, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
                    IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION)) {
    // This implementation only supports command buffers that are allowed to
    // execute inline. This mode is a contract with the caller that it is ok if
    // we begin executing prior to submission.
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "inline command buffers must have a mode with ALLOW_INLINE_EXECUTION");
  }
  if (binding_capacity > 0) {
    // We execute as we record and can't use binding tables to do that.
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "indirect command buffers do not support binding tables");
  }
  if (storage.data_length <
      iree_hal_coralnpu_command_buffer_size(mode, binding_capacity)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "storage must have at least the capacity as "
                            "defined by iree_hal_coralnpu_command_buffer_size");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_coralnpu_command_buffer_t *command_buffer =
      (iree_hal_coralnpu_command_buffer_t *)storage.data;
  memset(command_buffer, 0, sizeof(*command_buffer));

  iree_hal_command_buffer_initialize(
      device_allocator, mode, command_categories, queue_affinity,
      binding_capacity, (uint8_t *)command_buffer + sizeof(*command_buffer),
      &iree_hal_coralnpu_command_buffer_vtable, &command_buffer->base);
  command_buffer->device = device;
  command_buffer->host_allocator = host_allocator;

  *out_command_buffer = &command_buffer->base;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_coralnpu_command_buffer_create(
    iree_hal_device_t *device, iree_hal_allocator_t *device_allocator,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t **out_command_buffer) {
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_host_size_t storage_size =
      iree_hal_coralnpu_command_buffer_size(mode, binding_capacity);
  uint8_t *storage = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, storage_size, (void **)&storage);
  iree_hal_command_buffer_t *command_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_coralnpu_command_buffer_initialize(
        device, device_allocator, mode, command_categories, queue_affinity,
        binding_capacity, host_allocator,
        iree_make_byte_span(storage, storage_size), &command_buffer);
  }

  if (iree_status_is_ok(status)) {
    *out_command_buffer = command_buffer;
  } else {
    iree_allocator_free(host_allocator, storage);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_coralnpu_command_buffer_destroy(
    iree_hal_command_buffer_t *base_command_buffer) {
  iree_hal_coralnpu_command_buffer_t *command_buffer =
      iree_hal_coralnpu_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// iree_hal_coralnpu_command_buffer_t recording
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_coralnpu_command_buffer_begin(
    iree_hal_command_buffer_t *base_command_buffer) {
  return iree_ok_status();
}

static iree_status_t iree_hal_coralnpu_command_buffer_end(
    iree_hal_command_buffer_t *base_command_buffer) {
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_coralnpu_command_buffer_t debug utilities
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_coralnpu_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t *base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t *location) {
  // TODO: tracy event stack.
  return iree_ok_status();
}

static iree_status_t iree_hal_coralnpu_command_buffer_end_debug_group(
    iree_hal_command_buffer_t *base_command_buffer) {
  // TODO: tracy event stack.
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_execution_barrier
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_coralnpu_command_buffer_execution_barrier(
    iree_hal_command_buffer_t *base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t *memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t *buffer_barriers) {
  // No-op; we execute synchronously.
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_signal_event
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_coralnpu_command_buffer_signal_event(
    iree_hal_command_buffer_t *base_command_buffer, iree_hal_event_t *event,
    iree_hal_execution_stage_t source_stage_mask) {
  // No-op; we execute synchronously.
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_reset_event
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_coralnpu_command_buffer_reset_event(
    iree_hal_command_buffer_t *base_command_buffer, iree_hal_event_t *event,
    iree_hal_execution_stage_t source_stage_mask) {
  // No-op; we execute synchronously.
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_wait_events
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_coralnpu_command_buffer_wait_events(
    iree_hal_command_buffer_t *base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t **events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t *memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t *buffer_barriers) {
  // No-op; we execute synchronously.
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_advise_buffer
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_coralnpu_command_buffer_advise_buffer(
    iree_hal_command_buffer_t *base_command_buffer,
    iree_hal_buffer_ref_t buffer_ref, iree_hal_memory_advise_flags_t flags,
    uint64_t arg0, uint64_t arg1) {
  // Could be treated as a cache invalidation as it indicates we won't be using
  // the existing buffer contents again.
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_fill_buffer
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_coralnpu_command_buffer_fill_buffer(
    iree_hal_command_buffer_t *base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void *pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  return iree_hal_buffer_map_fill(target_ref.buffer, target_ref.offset,
                                  target_ref.length, pattern, pattern_length);
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_update_buffer
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_coralnpu_command_buffer_update_buffer(
    iree_hal_command_buffer_t *base_command_buffer, const void *source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
    iree_hal_update_flags_t flags) {
  return iree_hal_buffer_map_write(
      target_ref.buffer, target_ref.offset,
      (const uint8_t *)source_buffer + source_offset, target_ref.length);
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_copy_buffer
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_coralnpu_command_buffer_copy_buffer(
    iree_hal_command_buffer_t *base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref,
    iree_hal_copy_flags_t flags) {
  return iree_hal_buffer_map_copy(source_ref.buffer, source_ref.offset,
                                  target_ref.buffer, target_ref.offset,
                                  target_ref.length);
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_collective
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_coralnpu_command_buffer_collective(
    iree_hal_command_buffer_t *base_command_buffer, iree_hal_channel_t *channel,
    iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref,
    iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "collectives not yet implemented on CoralNPU");
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_dispatch
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_coralnpu_command_buffer_dispatch(
    iree_hal_command_buffer_t *base_command_buffer,
    iree_hal_executable_t *executable, iree_hal_executable_function_t function,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  iree_hal_coralnpu_command_buffer_t *command_buffer =
      iree_hal_coralnpu_command_buffer_cast(base_command_buffer);

  // TODO: support custom arguments (direct constants or indirect buffer).
  if (iree_hal_dispatch_uses_custom_arguments(flags)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "direct/indirect arguments are not supported on "
                            "the inline CoralNPU command buffer");
  }

  if (!iree_hal_coralnpu_executable_isa(executable)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported executable type for CoralNPU command buffer");
  }
  const iree_host_size_t function_count =
      iree_hal_executable_function_count(executable);
  if (!iree_hal_executable_function_is_index_in_range(function,
                                                      function_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "function id %" PRIu64
                            " out of range (count: %" PRIhsz ")",
                            function.value, function_count);
  }
  const uint32_t export_ordinal = iree_hal_executable_function_index(function);
  const iree_hal_executable_dispatch_attrs_v0_t *attrs =
      iree_hal_coralnpu_executable_dispatch_attrs(executable, export_ordinal);
  if (bindings.count > IREE_HAL_EXECUTABLE_MAX_BINDING_COUNT) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "dispatch uses %" PRIhsz " bindings but only %d are supported",
        bindings.count, IREE_HAL_EXECUTABLE_MAX_BINDING_COUNT);
  }

  // The backend consumes the state synchronously so it can live on the stack.
  void *binding_ptrs[IREE_HAL_EXECUTABLE_MAX_BINDING_COUNT];
  size_t binding_lengths[IREE_HAL_EXECUTABLE_MAX_BINDING_COUNT];
  iree_hal_executable_dispatch_state_v0_t dispatch_state = {
      .max_concurrency = 1,
      .binding_ptrs = binding_ptrs,
      .binding_lengths = binding_lengths,
  };

  // Workgroup size from the dispatch config.
  dispatch_state.workgroup_size_x =
      config.workgroup_size[0] ? config.workgroup_size[0] : 1;
  dispatch_state.workgroup_size_y =
      config.workgroup_size[1] ? config.workgroup_size[1] : 1;
  dispatch_state.workgroup_size_z =
      config.workgroup_size[2] ? config.workgroup_size[2] : 1;

  // Workgroup count.
  uint32_t workgroup_count[3] = {config.workgroup_count[0],
                                 config.workgroup_count[1],
                                 config.workgroup_count[2]};
  if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
    IREE_RETURN_IF_ERROR(iree_hal_buffer_map_read(
        config.workgroup_count_ref.buffer, config.workgroup_count_ref.offset,
        workgroup_count, sizeof(workgroup_count)));
  }
  dispatch_state.workgroup_count_x = workgroup_count[0];
  dispatch_state.workgroup_count_y = workgroup_count[1];
  dispatch_state.workgroup_count_z = workgroup_count[2];

  // Constants.
  if ((constants.data_length % sizeof(uint32_t)) != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "constants must be 4-byte aligned");
  }
  const iree_host_size_t constant_count =
      constants.data_length / sizeof(uint32_t);
  if (constant_count > IREE_HAL_EXECUTABLE_MAX_CONSTANT_COUNT) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "dispatch uses %" PRIhsz " constants but only %d are supported",
        constant_count, IREE_HAL_EXECUTABLE_MAX_CONSTANT_COUNT);
  }
  // Kernels index constants and bindings by the export's counts.
  if (constant_count != attrs->constant_count ||
      bindings.count != attrs->binding_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "dispatch provides %" PRIhsz " constants and %" PRIhsz
        " bindings but the export expects %u and %u",
        constant_count, bindings.count, (uint32_t)attrs->constant_count,
        (uint32_t)attrs->binding_count);
  }
  dispatch_state.constant_count = (uint16_t)constant_count;
  dispatch_state.constants = (const uint32_t *)constants.data;

  dispatch_state.binding_count = (uint8_t)bindings.count;

  // Read-only bindings must not be written back after simulation. Treat
  // SHARING_IMMUTABLE | SHARING_REPLICATE buffers as read-only to cache them.
  bool binding_writeable[IREE_HAL_EXECUTABLE_MAX_BINDING_COUNT];
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    if (!bindings.values[i].buffer) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "required binding %" PRIhsz " is NULL", i);
    }
    iree_hal_buffer_mapping_t buffer_mapping = {{0}};
    IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
        bindings.values[i].buffer, IREE_HAL_MAPPING_MODE_PERSISTENT,
        IREE_HAL_MEMORY_ACCESS_ANY, bindings.values[i].offset,
        bindings.values[i].length, &buffer_mapping));
    binding_ptrs[i] = buffer_mapping.contents.data;
    binding_lengths[i] = buffer_mapping.contents.data_length;
    binding_writeable[i] =
        (iree_hal_buffer_allowed_access(bindings.values[i].buffer) &
         IREE_HAL_MEMORY_ACCESS_WRITE) != 0 &&
        !iree_all_bits_set(
            iree_hal_buffer_allowed_usage(bindings.values[i].buffer),
            IREE_HAL_BUFFER_USAGE_SHARING_IMMUTABLE |
                IREE_HAL_BUFFER_USAGE_SHARING_REPLICATE);
  }

  const iree_host_size_t local_memory_size =
      attrs->local_memory_pages *
          IREE_HAL_EXECUTABLE_WORKGROUP_LOCAL_MEMORY_PAGE_SIZE +
      config.dynamic_workgroup_local_memory;
  const iree_hal_coralnpu_dispatch_info_t dispatch_info = {
      .executable_name = iree_hal_coralnpu_executable_name(executable),
      .export_name = iree_hal_coralnpu_executable_function_name(executable,
                                                                export_ordinal),
      .export_ordinal = export_ordinal,
  };
  return iree_hal_coralnpu_device_dispatch(
      command_buffer->device, &dispatch_info,
      iree_hal_coralnpu_executable_dispatch_image(executable), &dispatch_state,
      binding_writeable, export_ordinal, local_memory_size);
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_vtable_t
//===----------------------------------------------------------------------===//

static const iree_hal_command_buffer_vtable_t
    iree_hal_coralnpu_command_buffer_vtable = {
        .destroy = iree_hal_coralnpu_command_buffer_destroy,
        .begin = iree_hal_coralnpu_command_buffer_begin,
        .end = iree_hal_coralnpu_command_buffer_end,
        .begin_debug_group = iree_hal_coralnpu_command_buffer_begin_debug_group,
        .end_debug_group = iree_hal_coralnpu_command_buffer_end_debug_group,
        .execution_barrier = iree_hal_coralnpu_command_buffer_execution_barrier,
        .signal_event = iree_hal_coralnpu_command_buffer_signal_event,
        .reset_event = iree_hal_coralnpu_command_buffer_reset_event,
        .wait_events = iree_hal_coralnpu_command_buffer_wait_events,
        .advise_buffer = iree_hal_coralnpu_command_buffer_advise_buffer,
        .fill_buffer = iree_hal_coralnpu_command_buffer_fill_buffer,
        .update_buffer = iree_hal_coralnpu_command_buffer_update_buffer,
        .copy_buffer = iree_hal_coralnpu_command_buffer_copy_buffer,
        .collective = iree_hal_coralnpu_command_buffer_collective,
        .dispatch = iree_hal_coralnpu_command_buffer_dispatch,
};
