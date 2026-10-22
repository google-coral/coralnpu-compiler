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

#include "runtime/driver/coralnpu_device.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "iree/async/frontier.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/drivers/local_sync/sync_event.h"
#include "iree/hal/drivers/local_sync/sync_semaphore.h"
#include "iree/hal/utils/deferred_command_buffer.h"
#include "iree/hal/utils/file_registry.h"
#include "iree/hal/utils/file_transfer.h"
#include "iree/hal/utils/queue_emulation.h"
#include "runtime/driver/coralnpu_command_buffer.h"
#include "runtime/driver/coralnpu_executable.h"
#include "runtime/driver/coralnpu_executable_cache.h"

typedef struct iree_hal_coralnpu_device_t {
  iree_hal_resource_t resource;
  iree_string_view_t identifier;

  iree_allocator_t host_allocator;
  iree_hal_allocator_t *device_allocator;

  iree_async_proactor_pool_t *proactor_pool;
  iree_async_proactor_t *proactor;
  iree_async_frontier_tracker_t *frontier_tracker;
  iree_async_axis_t axis;
  iree_atomic_int64_t epoch;

  iree_hal_coralnpu_exec_backend_t exec_backend;

  // Observer of the dispatches; |dispatch_observer.begin_dispatch| is NULL if
  // the device has none.
  iree_hal_coralnpu_dispatch_observer_t dispatch_observer;
  void *dispatch_observer_context;

  // Optional provider used for creating/configuring collective channels.
  iree_hal_channel_provider_t *channel_provider;

  iree_hal_device_topology_info_t topology_info;

  // Block pool used for command buffers with a larger block size (as command
  // buffers can contain inlined data uploads).
  iree_arena_block_pool_t large_block_pool;
} iree_hal_coralnpu_device_t;

static const iree_hal_device_vtable_t iree_hal_coralnpu_device_vtable;

static iree_hal_coralnpu_device_t *iree_hal_coralnpu_device_cast(
    iree_hal_device_t *base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_coralnpu_device_vtable);
  return (iree_hal_coralnpu_device_t *)base_value;
}

static void iree_hal_coralnpu_device_advance_frontier(
    iree_hal_coralnpu_device_t *device) {
  if (!device->frontier_tracker) return;
  uint64_t epoch = (uint64_t)iree_atomic_fetch_add(&device->epoch, 1,
                                                   iree_memory_order_acq_rel) +
                   1;
  iree_async_frontier_tracker_advance(device->frontier_tracker, device->axis,
                                      epoch);
}

// Prints |observer_status|, an error of the dispatch observer, as a warning
// and ignores it; does nothing if it is OK.
static void iree_hal_coralnpu_device_warn_observer(
    iree_status_t observer_status) {
  if (iree_status_is_ok(observer_status)) return;
  // Statistics must not fail the dispatch.
  fprintf(stderr, "[CoralNPU] Warning: dispatch statistics are incomplete: ");
  iree_status_fprint(stderr, observer_status);
  iree_status_ignore(observer_status);
}

iree_status_t iree_hal_coralnpu_device_dispatch(
    iree_hal_device_t *base_device,
    const iree_hal_coralnpu_dispatch_info_t *info,
    iree_const_byte_span_t dispatch_image,
    const iree_hal_executable_dispatch_state_v0_t *dispatch_state,
    const bool *binding_writeable, iree_host_size_t ordinal,
    iree_host_size_t local_memory_size) {
  IREE_ASSERT_ARGUMENT(info);
  iree_hal_coralnpu_device_t *device =
      iree_hal_coralnpu_device_cast(base_device);
  const iree_hal_coralnpu_dispatch_observer_t *observer =
      &device->dispatch_observer;
  void *observer_context = device->dispatch_observer_context;

  // The dispatch runs with the hooks of the observer only if begin_dispatch
  // succeeds; without an observer, it runs without hooks.
  iree_hal_coralnpu_run_hooks_t hooks;
  memset(&hooks, 0, sizeof(hooks));
  bool observed = false;
  if (observer->begin_dispatch) {
    iree_status_t observer_status = observer->begin_dispatch(
        observer->self, observer_context, info, &hooks);
    observed = iree_status_is_ok(observer_status);
    iree_hal_coralnpu_device_warn_observer(observer_status);
  }

  iree_status_t status = device->exec_backend.dispatch(
      device->exec_backend.self, dispatch_image, dispatch_state,
      binding_writeable, ordinal, local_memory_size, observed ? &hooks : NULL);
  if (observed) {
    iree_hal_coralnpu_device_warn_observer(
        observer->end_dispatch(observer->self, observer_context, info, &hooks,
                               iree_status_code(status)));
  }
  return status;
}

void iree_hal_coralnpu_device_params_initialize(
    iree_hal_coralnpu_device_params_t *out_params) {
  memset(out_params, 0, sizeof(*out_params));
  out_params->arena_block_size = 32 * 1024;
}

// Validates |params| before creating a CoralNPU device.
static iree_status_t iree_hal_coralnpu_device_check_params(
    const iree_hal_coralnpu_device_params_t *params) {
  if (params->arena_block_size < 4096) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "arena block size too small (< 4096 bytes)");
  }
  if (params->dispatch_observer.begin_dispatch &&
      !params->dispatch_observer.end_dispatch) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "dispatch observer with begin_dispatch requires end_dispatch");
  }
  return iree_ok_status();
}

iree_status_t iree_hal_coralnpu_device_create(
    iree_string_view_t identifier,
    const iree_hal_coralnpu_device_params_t *params,
    const iree_hal_device_create_params_t *create_params,
    const iree_hal_coralnpu_exec_backend_t *exec_backend,
    iree_hal_allocator_t *device_allocator, iree_allocator_t host_allocator,
    iree_hal_device_t **out_device) {
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(create_params);
  IREE_ASSERT_ARGUMENT(create_params->proactor_pool);
  IREE_ASSERT_ARGUMENT(exec_backend);
  IREE_ASSERT_ARGUMENT(device_allocator);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_coralnpu_device_check_params(params));

  iree_hal_coralnpu_device_t *device = NULL;
  iree_host_size_t total_size = 0;
  iree_host_size_t identifier_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(sizeof(*device), &total_size,
                             IREE_STRUCT_FIELD_ALIGNED(identifier.size, char, 1,
                                                       &identifier_offset)));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc_aligned(
              host_allocator, total_size,
              iree_alignof(iree_hal_coralnpu_device_t), 0, (void **)&device));
  iree_hal_resource_initialize(&iree_hal_coralnpu_device_vtable,
                               &device->resource);
  iree_string_view_append_to_buffer(identifier, &device->identifier,
                                    (char *)device + identifier_offset);
  device->host_allocator = host_allocator;
  device->device_allocator = device_allocator;
  device->exec_backend = *exec_backend;
  // Use the observer only if it observes dispatches; without begin_dispatch,
  // its create and destroy hooks are not called.
  if (params->dispatch_observer.begin_dispatch) {
    device->dispatch_observer = params->dispatch_observer;
  }
  // Let the observer create its per-device context.
  if (device->dispatch_observer.create) {
    iree_status_t status = device->dispatch_observer.create(
        device->dispatch_observer.self, device->exec_backend.name,
        host_allocator, &device->dispatch_observer_context);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free_aligned(host_allocator, device);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
  }
  iree_hal_allocator_retain(device_allocator);
  iree_arena_block_pool_initialize(params->arena_block_size, host_allocator,
                                   &device->large_block_pool);

  device->proactor_pool = create_params->proactor_pool;
  iree_async_proactor_pool_retain(device->proactor_pool);
  iree_atomic_store(&device->epoch, 0, iree_memory_order_relaxed);
  iree_status_t status =
      iree_async_proactor_pool_get(device->proactor_pool, 0, &device->proactor);
  if (iree_status_is_ok(status)) {
    *out_device = (iree_hal_device_t *)device;
  } else {
    iree_hal_device_release((iree_hal_device_t *)device);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_coralnpu_device_clear_topology_info(
    iree_hal_coralnpu_device_t *device) {
  if (device->frontier_tracker) {
    iree_async_frontier_tracker_retire_axis(
        device->frontier_tracker, device->axis,
        iree_status_from_code(IREE_STATUS_CANCELLED));
    iree_async_frontier_tracker_release(device->frontier_tracker);
    device->frontier_tracker = NULL;
    device->axis = 0;
  }
  memset(&device->topology_info, 0, sizeof(device->topology_info));
}

// Destroys the device when its last reference is released.
static void iree_hal_coralnpu_device_destroy(iree_hal_device_t *base_device) {
  iree_hal_coralnpu_device_t *device =
      iree_hal_coralnpu_device_cast(base_device);
  iree_allocator_t host_allocator = device->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Lets the observer flush its output (e.g. a summary of the dispatches).
  if (device->dispatch_observer.destroy) {
    device->dispatch_observer.destroy(device->dispatch_observer.self,
                                      host_allocator,
                                      device->dispatch_observer_context);
  }

  iree_hal_coralnpu_device_clear_topology_info(device);
  iree_hal_allocator_release(device->device_allocator);
  iree_hal_channel_provider_release(device->channel_provider);
  iree_async_proactor_pool_release(device->proactor_pool);

  iree_arena_block_pool_deinitialize(&device->large_block_pool);

  iree_allocator_free_aligned(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
}

static iree_string_view_t iree_hal_coralnpu_device_id(
    iree_hal_device_t *base_device) {
  iree_hal_coralnpu_device_t *device =
      iree_hal_coralnpu_device_cast(base_device);
  return device->identifier;
}

static iree_allocator_t iree_hal_coralnpu_device_host_allocator(
    iree_hal_device_t *base_device) {
  iree_hal_coralnpu_device_t *device =
      iree_hal_coralnpu_device_cast(base_device);
  return device->host_allocator;
}

static iree_hal_allocator_t *iree_hal_coralnpu_device_allocator(
    iree_hal_device_t *base_device) {
  iree_hal_coralnpu_device_t *device =
      iree_hal_coralnpu_device_cast(base_device);
  return device->device_allocator;
}

static void iree_hal_coralnpu_replace_device_allocator(
    iree_hal_device_t *base_device, iree_hal_allocator_t *new_allocator) {
  iree_hal_coralnpu_device_t *device =
      iree_hal_coralnpu_device_cast(base_device);
  iree_hal_allocator_retain(new_allocator);
  iree_hal_allocator_release(device->device_allocator);
  device->device_allocator = new_allocator;
}

static void iree_hal_coralnpu_replace_channel_provider(
    iree_hal_device_t *base_device, iree_hal_channel_provider_t *new_provider) {
  iree_hal_coralnpu_device_t *device =
      iree_hal_coralnpu_device_cast(base_device);
  iree_hal_channel_provider_retain(new_provider);
  iree_hal_channel_provider_release(device->channel_provider);
  device->channel_provider = new_provider;
}

static iree_status_t iree_hal_coralnpu_device_trim(
    iree_hal_device_t *base_device) {
  iree_hal_coralnpu_device_t *device =
      iree_hal_coralnpu_device_cast(base_device);
  return iree_hal_allocator_trim(device->device_allocator);
}

static iree_status_t iree_hal_coralnpu_device_query_i64(
    iree_hal_device_t *base_device, iree_string_view_t category,
    iree_string_view_t key, int64_t *out_value) {
  iree_hal_coralnpu_device_t *device =
      iree_hal_coralnpu_device_cast(base_device);
  *out_value = 0;

  if (iree_string_view_equal(category, IREE_SV("hal.device.id"))) {
    *out_value =
        iree_string_view_match_pattern(device->identifier, key) ? 1 : 0;
    return iree_ok_status();
  }

  if (iree_string_view_equal(category, IREE_SV("hal.executable.format"))) {
    *out_value = iree_hal_coralnpu_is_executable_format(key) ? 1 : 0;
    return iree_ok_status();
  }

  if (iree_string_view_equal(category, IREE_SV("hal.device")) ||
      iree_string_view_equal(category, IREE_SV("hal.dispatch"))) {
    if (iree_string_view_equal(key, IREE_SV("concurrency"))) {
      *out_value = 1;
      return iree_ok_status();
    }
  }

  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "unknown device configuration key value '%.*s :: %.*s'",
      (int)category.size, category.data, (int)key.size, key.data);
}

static iree_status_t iree_hal_coralnpu_device_query_capabilities(
    iree_hal_device_t *base_device,
    iree_hal_device_capabilities_t *out_capabilities) {
  memset(out_capabilities, 0, sizeof(*out_capabilities));
  return iree_ok_status();
}

static const iree_hal_device_topology_info_t *
iree_hal_coralnpu_device_topology_info(iree_hal_device_t *base_device) {
  iree_hal_coralnpu_device_t *device =
      iree_hal_coralnpu_device_cast(base_device);
  return &device->topology_info;
}

static iree_status_t iree_hal_coralnpu_device_refine_topology_edge(
    iree_hal_device_t *src_device, iree_hal_device_t *dst_device,
    iree_hal_topology_edge_t *edge) {
  return iree_ok_status();
}

static iree_status_t iree_hal_coralnpu_device_assign_topology_info(
    iree_hal_device_t *base_device,
    const iree_hal_device_topology_info_t *topology_info) {
  iree_hal_coralnpu_device_t *device =
      iree_hal_coralnpu_device_cast(base_device);
  iree_hal_coralnpu_device_clear_topology_info(device);
  if (!topology_info) {
    return iree_ok_status();
  }
  iree_async_frontier_tracker_t *frontier_tracker =
      topology_info->frontier.tracker;
  iree_async_axis_t axis = topology_info->frontier.base_axis;
  if (frontier_tracker) {
    IREE_RETURN_IF_ERROR(iree_async_frontier_tracker_register_axis(
        frontier_tracker, axis, /*semaphore=*/NULL));
    iree_async_frontier_tracker_retain(frontier_tracker);
  }
  device->topology_info = *topology_info;
  device->frontier_tracker = frontier_tracker;
  device->axis = axis;
  return iree_ok_status();
}

static iree_status_t iree_hal_coralnpu_device_create_channel(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t **out_channel) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "collectives not implemented");
}

static iree_status_t iree_hal_coralnpu_device_create_command_buffer(
    iree_hal_device_t *base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t **out_command_buffer) {
  iree_hal_coralnpu_device_t *device =
      iree_hal_coralnpu_device_cast(base_device);
  if (iree_all_bits_set(mode,
                        IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION)) {
    return iree_hal_coralnpu_command_buffer_create(
        base_device, device->device_allocator, mode, command_categories,
        queue_affinity, binding_capacity, device->host_allocator,
        out_command_buffer);
  }
  return iree_hal_deferred_command_buffer_create(
      device->device_allocator, mode, command_categories, queue_affinity,
      binding_capacity, &device->large_block_pool, device->host_allocator,
      out_command_buffer);
}

static iree_status_t iree_hal_coralnpu_device_create_event(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_event_flags_t flags, iree_hal_event_t **out_event) {
  return iree_hal_sync_event_create(queue_affinity, flags,
                                    iree_hal_device_host_allocator(base_device),
                                    out_event);
}

static iree_status_t iree_hal_coralnpu_device_create_executable_cache(
    iree_hal_device_t *base_device, iree_string_view_t identifier,
    iree_hal_executable_cache_t **out_executable_cache) {
  return iree_hal_coralnpu_executable_cache_create(
      iree_hal_device_host_allocator(base_device), out_executable_cache);
}

static iree_status_t iree_hal_coralnpu_device_import_file(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t *handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t **out_file) {
  return iree_hal_file_from_handle(
      iree_hal_device_allocator(base_device), queue_affinity, access, handle,
      /*proactor=*/NULL, iree_hal_device_host_allocator(base_device), out_file);
}

static iree_status_t iree_hal_coralnpu_device_create_semaphore(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t **out_semaphore) {
  iree_hal_coralnpu_device_t *device =
      iree_hal_coralnpu_device_cast(base_device);
  return iree_hal_sync_semaphore_create(device->proactor, queue_affinity,
                                        initial_value, flags,
                                        device->host_allocator, out_semaphore);
}

static iree_hal_semaphore_compatibility_t
iree_hal_coralnpu_device_query_semaphore_compatibility(
    iree_hal_device_t *base_device, iree_hal_semaphore_t *semaphore) {
  // The synchronous submission queue handles all semaphores as if host-side.
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_ONLY;
}

static iree_status_t iree_hal_coralnpu_device_query_queue_pool_backend(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_queue_pool_backend_t *out_backend) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "queue pool backend not implemented");
}

// Waits for all semaphore dependencies before a queue operation body.
static inline iree_status_t iree_hal_coralnpu_device_queue_op_begin(
    iree_hal_coralnpu_device_t *device,
    const iree_hal_semaphore_list_t wait_semaphore_list) {
  return iree_hal_semaphore_list_wait(
      wait_semaphore_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
}

// Signals semaphores after a queue operation body completes (or fails them on
// error) and advances the frontier tracker.
static inline iree_status_t iree_hal_coralnpu_device_queue_op_end(
    iree_hal_coralnpu_device_t *device,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_status_t status) {
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_signal(signal_semaphore_list,
                                            /*frontier=*/NULL);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_coralnpu_device_advance_frontier(device);
  } else {
    iree_hal_semaphore_list_fail(signal_semaphore_list,
                                 iree_status_clone(status));
    if (device->frontier_tracker) {
      iree_async_frontier_tracker_fail_axis(
          device->frontier_tracker, device->axis,
          iree_status_from_code(iree_status_code(status)));
    }
  }
  return status;
}

static iree_status_t iree_hal_coralnpu_device_queue_alloca(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t *pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t **IREE_RESTRICT out_buffer) {
  iree_hal_coralnpu_device_t *device =
      iree_hal_coralnpu_device_cast(base_device);
  // TODO(benvanik): queue-ordered allocations.
  IREE_RETURN_IF_ERROR(
      iree_hal_coralnpu_device_queue_op_begin(device, wait_semaphore_list));
  iree_hal_buffer_t *buffer = NULL;
  iree_status_t status = iree_hal_allocator_allocate_buffer(
      device->device_allocator, params, allocation_size, &buffer);
  status = iree_hal_coralnpu_device_queue_op_end(device, signal_semaphore_list,
                                                 status);
  if (iree_status_is_ok(status)) {
    *out_buffer = buffer;
  } else {
    iree_hal_buffer_release(buffer);
  }
  return status;
}

static iree_status_t iree_hal_coralnpu_device_queue_dealloca(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t *buffer, iree_hal_dealloca_flags_t flags) {
  // TODO(benvanik): queue-ordered allocations.
  return iree_hal_device_queue_barrier(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_EXECUTE_FLAG_NONE);
}

static iree_status_t iree_hal_coralnpu_device_queue_read(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t *source_file, uint64_t source_offset,
    iree_hal_buffer_t *target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  // TODO: expose streaming chunk count/size options.
  iree_hal_file_transfer_options_t options = {
      .chunk_count = IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
      .chunk_size = IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
  };
  return iree_hal_device_queue_read_streaming(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      source_file, source_offset, target_buffer, target_offset, length, flags,
      options);
}

static iree_status_t iree_hal_coralnpu_device_queue_write(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t *source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t *target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  // TODO: expose streaming chunk count/size options.
  iree_hal_file_transfer_options_t options = {
      .chunk_count = IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
      .chunk_size = IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
  };
  return iree_hal_device_queue_write_streaming(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      source_buffer, source_offset, target_file, target_offset, length, flags,
      options);
}

static iree_status_t iree_hal_coralnpu_device_queue_host_call(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  iree_hal_coralnpu_device_t *device =
      iree_hal_coralnpu_device_cast(base_device);
  IREE_RETURN_IF_ERROR(
      iree_hal_coralnpu_device_queue_op_begin(device, wait_semaphore_list));

  // If non-blocking then immediately signal the dependencies instead of letting
  // the call do it. We don't expect this to allow more work to proceed in the
  // coralnpu device case _on this device_ but it may on others.
  const bool is_nonblocking =
      iree_any_bit_set(flags, IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING);
  if (is_nonblocking) {
    // NOTE: the signals can fail in which case we never perform the call.
    // That's ok as failure to signal is considered a device-loss/death
    // situation as there's no telling what has gone wrong.
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_signal(signal_semaphore_list,
                                                        /*frontier=*/NULL));
    iree_hal_coralnpu_device_advance_frontier(device);
  }

  // Issue the call.
  iree_hal_host_call_context_t context = {
      .device = base_device,
      .queue_affinity = queue_affinity,
      .signal_semaphore_list = is_nonblocking ? iree_hal_semaphore_list_empty()
                                              : signal_semaphore_list,
  };
  iree_status_t call_status = call.fn(call.user_data, args, &context);

  if (is_nonblocking || iree_status_is_deferred(call_status)) {
    // User callback will signal in the future (or they are fire-and-forget).
    iree_status_free(call_status);
    return iree_ok_status();
  } else if (iree_status_is_ok(call_status)) {
    // Signal callback completed synchronously.
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_signal(signal_semaphore_list,
                                                        /*frontier=*/NULL));
    iree_hal_coralnpu_device_advance_frontier(device);
    return iree_ok_status();
  } else {
    // If the call failed we need to fail all dependent semaphores to propagate
    // the error.
    if (device->frontier_tracker) {
      iree_async_frontier_tracker_fail_axis(
          device->frontier_tracker, device->axis,
          iree_status_from_code(iree_status_code(call_status)));
    }
    iree_hal_semaphore_list_fail(signal_semaphore_list, call_status);
    return iree_ok_status();
  }
}

static iree_status_t iree_hal_coralnpu_device_apply_deferred_command_buffer(
    iree_hal_coralnpu_device_t *device,
    iree_hal_command_buffer_t *command_buffer,
    iree_hal_buffer_binding_table_t binding_table) {
  // If there were no deferred command buffers no-op this call - they've already
  // been issued.
  if (!command_buffer ||
      !iree_hal_deferred_command_buffer_isa(command_buffer)) {
    return iree_ok_status();
  }

  // NOTE: we need to validate if a binding table is provided as the bindings
  // were not known when the deferred command buffer was originally recorded.
  // Otherwise we run unvalidated as inline command buffers don't support
  // binding tables and can be validated entirely while recording.
  const iree_hal_command_buffer_mode_t mode =
      iree_hal_command_buffer_mode(command_buffer) |
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
      IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION |
      (iree_hal_buffer_binding_table_is_empty(binding_table)
           ? IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED
           : 0);

  // Stack allocate storage for an inline command buffer we'll use to replay
  // the deferred command buffers.
  iree_host_size_t storage_size =
      iree_hal_coralnpu_command_buffer_size(mode, /*binding_capacity=*/0);
  iree_byte_span_t storage =
      iree_make_byte_span(iree_alloca(storage_size), storage_size);

  iree_hal_command_buffer_t *coralnpu_command_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_coralnpu_command_buffer_initialize(
      (iree_hal_device_t *)device, device->device_allocator, mode,
      iree_hal_command_buffer_allowed_categories(command_buffer),
      IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, device->host_allocator, storage,
      &coralnpu_command_buffer));

  return iree_hal_deferred_command_buffer_apply(
      command_buffer, coralnpu_command_buffer, binding_table);
}

static iree_status_t iree_hal_coralnpu_device_queue_execute(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t *command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  iree_hal_coralnpu_device_t *device =
      iree_hal_coralnpu_device_cast(base_device);

  // Wait for semaphores to be signaled before performing any work.
  IREE_RETURN_IF_ERROR(
      iree_hal_coralnpu_device_queue_op_begin(device, wait_semaphore_list));

  // Run all deferred command buffers - any we could have run inline we already
  // did during recording.
  iree_status_t status = iree_hal_coralnpu_device_apply_deferred_command_buffer(
      device, command_buffer, binding_table);
  return iree_hal_coralnpu_device_queue_op_end(device, signal_semaphore_list,
                                               status);
}

static iree_status_t iree_hal_coralnpu_device_queue_flush(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity) {
  // Currently unused; we flush as submissions are made.
  return iree_ok_status();
}

static iree_status_t iree_hal_coralnpu_device_profiling_begin(
    iree_hal_device_t *base_device,
    const iree_hal_device_profiling_options_t *options) {
  // Unimplemented (and that's ok).
  return iree_ok_status();
}

static iree_status_t iree_hal_coralnpu_device_profiling_flush(
    iree_hal_device_t *base_device) {
  // Unimplemented (and that's ok).
  return iree_ok_status();
}

static iree_status_t iree_hal_coralnpu_device_profiling_end(
    iree_hal_device_t *base_device) {
  // Unimplemented (and that's ok).
  return iree_ok_status();
}

static iree_status_t iree_hal_coralnpu_device_external_capture_begin(
    iree_hal_device_t *base_device,
    const iree_hal_device_external_capture_options_t *options) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "coralnpu external capture not implemented");
}

static iree_status_t iree_hal_coralnpu_device_external_capture_end(
    iree_hal_device_t *base_device) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "coralnpu external capture not implemented");
}

static const iree_hal_device_vtable_t iree_hal_coralnpu_device_vtable = {
    .destroy = iree_hal_coralnpu_device_destroy,
    .id = iree_hal_coralnpu_device_id,
    .host_allocator = iree_hal_coralnpu_device_host_allocator,
    .device_allocator = iree_hal_coralnpu_device_allocator,
    .replace_device_allocator = iree_hal_coralnpu_replace_device_allocator,
    .replace_channel_provider = iree_hal_coralnpu_replace_channel_provider,
    .trim = iree_hal_coralnpu_device_trim,
    .query_i64 = iree_hal_coralnpu_device_query_i64,
    .query_capabilities = iree_hal_coralnpu_device_query_capabilities,
    .topology_info = iree_hal_coralnpu_device_topology_info,
    .refine_topology_edge = iree_hal_coralnpu_device_refine_topology_edge,
    .assign_topology_info = iree_hal_coralnpu_device_assign_topology_info,
    .create_channel = iree_hal_coralnpu_device_create_channel,
    .create_command_buffer = iree_hal_coralnpu_device_create_command_buffer,
    .create_event = iree_hal_coralnpu_device_create_event,
    .create_executable_cache = iree_hal_coralnpu_device_create_executable_cache,
    .import_file = iree_hal_coralnpu_device_import_file,
    .create_semaphore = iree_hal_coralnpu_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_coralnpu_device_query_semaphore_compatibility,
    .query_queue_pool_backend =
        iree_hal_coralnpu_device_query_queue_pool_backend,
    .queue_alloca = iree_hal_coralnpu_device_queue_alloca,
    .queue_dealloca = iree_hal_coralnpu_device_queue_dealloca,
    .queue_fill = iree_hal_device_queue_emulated_fill,
    .queue_update = iree_hal_device_queue_emulated_update,
    .queue_copy = iree_hal_device_queue_emulated_copy,
    .queue_read = iree_hal_coralnpu_device_queue_read,
    .queue_write = iree_hal_coralnpu_device_queue_write,
    .queue_host_call = iree_hal_coralnpu_device_queue_host_call,
    .queue_dispatch = iree_hal_device_queue_emulated_dispatch,
    .queue_execute = iree_hal_coralnpu_device_queue_execute,
    .queue_flush = iree_hal_coralnpu_device_queue_flush,
    .profiling_begin = iree_hal_coralnpu_device_profiling_begin,
    .profiling_flush = iree_hal_coralnpu_device_profiling_flush,
    .profiling_end = iree_hal_coralnpu_device_profiling_end,
    .external_capture_begin = iree_hal_coralnpu_device_external_capture_begin,
    .external_capture_end = iree_hal_coralnpu_device_external_capture_end,
};
