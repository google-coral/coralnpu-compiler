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

#include "runtime/driver/coralnpu_semaphore.h"

#include "iree/hal/drivers/local_sync/sync_semaphore.h"

void iree_hal_coralnpu_semaphore_state_initialize(
    iree_hal_coralnpu_semaphore_state_t *out_shared_state) {
  iree_hal_sync_semaphore_state_initialize(
      (iree_hal_sync_semaphore_state_t *)out_shared_state);
}

void iree_hal_coralnpu_semaphore_state_deinitialize(
    iree_hal_coralnpu_semaphore_state_t *shared_state) {
  iree_hal_sync_semaphore_state_deinitialize(
      (iree_hal_sync_semaphore_state_t *)shared_state);
}

iree_status_t iree_hal_coralnpu_semaphore_create(
    iree_hal_coralnpu_semaphore_state_t *shared_state, uint64_t initial_value,
    iree_allocator_t host_allocator, iree_hal_semaphore_t **out_semaphore) {
  return iree_hal_sync_semaphore_create(
      (iree_hal_sync_semaphore_state_t *)shared_state, initial_value,
      host_allocator, out_semaphore);
}

iree_status_t iree_hal_coralnpu_semaphore_multi_signal(
    iree_hal_coralnpu_semaphore_state_t *shared_state,
    const iree_hal_semaphore_list_t semaphore_list) {
  return iree_hal_sync_semaphore_multi_signal(
      (iree_hal_sync_semaphore_state_t *)shared_state, semaphore_list);
}

iree_status_t iree_hal_coralnpu_semaphore_multi_wait(
    iree_hal_coralnpu_semaphore_state_t *shared_state,
    iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout,
    iree_hal_wait_flags_t flags) {
  return iree_hal_sync_semaphore_multi_wait(
      (iree_hal_sync_semaphore_state_t *)shared_state, wait_mode,
      semaphore_list, timeout, flags);
}
