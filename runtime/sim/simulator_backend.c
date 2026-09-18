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

#include "runtime/sim/simulator_backend.h"

#include "iree/base/threading/call_once.h"
#include "iree/base/threading/mutex.h"
#include "runtime/sim/simulator_inline.h"

// Serializes dispatches, which share process-wide state: the cached simulator
// instance and the read-only binding and ELF layout caches.
static iree_once_flag iree_hal_coralnpu_simulator_mutex_once =
    IREE_ONCE_FLAG_INIT;
static iree_slim_mutex_t iree_hal_coralnpu_simulator_mutex;

static void iree_hal_coralnpu_simulator_mutex_initialize(void) {
  iree_slim_mutex_initialize(&iree_hal_coralnpu_simulator_mutex);
}

// MPACT and FPGA reuse one simulator across dispatches; other backends (Spike,
// Verilator) get a fresh one per dispatch.
static iree_status_t iree_hal_coralnpu_simulator_backend_dispatch_locked(
    coralnpu_simulator_create_fn_t factory, bool reuse_sim,
    iree_const_byte_span_t dispatch_image,
    const iree_hal_executable_dispatch_state_v0_t *dispatch_state,
    const bool *binding_writeable, iree_host_size_t ordinal,
    iree_host_size_t local_memory_size,
    iree_hal_coralnpu_dispatch_stats_t *out_stats) {
  static coralnpu_simulator_create_fn_t cached_factory = NULL;
  static coralnpu_simulator_t *cached_sim = NULL;
  if (reuse_sim && cached_sim && cached_factory != factory) {
    iree_hal_simulator_inline_forget(cached_sim);
    coralnpu_simulator_destroy(cached_sim);
    cached_sim = NULL;
    cached_factory = NULL;
  }
  coralnpu_simulator_t *sim = reuse_sim && cached_sim ? cached_sim : factory();
  // The FPGA factory returns NULL when no device is connected.
  if (!sim) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "failed to initialize CoralNPU device backend "
                            "(hardware device not connected or unavailable)");
  }
  if (reuse_sim) {
    cached_factory = factory;
    cached_sim = sim;
  }
  iree_status_t status = iree_hal_simulator_issue_dispatch_inline(
      sim, dispatch_image, dispatch_state, binding_writeable, ordinal,
      local_memory_size, out_stats);
  if (reuse_sim && iree_status_is_ok(status)) return status;
  if (reuse_sim) {
    cached_factory = NULL;
    cached_sim = NULL;
  }
  iree_hal_simulator_inline_forget(sim);
  coralnpu_simulator_destroy(sim);
  return status;
}

static iree_status_t iree_hal_coralnpu_simulator_backend_dispatch_with_policy(
    void *self, bool reuse_sim, iree_const_byte_span_t dispatch_image,
    const iree_hal_executable_dispatch_state_v0_t *dispatch_state,
    const bool *binding_writeable, iree_host_size_t ordinal,
    iree_host_size_t local_memory_size,
    iree_hal_coralnpu_dispatch_stats_t *out_stats) {
  iree_call_once(&iree_hal_coralnpu_simulator_mutex_once,
                 iree_hal_coralnpu_simulator_mutex_initialize);
  iree_slim_mutex_lock(&iree_hal_coralnpu_simulator_mutex);
  iree_status_t status = iree_hal_coralnpu_simulator_backend_dispatch_locked(
      (coralnpu_simulator_create_fn_t)self, reuse_sim, dispatch_image,
      dispatch_state, binding_writeable, ordinal, local_memory_size, out_stats);
  iree_slim_mutex_unlock(&iree_hal_coralnpu_simulator_mutex);
  return status;
}

static iree_status_t iree_hal_coralnpu_simulator_backend_dispatch_reused(
    void *self, iree_const_byte_span_t dispatch_image,
    const iree_hal_executable_dispatch_state_v0_t *dispatch_state,
    const bool *binding_writeable, iree_host_size_t ordinal,
    iree_host_size_t local_memory_size,
    iree_hal_coralnpu_dispatch_stats_t *out_stats) {
  return iree_hal_coralnpu_simulator_backend_dispatch_with_policy(
      self, /*reuse_sim=*/true, dispatch_image, dispatch_state,
      binding_writeable, ordinal, local_memory_size, out_stats);
}

static iree_status_t iree_hal_coralnpu_simulator_backend_dispatch_fresh(
    void *self, iree_const_byte_span_t dispatch_image,
    const iree_hal_executable_dispatch_state_v0_t *dispatch_state,
    const bool *binding_writeable, iree_host_size_t ordinal,
    iree_host_size_t local_memory_size,
    iree_hal_coralnpu_dispatch_stats_t *out_stats) {
  return iree_hal_coralnpu_simulator_backend_dispatch_with_policy(
      self, /*reuse_sim=*/false, dispatch_image, dispatch_state,
      binding_writeable, ordinal, local_memory_size, out_stats);
}

iree_hal_coralnpu_exec_backend_t iree_hal_coralnpu_simulator_backend_make(
    coralnpu_simulator_create_fn_t factory, iree_string_view_t name) {
  const bool reuse_sim = iree_string_view_equal(name, IREE_SV("mpact")) ||
                         iree_string_view_equal(name, IREE_SV("fpga"));
  iree_hal_coralnpu_exec_backend_t backend = {
      .self = (void *)factory,
      .name = name,
      .dispatch = reuse_sim
                      ? iree_hal_coralnpu_simulator_backend_dispatch_reused
                      : iree_hal_coralnpu_simulator_backend_dispatch_fresh,
  };
  return backend;
}
