# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

if(TARGET coralnpu_runtime::sim::simulator_mpact)
  # Registers the driver with IREE, selects the execution backend it runs on
  # and sets up the statistics writer of the --coralnpu_stats_* flags
  # (coralnpu_metrics_runtime). The simulator and metrics dependencies are
  # confined to this target so that the HAL driver in
  # coralnpu_runtime::driver::coralnpu stays unaware of them.
  iree_cc_library(
    NAME
      registration
    HDRS
      "driver_module.h"
    SRCS
      "driver_module.c"
    DEPS
      coralnpu_metrics_runtime::stats_flags
      coralnpu_runtime::driver::coralnpu
      coralnpu_runtime::sim::simulation
      coralnpu_runtime::sim::simulator_mpact
      iree::base
      iree::base::internal::dynamic_library
      iree::base::tooling::flags
      iree::hal
    PUBLIC
  )
endif()
