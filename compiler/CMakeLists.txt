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

iree_setup_c_src_root(
  PACKAGE_ROOT_PREFIX
    coralnpu_compiler
)

add_subdirectory(Pipelines)

iree_cc_library(
  NAME
    registration
  SRCS
    PluginRegistration.cpp
  DEPS
    coralnpu_compiler::Pipelines
    iree::compiler::Dialect::HAL::Target
    iree::compiler::PluginAPI
    MLIRIR
  PUBLIC
  ALWAYSLINK
)

iree_compiler_register_plugin(
  PLUGIN_ID
    coralnpu
  TARGET
    ::registration
)
