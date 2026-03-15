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
    "coralnpu_runtime"
  DEFAULT_EXPORT_SET
    Runtime
  DEFAULT_INSTALL_COMPONENT
    IREEDevLibraries-Runtime
)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

if(IREE_ENABLE_CORALNPU_DRIVER)
  add_subdirectory(sim)
  add_subdirectory(driver)
endif()
