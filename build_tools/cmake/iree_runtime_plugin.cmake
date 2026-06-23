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

set(IREE_CORALNPU_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

set(IREE_ENABLE_CORALNPU_DRIVER OFF)

if("coralnpu" IN_LIST IREE_EXTERNAL_HAL_DRIVERS)
  message(
    STATUS "Enabling coralnpu build because it is an enabled HAL driver")
  set(IREE_ENABLE_CORALNPU_DRIVER ON)
endif()

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../runtime coralnpu_runtime)
