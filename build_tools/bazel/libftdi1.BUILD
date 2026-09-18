# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["restricted"],  # LGPL-2.1
)

genrule(
    name = "ftdi_version_i_h",
    srcs = ["src/ftdi_version_i.h.in"],
    outs = ["src/ftdi_version_i.h"],
    cmd = """sed \
        -e 's/@MAJOR_VERSION@/1/g' \
        -e 's/@MINOR_VERSION@/5/g' \
        -e 's/@MICRO_VERSION@/0/g' \
        -e 's/@VERSION_STRING@/1.5/g' \
        -e 's/@SNAPSHOT_VERSION@/unknown/g' \
        $< > $@""",
)

cc_library(
    name = "libftdi1",
    srcs = [
        "src/ftdi.c",
        "src/ftdi_i.h",
        "src/ftdi_stream.c",
        ":ftdi_version_i_h",
    ],
    hdrs = ["src/ftdi.h"],
    includes = ["src"],
    deps = ["@libusb"],
)
