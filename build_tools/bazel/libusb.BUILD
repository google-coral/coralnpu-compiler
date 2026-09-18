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

# Linux POSIX configuration using the kernel netlink backend instead of libudev
# so that libusb can be linked both dynamically and statically (-static).
genrule(
    name = "config_h",
    outs = ["libusb/config.h"],
    cmd = """cat << 'EOF' > $@
#ifndef LIBUSB_CONFIG_H
#define LIBUSB_CONFIG_H

#define _GNU_SOURCE 1
#define DEFAULT_VISIBILITY __attribute__((visibility("default")))
#define PRINTF_FORMAT(a, b) __attribute__((__format__(__printf__, a, b)))

#define ENABLE_LOGGING 1
#define PLATFORM_POSIX 1

#define HAVE_ASM_TYPES_H 1
#define HAVE_CLOCK_GETTIME 1
#define HAVE_DECL_EFD_CLOEXEC 1
#define HAVE_DECL_EFD_NONBLOCK 1
#define HAVE_DECL_TFD_CLOEXEC 1
#define HAVE_DECL_TFD_NONBLOCK 1
#define HAVE_EVENTFD 1
#define HAVE_INTTYPES_H 1
#define HAVE_NFDS_T 1
#define HAVE_PIPE2 1
#define HAVE_PTHREAD_CONDATTR_SETCLOCK 1
#define HAVE_PTHREAD_SETNAME_NP 1
#define HAVE_STDINT_H 1
#define HAVE_STDIO_H 1
#define HAVE_STDLIB_H 1
#define HAVE_STRINGS_H 1
#define HAVE_STRING_H 1
#define HAVE_STRUCT_TIMESPEC 1
#define HAVE_SYSLOG 1
#define HAVE_SYS_STAT_H 1
#define HAVE_SYS_TIME_H 1
#define HAVE_SYS_TYPES_H 1
#define HAVE_TIMERFD 1
#define HAVE_UNISTD_H 1

#endif  // LIBUSB_CONFIG_H
EOF
""",
)

cc_library(
    name = "libusb",
    srcs = [
        "libusb/core.c",
        "libusb/descriptor.c",
        "libusb/hotplug.c",
        "libusb/io.c",
        "libusb/os/events_posix.c",
        "libusb/os/linux_netlink.c",
        "libusb/os/linux_usbfs.c",
        "libusb/os/threads_posix.c",
        "libusb/strerror.c",
        "libusb/sync.c",
        ":config_h",
    ] + glob([
        "libusb/*.h",
        "libusb/os/*.h",
    ]),
    hdrs = ["libusb/libusb.h"],
    includes = ["libusb"],
    linkopts = ["-pthread"],
)
