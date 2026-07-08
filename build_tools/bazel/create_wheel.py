#!/usr/bin/env python3
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
"""Script to create a Python wheel (.whl) from a staged package directory."""

import argparse
import base64
import hashlib
import os
import sys
import zipfile


def urlsafe_b64encode(data: bytes) -> str:
  return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def main():
  parser = argparse.ArgumentParser(description="Create a python wheel.")
  parser.add_argument("--package-dir",
                      required=True,
                      help="Directory to package")
  parser.add_argument("--dist-name",
                      required=True,
                      help="Distribution name (e.g. coralnpu_compiler)")
  parser.add_argument("--version", required=True, help="Version (e.g. 0.0.1)")
  parser.add_argument("--output", required=True, help="Output .whl file path")
  parser.add_argument("--summary", default="CoralNPU Package", help="Summary")
  parser.add_argument(
      "--requires-dist",
      action="append",
      default=[],
      help="Package dependency (Requires-Dist)",
  )
  parser.add_argument(
      "--py-tag",
      default="py3-none-any",
      help=
      "Wheel platform tag (e.g. py3-none-any or py3-none-manylinux_2_28_x86_64)",
  )
  args = parser.parse_args()

  dist_info_dir = f"{args.dist_name}-{args.version}.dist-info"

  # For platform-specific C/C++ extension wheels, set platform-specific wheel tag:
  # e.g., py3-none-manylinux_2_28_x86_64, py3-none-manylinux_2_28_aarch64, py3-none-macosx_11_0_arm64, py3-none-win_amd64
  wheel_metadata = ("Wheel-Version: 1.0\n"
                    "Generator: coralnpu_wheelmaker\n"
                    "Root-Is-Purelib: false\n"
                    f"Tag: {args.py_tag}\n")

  pkg_metadata_lines = [
      "Metadata-Version: 2.1",
      f"Name: {args.dist_name}",
      f"Version: {args.version}",
      f"Summary: {args.summary}",
  ]
  for req in args.requires_dist:
    pkg_metadata_lines.append(f"Requires-Dist: {req}")
  pkg_metadata = "\n".join(pkg_metadata_lines) + "\n"

  record_lines = []

  with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_STORED) as zf:
    # 1. Package all files from package_dir
    for root, _, files in os.walk(args.package_dir):
      for file in files:
        abs_path = os.path.join(root, file)
        rel_path = os.path.relpath(abs_path, args.package_dir)

        with open(abs_path, "rb") as f:
          data = f.read()

        sha256_hash = hashlib.sha256(data).digest()
        b64_hash = urlsafe_b64encode(sha256_hash)
        file_size = len(data)

        st_mode = os.stat(abs_path).st_mode
        zip_info = zipfile.ZipInfo(rel_path)
        zip_info.external_attr = st_mode << 16

        # Write to zip
        print(f"Adding file to wheel: {rel_path} ({file_size} bytes)",
              file=sys.stderr)
        zf.writestr(zip_info, data)
        record_lines.append(f"{rel_path},sha256={b64_hash},{file_size}")

    # 2. Package dist-info files
    for name, content in [("WHEEL", wheel_metadata),
                          ("METADATA", pkg_metadata)]:
      path_in_wheel = f"{dist_info_dir}/{name}"
      data = content.encode("utf-8")
      sha256_hash = hashlib.sha256(data).digest()
      b64_hash = urlsafe_b64encode(sha256_hash)
      zf.writestr(path_in_wheel, data)
      record_lines.append(f"{path_in_wheel},sha256={b64_hash},{len(data)}")

    # 3. Write RECORD
    record_path = f"{dist_info_dir}/RECORD"
    record_lines.append(f"{record_path},,")
    zf.writestr(record_path, "\n".join(record_lines) + "\n")

  print(f"Wheel successfully created: {args.output}")


if __name__ == "__main__":
  main()
