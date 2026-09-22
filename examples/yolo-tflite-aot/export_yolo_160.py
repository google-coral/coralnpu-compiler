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

import os

from tosa_converter_for_tflite import (
    TosaConverterOutputFormat,
    tflite_flatbuffer_to_tosa_mlir,
)
from examples import utils


def main():
  args = utils.parse_export_args()
  script_dir = os.path.dirname(os.path.abspath(__file__))
  tflite_path = os.path.join(script_dir, "yolov8n_160.tflite")
  tflite_flatbuffer_to_tosa_mlir(tflite_path, args.output,
                                 TosaConverterOutputFormat.Text)
  print(f"Exported {args.output}")


if __name__ == "__main__":
  main()
