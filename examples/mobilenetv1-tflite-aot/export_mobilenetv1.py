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
import urllib.request

from tosa_converter_for_tflite import (
    TosaConverterDebugInfo,
    TosaConverterOutputFormat,
    _tflite_flatbuffer_to_tosa_mlir,
)
from examples import utils

MODEL_URL = ("https://github.com/google-coral/coralnpu/raw/"
             "4d82d87019790b061176893e254309f942bfb01e/"
             "tests/cocotb/tutorial/tfmicro/models/"
             "mobilenet_v1_0.25_224_int8_dummy.tflite")


def main():
  args = utils.parse_export_args()
  output_dir = os.environ.get("BUILD_WORKING_DIRECTORY", ".")
  tflite_path = os.path.join(output_dir,
                             "mobilenet_v1_0.25_224_int8_dummy.tflite")
  if not os.path.exists(tflite_path):
    urllib.request.urlretrieve(MODEL_URL, tflite_path)

  # Specialize dynamic batch (-1) to static batch 1 and emit MLIR bytecode so
  # TOSA 1.0 ops load cleanly across LLVM textual syntax changes.
  _tflite_flatbuffer_to_tosa_mlir(
      tflite_path,
      args.output,
      TosaConverterOutputFormat.Bytecode,
      TosaConverterDebugInfo.Disabled,
      {"serving_default_input_1:0": [1, 224, 224, 3]},
  )
  print(f"Exported {args.output}")


if __name__ == "__main__":
  main()
