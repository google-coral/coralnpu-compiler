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

import io
import os
import tarfile
import time
import urllib.request
from tosa_converter_for_tflite import (
    TosaConverterOutputFormat,
    tflite_flatbuffer_to_tosa_mlir,
)

MODEL_URL = "https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz"


def main():
  workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
  if workspace_dir:
    output_dir = os.path.join(workspace_dir, "examples",
                              "mobilenetv2-tflite-aot")
  else:
    output_dir = os.path.dirname(os.path.abspath(__file__))

  os.makedirs(output_dir, exist_ok=True)
  tflite_path = os.path.join(output_dir, "mobilenet_v2.tflite")
  mlir_path = os.path.join(output_dir, "mobilenet_v2.mlir")

  if not os.path.exists(tflite_path):
    print(f"Downloading MobileNetV2 TFLite model archive from {MODEL_URL}...")
    t0 = time.time()
    req = urllib.request.Request(MODEL_URL,
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
      tar_bytes = resp.read()
    print(f"Downloaded in {time.time() - t0:.2f}s ({len(tar_bytes)} bytes)")

    print(f"Extracting mobilenet_v2_1.0_224.tflite to {tflite_path}...")
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
      extracted = False
      for member in tar.getmembers():
        if member.name.endswith("mobilenet_v2_1.0_224.tflite"):
          f = tar.extractfile(member)
          with open(tflite_path, "wb") as out_f:
            out_f.write(f.read())
          extracted = True
          break
      if not extracted:
        raise RuntimeError("mobilenet_v2_1.0_224.tflite not found in archive")
    print("TFLite model extracted successfully.")
  else:
    print(f"Using existing TFLite model at {tflite_path}")

  print("Converting TFLite model to TOSA MLIR via tosa-converter-for-tflite...")
  t0 = time.time()
  tflite_flatbuffer_to_tosa_mlir(tflite_path, mlir_path,
                                 TosaConverterOutputFormat.Text)
  print(f"Converted to {mlir_path} in {time.time() - t0:.2f}s")
  print("Done.")


if __name__ == "__main__":
  main()
