#!/usr/bin/env -S uv run --script
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
#
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "torch",
#   "torchvision",
#   "ultralytics<8.4.83",
#   "tensorflow-cpu<=2.19.0",
#   "tf_keras<=2.19.0",
#   "onnx",
#   "onnxruntime",
#   "onnx2tf",
#   "onnxslim",
#   "onnx-graphsurgeon>=0.3.26",
#   "sng4onnx",
#   "ai-edge-litert",
#   "psutil",
#   "flatbuffers",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# torchvision = { index = "pytorch-cpu" }
# ///
"""Downloads YOLOv8n (yolov8n.pt) and exports a 160x160 float32 TFLite model."""

import os
import pathlib
import shutil
import tempfile
from ultralytics import YOLO


def main():
  script_dir = pathlib.Path(__file__).resolve().parent
  target_path = script_dir / "yolov8n_160.tflite"
  with tempfile.TemporaryDirectory() as tmp_dir:
    os.chdir(tmp_dir)
    pt_path = pathlib.Path(tmp_dir) / "yolov8n.pt"
    model = YOLO(str(pt_path))
    exported_path = model.export(format="tflite", imgsz=160)
    shutil.copyfile(exported_path, target_path)
  size_mb = target_path.stat().st_size / (1024 * 1024)
  print(f"Saved {target_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
  main()
