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
import sys
import time

# Set backend before importing keras
os.environ["KERAS_BACKEND"] = "jax"

import keras
import jax
import numpy as np
from PIL import Image
import iree.runtime as ireert


def init_iree_func(vmfb_path, device_name="local-sync"):
  config = ireert.Config(device_name)

  print(f"Loading VMFB from {vmfb_path}...")
  try:
    vm_module = ireert.VmModule.mmap(config.vm_instance, vmfb_path)
  except Exception as e:
    print(f"mmap failed: {e}. Trying from_flatbuffer...")
    with open(vmfb_path, "rb") as f:
      vm_module = ireert.VmModule.from_flatbuffer(config.vm_instance, f.read())

  ctx = ireert.SystemContext(config=config)
  ctx.add_vm_module(vm_module)
  return ctx.modules.jit_predict.main


def preprocess_image(image_path):
  img = Image.open(image_path).resize((224, 224))
  x = np.array(img, dtype=np.float32)
  # Ensure 3 channels (RGB)
  if x.shape[-1] == 4:
    x = x[..., :3]
  elif len(x.shape) == 2:  # Grayscale
    x = np.stack([x] * 3, axis=-1)
  x = np.expand_dims(x, axis=0)
  # MobileNetV2 preprocessing: scale to [-1, 1]
  x = keras.applications.mobilenet_v2.preprocess_input(x)
  return x


def main():
  if len(sys.argv) < 2:
    print("Usage: python classify.py <image_path>")
    sys.exit(2)

  image_path = sys.argv[1]
  if not os.path.exists(image_path):
    print(f"Error: File {image_path} not found.")
    sys.exit(1)

  vmfb_path = "./mobilenet_v2.vmfb"
  # If running via Bazel, we might need to find it relative to script
  if not os.path.exists(vmfb_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vmfb_path = os.path.join(script_dir, "mobilenet_v2.vmfb")

  if not os.path.exists(vmfb_path):
    print(f"Error: VMFB file {vmfb_path} not found. Please compile it first.")
    sys.exit(1)

  predict_func = init_iree_func(vmfb_path)

  print(f"Preprocessing image {image_path}...")
  input_data = preprocess_image(image_path)

  print("Running inference...")
  t0 = time.time()
  output = predict_func(input_data)
  elapsed = time.time() - t0
  print(f"Inference completed in {elapsed:.4f}s")

  output_np = np.asarray(output)

  try:
    results = keras.applications.mobilenet_v2.decode_predictions(output_np,
                                                                 top=5)[0]
    print("\nPredictions:")
    for i, (imagenet_id, label, score) in enumerate(results):
      print(f"{i+1}: {label} ({score:.4f})")
  except Exception as e:
    print(f"Failed to decode predictions: {e}")
    top_indices = np.argsort(output_np[0])[-5:][::-1]
    print("\nTop indices:", top_indices)
    print("Values:", output_np[0][top_indices])


if __name__ == "__main__":
  main()
