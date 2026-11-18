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

import numpy as np
from PIL import Image
from examples import utils

DUMMY_CLASSES = [f"class_{i}" for i in range(5)]


def preprocess_image(image_path):
  img = Image.open(image_path).convert("RGB").resize((224, 224))
  x_u8 = np.array(img, dtype=np.int16)
  return np.expand_dims((x_u8 - 128).astype(np.int8), axis=0)


def dequantize_output(output_i8):
  return (output_i8.astype(np.float32) + 128.0) / 256.0


def main():
  args = utils.parse_inference_args()
  script_dir = os.path.dirname(os.path.abspath(__file__))
  image_path = args.input_path or os.path.join(
      script_dir, "../mobilenetv2-jax-aot/cat.jpg")
  input_data = preprocess_image(image_path)

  predict_func = utils.load_vmfb(args.vmfb, module_name="module")
  output_i8 = np.asarray(predict_func(input_data), dtype=np.int8)
  probs = dequantize_output(output_i8)

  scores = probs[0]
  ranked = np.argsort(scores)[::-1]
  print("\nTop-5 MobileNetV1 (0.25, 224x224 INT8) Predictions:")
  for rank, cls_idx in enumerate(ranked, 1):
    print(f"  {rank}: {DUMMY_CLASSES[cls_idx]:<10}"
          f" ({scores[cls_idx]:.4f}, q={int(output_i8[0, cls_idx])})")
  utils.handle_results(probs, args)


if __name__ == "__main__":
  main()
