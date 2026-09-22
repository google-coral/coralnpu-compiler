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

os.environ["KERAS_BACKEND"] = "jax"

import keras
import numpy as np
from PIL import Image
from examples import utils

SAMPLE_IMAGE_URL = (
    "https://github.com/tensorflow/tflite-micro/raw/main/"
    "tensorflow/lite/micro/examples/person_detection/testdata/person.bmp")


def preprocess_image(image_path):
  img = Image.open(image_path).convert("RGB").resize((96, 96))
  x = np.array(img, dtype=np.float32)
  return np.expand_dims((x / 127.5) - 1.0, axis=0)


def main():
  args = utils.parse_inference_args()
  image_path = args.input_path or keras.utils.get_file("person.bmp",
                                                       SAMPLE_IMAGE_URL)
  input_data = preprocess_image(image_path)

  predict_func = utils.load_vmfb(args.vmfb)
  probs_batch = np.asarray(predict_func(input_data))
  probs = probs_batch[0]
  label = "person" if probs[1] > probs[0] else "non-person"
  print("\nPerson Detection Predictions:")
  print(f"  non-person : {probs[0]:.4f}")
  print(f"  person     : {probs[1]:.4f}")
  print(f"  Prediction : {label} ({max(probs):.4f})")
  utils.handle_results(probs_batch, args)


if __name__ == "__main__":
  main()
