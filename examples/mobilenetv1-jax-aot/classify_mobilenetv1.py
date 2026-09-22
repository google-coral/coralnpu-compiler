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


def preprocess_image(image_path):
  img = Image.open(image_path).convert("RGB").resize((128, 128))
  x = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
  return keras.applications.mobilenet.preprocess_input(x)


def main():
  args = utils.parse_inference_args()
  script_dir = os.path.dirname(os.path.abspath(__file__))
  image_path = args.input_path or os.path.join(
      script_dir, "../mobilenetv2-jax-aot/cat.jpg")
  input_data = preprocess_image(image_path)

  predict_func = utils.load_vmfb(args.vmfb)
  probs = np.asarray(predict_func(input_data))
  results = keras.applications.mobilenet.decode_predictions(probs, top=5)[0]
  print("\nTop-5 MobileNetV1 Predictions:")
  for rank, (_, label, score) in enumerate(results, 1):
    print(f"  {rank}: {label:<20} ({score:.4f})")
  utils.handle_results(probs, args)


if __name__ == "__main__":
  main()
