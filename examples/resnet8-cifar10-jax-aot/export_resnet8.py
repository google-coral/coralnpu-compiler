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

os.environ["KERAS_BACKEND"] = "jax"

import keras
from examples import utils

MODEL_URL = (
    "https://github.com/mlcommons/tiny/raw/"
    "debd5310ffbad653932e2995816e853197116dd9/"
    "benchmark/training/image_classification/trained_models/pretrainedResnet.h5"
)


def main():
  args = utils.parse_export_args()
  output_dir = os.environ.get("BUILD_WORKING_DIRECTORY", ".")
  h5_path = os.path.join(output_dir, "pretrainedResnet.h5")
  if not os.path.exists(h5_path):
    urllib.request.urlretrieve(MODEL_URL, h5_path)

  model = keras.models.load_model(h5_path, compile=False)
  utils.export_stablehlo(
      lambda x: model(x, training=False),
      input_shape=(1, 32, 32, 3),
      output_path=args.output,
  )


if __name__ == "__main__":
  main()
