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
from examples import utils


def main():
  args = utils.parse_export_args()
  model = keras.applications.EfficientNetB0(weights="imagenet")
  utils.export_stablehlo(
      lambda x: model(x, training=False),
      input_shape=(1, 224, 224, 3),
      output_path=args.output,
  )


if __name__ == "__main__":
  main()
