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
import types
import zipfile

os.environ["KERAS_BACKEND"] = "jax"

import keras
from examples import utils

MOONSHINE_WHL_URL = "https://files.pythonhosted.org/packages/e5/a7/4dfd1a68629e9567b5c46ab9621fc0dbf1b4e1a56cf296e91aa7283c3635/useful_moonshine-20241016-py3-none-any.whl"
HF_BASE_URL = "https://huggingface.co/UsefulSensors/moonshine/resolve/main/tiny"


def _install_einops_shim():
  einops_mod = types.ModuleType("einops")

  def rearrange(x, pattern, **axes):
    if pattern == "... (d r) -> ... d r":
      return keras.ops.reshape(x, (*x.shape[:-1], x.shape[-1] // 2, 2))
    if pattern == "... d r -> ... (d r)":
      return keras.ops.reshape(x, (*x.shape[:-2], x.shape[-2] * x.shape[-1]))
    if pattern == "x y -> x 1 y":
      return keras.ops.expand_dims(x, axis=1)
    raise ValueError(pattern)

  einops_mod.rearrange = rearrange
  sys.modules["einops"] = einops_mod


def load_moonshine_tiny():
  _install_einops_shim()
  whl_path = keras.utils.get_file("useful_moonshine-20241016-py3-none-any.whl",
                                  MOONSHINE_WHL_URL)
  extract_dir = os.path.join(os.path.dirname(whl_path), "moonshine_extracted")
  with zipfile.ZipFile(whl_path, "r") as zf:
    zf.extract("moonshine/model.py", path=extract_dir)
  sys.path.insert(0, os.path.join(extract_dir, "moonshine"))
  import model as moonshine_model

  prep_w = keras.utils.get_file(
      "moonshine_tiny_preprocessor.weights.h5",
      f"{HF_BASE_URL}/preprocessor.weights.h5",
  )
  enc_w = keras.utils.get_file(
      "moonshine_tiny_encoder.weights.h5",
      f"{HF_BASE_URL}/encoder.weights.h5",
  )
  dec_w = keras.utils.get_file(
      "moonshine_tiny_decoder.weights.h5",
      f"{HF_BASE_URL}/decoder.weights.h5",
  )

  moonshine_model.Arange.call = lambda self, inputs: keras.ops.arange(
      getattr(self, "static_len", 40))
  model = moonshine_model.Moonshine(288, 288, 8, 6, 6)
  model._load_weights(prep_w, enc_w, dec_w)
  for layer in model.encoder.encoder.layers:
    if isinstance(layer, moonshine_model.Arange):
      layer.static_len = 40
  for layer in model.decoder.uncached_call.layers:
    if isinstance(layer, moonshine_model.Arange):
      layer.static_len = 1
  return model


def main():
  args = utils.parse_export_args()
  model = load_moonshine_tiny()

  def forward(audio_tensor):
    features = model.preprocessor(audio_tensor)
    seq_len = keras.ops.convert_to_tensor([40], dtype="int32")
    encoded = model.encoder(features, seq_len)
    tokens = keras.ops.convert_to_tensor([[1]], dtype="int32")
    dec_seq_len = keras.ops.convert_to_tensor([1], dtype="int32")
    return model.decoder.uncached_call([tokens, encoded, dec_seq_len])[0]

  utils.export_stablehlo(
      forward,
      input_shape=(1, 16000, 1),
      output_path=args.output,
  )


if __name__ == "__main__":
  main()
