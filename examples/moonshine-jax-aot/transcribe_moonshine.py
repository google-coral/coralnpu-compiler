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

import json
import os
import zipfile

os.environ["KERAS_BACKEND"] = "jax"

import keras
import numpy as np
from scipy.io import wavfile
from examples import utils

MOONSHINE_WHL_URL = "https://files.pythonhosted.org/packages/e5/a7/4dfd1a68629e9567b5c46ab9621fc0dbf1b4e1a56cf296e91aa7283c3635/useful_moonshine-20241016-py3-none-any.whl"


def load_assets():
  whl_path = keras.utils.get_file("useful_moonshine-20241016-py3-none-any.whl",
                                  MOONSHINE_WHL_URL)
  extract_dir = os.path.join(os.path.dirname(whl_path), "moonshine_extracted")
  with zipfile.ZipFile(whl_path, "r") as zf:
    zf.extract("moonshine/assets/beckett.wav", path=extract_dir)
    zf.extract("moonshine/assets/tokenizer.json", path=extract_dir)
  wav_path = os.path.join(extract_dir, "moonshine/assets/beckett.wav")
  tok_path = os.path.join(extract_dir, "moonshine/assets/tokenizer.json")
  with open(tok_path, "r") as f:
    tok_data = json.load(f)
  vocab = {v: k for k, v in tok_data["model"]["vocab"].items()}
  return wav_path, vocab


def main():
  args = utils.parse_inference_args()
  default_wav_path, vocab = load_assets()
  wav_path = args.input_path or default_wav_path
  _, audio = wavfile.read(wav_path)
  input_data = (audio[:16000].astype(np.float32) / 32768.0).reshape(1, 16000, 1)

  predict_func = utils.load_vmfb(args.vmfb)
  logits = np.asarray(predict_func(input_data))
  last_logits = logits[0, -1]
  top_tokens = np.argsort(last_logits)[-5:][::-1]
  print("\nTop-5 Next-Token Predictions:")
  for rank, tok_id in enumerate(top_tokens, 1):
    piece = vocab.get(int(tok_id), str(tok_id))
    print(f"  {rank}: {piece!r:<12} (id={int(tok_id):<6}"
          f" logit={float(last_logits[tok_id]):.4f})")
  top_token = int(top_tokens[0])
  top_score = float(last_logits[top_token])
  summary = np.array([float(top_token), top_score], dtype=np.float32)
  utils.handle_results(summary, args)


if __name__ == "__main__":
  main()
