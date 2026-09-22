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
from scipy import signal
from scipy.io import wavfile
from examples import utils

SAMPLE_WAV_URL = (
    "https://github.com/tensorflow/tflite-micro/raw/main/"
    "tensorflow/lite/micro/examples/micro_speech/testdata/yes_1000ms.wav")


def extract_features(wav_path):
  sr, audio = wavfile.read(wav_path)
  audio_f = audio.astype(np.float32)
  _, _, zxx = signal.stft(audio_f[:16000],
                          fs=sr,
                          nperseg=1024,
                          noverlap=512,
                          nfft=1024)
  power_spec = (np.abs(zxx[:128, :5])**2).T
  return (10.0 * np.log10(power_spec + 1e-10)).reshape(1,
                                                       640).astype(np.float32)


def main():
  args = utils.parse_inference_args()
  wav_path = args.input_path or keras.utils.get_file("yes_1000ms.wav",
                                                     SAMPLE_WAV_URL)
  input_data = extract_features(wav_path)

  predict_func = utils.load_vmfb(args.vmfb)
  reconstructed = np.asarray(predict_func(input_data))
  mse = float(np.mean(np.square(input_data - reconstructed)))
  status = "ANOMALY" if mse > 10.0 else "NORMAL"
  print("\nAnomaly Detection Result:")
  print(
      f"  Input range:         [{input_data.min():.2f}, {input_data.max():.2f}]"
  )
  print(f"  Reconstructed range: [{reconstructed.min():.2f},"
        f" {reconstructed.max():.2f}]")
  print(f"  Anomaly Score (MSE): {mse:.4f} ({status})")
  utils.handle_results(reconstructed, args)


if __name__ == "__main__":
  main()
