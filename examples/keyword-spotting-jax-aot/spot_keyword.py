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

CLASSES = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "unknown",
    "_background_noise_",
]


def extract_spectrogram(wav_path):
  sr, audio = wavfile.read(wav_path)
  audio_f = audio.astype(np.float32) / 32768.0
  _, _, zxx = signal.stft(audio_f,
                          fs=sr,
                          nperseg=480,
                          noverlap=160,
                          nfft=512,
                          boundary=None)
  mag = np.abs(zxx[:128, :50]).T
  if mag.shape[0] < 50:
    mag = np.pad(mag, ((0, 50 - mag.shape[0]), (0, 0)))

  mels = np.linspace(
      2595.0 * np.log10(1.0 + 125.0 / 700.0),
      2595.0 * np.log10(1.0 + 3800.0 / 700.0),
      12,
  )
  hz = 700.0 * (10.0**(mels / 2595.0) - 1.0)
  bins = np.floor((512 + 1) * hz / sr).astype(int)
  fb = np.zeros((10, 128), dtype=np.float32)
  for m in range(1, 11):
    for k in range(bins[m - 1], bins[m]):
      fb[m - 1, k] = (k - bins[m - 1]) / max(1, (bins[m] - bins[m - 1]))
    for k in range(bins[m], min(128, bins[m + 1])):
      fb[m - 1, k] = (bins[m + 1] - k) / max(1, (bins[m + 1] - bins[m]))

  log_mel = np.log(np.dot(mag, fb.T) + 1e-6)
  norm = ((log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-6) *
          255.0)
  return norm[None, :, :, None].astype(np.float32)


def main():
  args = utils.parse_inference_args()
  wav_path = args.input_path or keras.utils.get_file("yes_1000ms.wav",
                                                     SAMPLE_WAV_URL)
  input_data = extract_spectrogram(wav_path)

  predict_func = utils.load_vmfb(args.vmfb)
  probs_batch = np.asarray(predict_func(input_data))
  probs = probs_batch[0]
  top_indices = np.argsort(probs)[-5:][::-1]
  print("\nTop-5 Keyword Predictions:")
  for rank, idx in enumerate(top_indices, 1):
    print(f"  {rank}: {CLASSES[idx]:<20} ({probs[idx]:.4f})")
  utils.handle_results(probs_batch, args)


if __name__ == "__main__":
  main()
