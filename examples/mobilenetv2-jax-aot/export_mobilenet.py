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
# Set backend before importing keras
os.environ["KERAS_BACKEND"] = "jax"

import keras
import jax
import jax.numpy as jnp
import time


def main():
  # Make sure we use JAX backend.
  assert keras.backend.backend() == "jax", "Keras backend must be JAX"

  print("Loading MobileNetV2...")
  # We use ImageNet weights.
  model = keras.applications.MobileNetV2(weights="imagenet")
  print("Model loaded.")

  # We need to extract the variables (parameters) and the forward pass function.
  # Keras models in JAX backend are JAX-traceable.
  # We can just JIT the model call.
  # Note: Keras models might have state updates (e.g. batch norm moving averages)
  # if training=True, but for inference we use training=False (default).

  @jax.jit
  def predict(x):
    return model(x)

  # MobileNetV2 input shape is typically (batch, 224, 224, 3)
  dummy_input = jnp.zeros((1, 224, 224, 3), dtype=jnp.float32)

  # Warmup/trace
  print("Tracing model...")
  _ = predict(dummy_input)
  print("Model traced.")

  print("Lowering to StableHLO...")
  t0 = time.time()
  lowered = predict.lower(dummy_input)
  stablehlo_ir = lowered.compiler_ir(dialect="stablehlo")
  print(f"Lowered in {time.time() - t0:.2f}s")

  workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
  if workspace_dir:
    output_dir = os.path.join(workspace_dir, "examples", "mobilenetv2-jax-aot")
  else:
    output_dir = os.path.dirname(os.path.abspath(__file__))

  output_path = os.path.join(output_dir, "mobilenet_v2.mlir")
  print(f"Writing MLIR to {output_path}...")
  with open(output_path, "w") as f:
    f.write(str(stablehlo_ir))
  print("Done.")


if __name__ == "__main__":
  main()
