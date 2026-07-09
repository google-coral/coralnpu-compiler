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

import sys
import time
from gemma import gm
import jax
import jax.numpy as jnp


def load_params(device):
  checkpoint_path = gm.ckpts.CheckpointPath.GEMMA3_270M_IT

  print("Loading params...")
  params = gm.ckpts.load_params(checkpoint_path)
  params = jax.tree.map(lambda x: jax.device_put(x, device), params)
  print("Params loaded.")
  return params


def format_prompt(history, user_input):
  return f"{history}<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"


def run_inference(model, params, tokens_list, max_new_tokens=10):
  generated_tokens = []
  for _ in range(max_new_tokens):
    tokens_arr = jnp.array([tokens_list + generated_tokens], dtype=jnp.int32)
    out = model.apply({"params": params}, tokens_arr, return_last_only=True)
    next_token = int(jnp.argmax(out.logits[0, :]))
    if next_token in (
        gm.text.Gemma3Tokenizer.special_tokens.EOS,
        gm.text.Gemma3Tokenizer.special_tokens.END_OF_TURN,
    ):
      break
    generated_tokens.append(next_token)
  return generated_tokens


def decode_response(tokenizer, output_tokens):
  return tokenizer.decode(output_tokens)


def main():
  print(sys.executable)
  print(sys.version)

  print(f"jax version={jax.__version__}")
  jax.config.update("jax_platforms", "coralnpu_plugin")
  jax.config.update("jax_use_shardy_partitioner", False)

  devices = jax.devices("coralnpu_plugin")
  print(f"Devices found: {devices}")
  cpu_dev = devices[0]
  npu_dev = devices[1]
  print(f"Using CPU device (device 0): {cpu_dev}")
  print(f"Using CoralNPU device (device 1): {npu_dev}")

  model = gm.nn.Gemma3_270M()
  params = load_params(cpu_dev)

  tokenizer = gm.text.Gemma3Tokenizer()

  print("\nInteractive Chat started. Type 'exit' or 'quit' to end.")

  history = ""
  turn = 1
  while True:
    print(f"\n--- Turn {turn} ---\n")

    try:
      user_input = input("User: ").strip()
      if not sys.stdin.isatty():
        print(user_input)
    except (EOFError, KeyboardInterrupt):
      user_input = "exit"
      if not sys.stdin.isatty():
        print(user_input)

    if not user_input or user_input.lower() in ("exit", "quit"):
      print("Done")
      break

    history = format_prompt(history, user_input)

    # Tokenize
    tokens_list = tokenizer.encode(history, add_bos=True)
    print(f"[Debug] Prompt length: {len(tokens_list)} tokens")

    print("Model is thinking...")
    t0 = time.time()
    output_tokens = run_inference(model, params, tokens_list, max_new_tokens=10)
    elapsed = time.time() - t0
    print(f"[Debug] Generated in {elapsed:.2f}s")

    response_text = decode_response(tokenizer, output_tokens)
    print(f"Model: {response_text}")

    history += response_text + "<end_of_turn>\n"
    turn += 1


if __name__ == "__main__":
  main()
