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
import numpy as np
import iree.runtime as ireert


def load_and_flatten_params():
  checkpoint_path = gm.ckpts.CheckpointPath.GEMMA3_270M_IT

  print("Loading params...")
  params = gm.ckpts.load_params(checkpoint_path)
  print("Params loaded.")

  print("Flattening params...")
  flat_params, _ = jax.tree_util.tree_flatten(params)
  return [np.asarray(p) for p in flat_params]


def init_iree_func(vmfb_path, device_name="local-sync"):
  config = ireert.Config(device_name)

  print(f"Loading VMFB from {vmfb_path}...")
  try:
    vm_module = ireert.VmModule.mmap(config.vm_instance, vmfb_path)
  except Exception as e:
    print(f"mmap failed: {e}. Trying from_flatbuffer...")
    with open(vmfb_path, "rb") as f:
      vm_module = ireert.VmModule.from_flatbuffer(config.vm_instance, f.read())

  ctx = ireert.SystemContext(config=config)
  ctx.add_vm_module(vm_module)
  return ctx.modules.jit_generate_tokens.main


def format_prompt(history, user_input):
  return f"{history}<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"


def run_inference(main_func, flat_params_np, tokens_list, rng):
  prompt_len = len(tokens_list)
  padded_tokens = np.zeros((1, 128), dtype=np.int32)
  padded_tokens[0, :prompt_len] = tokens_list

  inputs = flat_params_np + [padded_tokens, rng]
  output_tokens = main_func(*inputs)
  return np.asarray(output_tokens)[0]


def decode_response(tokenizer, output_tokens_np):
  response_tokens = []
  for t in output_tokens_np:
    if t in (0, tokenizer.special_tokens.EOS,
             tokenizer.special_tokens.END_OF_TURN):
      break
    response_tokens.append(int(t))
  return tokenizer.decode(response_tokens)


def main():
  flat_params_np = load_and_flatten_params()

  vmfb_path = "./gemma3_270m_cpu.vmfb"
  main_func = init_iree_func(vmfb_path)

  tokenizer = gm.text.Gemma3Tokenizer()

  key = jax.random.PRNGKey(0)

  print("\nInteractive Chat started. Type 'exit' or 'quit' to end.")

  history = ""
  turn = 1
  while True:
    print(f"\n--- Turn {turn} ---\n")

    try:
      user_input = input("User: ").strip()
    except (EOFError, KeyboardInterrupt):
      user_input = "exit"

    if not user_input or user_input.lower() in ("exit", "quit"):
      print("Done")
      break

    history = format_prompt(history, user_input)

    # Tokenize
    tokens_list = tokenizer.encode(history, add_bos=True)
    print(f"[Debug] Prompt length: {len(tokens_list)} tokens")

    if len(tokens_list) > 128:
      print("ERROR: History exceeds limit.")
      break

    # Split RNG key
    key, subkey = jax.random.split(key)

    print("Model is thinking...")
    t0 = time.time()
    output_tokens_np = run_inference(main_func, flat_params_np, tokens_list,
                                     np.asarray(subkey))
    elapsed = time.time() - t0
    print(f"[Debug] Generated in {elapsed:.2f}s")

    response_text = decode_response(tokenizer, output_tokens_np)
    #.removesuffix("<end_of_turn>")
    print(f"Model: {response_text}")

    history += response_text + "<end_of_turn>\n"

    turn += 1


if __name__ == "__main__":
  main()
