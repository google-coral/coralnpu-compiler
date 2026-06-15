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


def load_and_flatten_split_params():
  checkpoint_path = gm.ckpts.CheckpointPath.GEMMA3_270M_IT

  print("Loading params...")
  params = gm.ckpts.load_params(checkpoint_path)
  print("Params loaded.")

  print("Extracting Part 1, Part 2, Part 3 params...")
  flat_params, _ = jax.tree_util.tree_flatten(params)

  p1_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8]
  p1_np = [np.asarray(flat_params[0])] + [
      np.asarray(p)
      for l in p1_layers
      for p in jax.tree_util.tree_leaves(params[f"layer_{l}"])
  ]

  p2_layers = [10, 11, 12, 13, 14, 15, 16, 17, 9]
  p2_np = [np.asarray(flat_params[1])] + [
      np.asarray(p)
      for l in p2_layers
      for p in jax.tree_util.tree_leaves(params[f"layer_{l}"])
  ]

  p3_np = [np.asarray(flat_params[0])]
  return p1_np, p2_np, p3_np


def init_iree_funcs(device_name="local-sync"):
  config = ireert.Config(device_name)
  ctx = ireert.SystemContext(config=config)

  modules = [
      "./gemma3_270m_part1.vmfb",
      "./gemma3_270m_part2.vmfb",
      "./gemma3_270m_part3.vmfb",
  ]

  for m_path in modules:
    print(f"Loading {m_path}...")
    vm_mod = ireert.VmModule.mmap(config.vm_instance, m_path)
    ctx.add_vm_module(vm_mod)

  m1_dec = ctx.modules.jit_p1_dec.main
  m2_dec = ctx.modules.jit_p2_dec.main
  m3_dec = ctx.modules.jit_p3_dec.main
  return m1_dec, m2_dec, m3_dec


def init_split_caches():
  dummy_model = gm.nn.Gemma3_270M()
  dict_cache = dummy_model.init_cache(batch_size=1,
                                      dtype=jnp.bfloat16,
                                      cache_length=128)

  c1_init = [dict_cache[f"layer_{i}"] for i in range(9)]
  flat_c1, _ = jax.tree_util.tree_flatten(c1_init)
  c1 = [np.asarray(c) for c in flat_c1]

  c2_init = [dict_cache[f"layer_{i}"] for i in range(9, 18)]
  flat_c2, _ = jax.tree_util.tree_flatten(c2_init)
  c2 = [np.asarray(c) for c in flat_c2]

  return c1, c2


def format_prompt(history, user_input):
  return f"{history}<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"


def run_split_inference(m1_dec, m2_dec, m3_dec, p1_np, p2_np, p3_np,
                        tokens_list):
  prompt_len = len(tokens_list)
  print(f"[Debug] Prefilling {prompt_len} tokens...")

  c1, c2 = init_split_caches()

  # 1. Prefill
  for step in range(prompt_len):
    curr_tok = np.array([[tokens_list[step]]], dtype=np.int32)
    pos = np.array([[step]], dtype=np.int32)
    mask = np.zeros((1, 1, 128), dtype=np.int8)
    mask[0, 0, :step + 1] = 1

    res1 = m1_dec(*(p1_np + [curr_tok, pos] + c1 + [mask]))
    x1 = res1[0]
    c1 = list(res1[1:])

    res2 = m2_dec(*(p2_np + [x1, pos] + c2 + [mask]))
    x2 = res2[0]
    c2 = list(res2[1:])

  # 2. Decode first token
  pred_tok = m3_dec(*(p3_np + [x2]))
  first_tok = int(np.asarray(pred_tok)[0, 0])
  generated = [first_tok]

  curr_len = prompt_len
  curr_tok = np.array([[first_tok]], dtype=np.int32)

  while curr_len < 128:
    pos = np.array([[curr_len]], dtype=np.int32)
    mask = np.zeros((1, 1, 128), dtype=np.int8)
    mask[0, 0, :curr_len + 1] = 1

    res1 = m1_dec(*(p1_np + [curr_tok, pos] + c1 + [mask]))
    x1 = res1[0]
    c1 = list(res1[1:])

    res2 = m2_dec(*(p2_np + [x1, pos] + c2 + [mask]))
    x2 = res2[0]
    c2 = list(res2[1:])

    pred_tok = m3_dec(*(p3_np + [x2]))
    next_tok = int(np.asarray(pred_tok)[0, 0])

    generated.append(next_tok)

    if next_tok in (0, 106, 107):
      break

    curr_tok = np.array([[next_tok]], dtype=np.int32)
    curr_len += 1

  return generated


def decode_response(tokenizer, output_tokens):
  response_tokens = []
  for t in output_tokens:
    if t in (0, tokenizer.special_tokens.EOS,
             tokenizer.special_tokens.END_OF_TURN):
      break
    response_tokens.append(int(t))
  return tokenizer.decode(response_tokens)


def main():
  p1_np, p2_np, p3_np = load_and_flatten_split_params()

  m1_d, m2_d, m3_d = init_iree_funcs()

  tokenizer = gm.text.Gemma3Tokenizer()

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

    tokens_list = tokenizer.encode(history, add_bos=True)
    print(f"[Debug] Prompt length: {len(tokens_list)} tokens")

    if len(tokens_list) > 128:
      print("ERROR: History exceeds limit.")
      break

    t0 = time.time()
    output_tokens = run_split_inference(m1_d, m2_d, m3_d, p1_np, p2_np, p3_np,
                                        tokens_list)
    t1 = time.time()

    model_response = decode_response(tokenizer, output_tokens)
    print(f"Model: {model_response}")
    print(f"[Inference took {t1 - t0:.2f}s]")

    history += model_response + "<end_of_turn>\n"

    turn += 1


if __name__ == "__main__":
  main()
