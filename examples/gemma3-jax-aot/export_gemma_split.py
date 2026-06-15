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
import time
import typing
import gemma.gm.utils._types
from gemma import gm
import jax
import jax.numpy as jnp


class SplitGemmaKV(gm.nn.Gemma3_270M):

  def apply_part1(
      self,
      tokens: jax.Array,
      positions: jax.Array,
      cache_list: list[typing.Any],
      attention_mask: jax.Array,
  ) -> tuple[jax.Array, list[typing.Any]]:
    x = self.embedder.encode(tokens)
    mask = attention_mask.astype(jnp.bool_)
    new_cache = []
    for i in range(9):
      c, x = self.blocks[i](
          x,
          positions,
          cache_list[i],
          mask,
      )
      new_cache.append(c)
    return x.astype(jnp.float32), new_cache

  def apply_part2(
      self,
      x: jax.Array,
      positions: jax.Array,
      cache_list: list[typing.Any],
      attention_mask: jax.Array,
  ) -> tuple[jax.Array, list[typing.Any]]:
    x = x.astype(jnp.bfloat16)
    mask = attention_mask.astype(jnp.bool_)
    new_cache = []
    for i in range(9, 18):
      c, x = self.blocks[i](
          x,
          positions,
          cache_list[i - 9],
          mask,
      )
      new_cache.append(c)
    x = self.final_norm(x)
    return x.astype(jnp.float32), new_cache

  def apply_part3(self, x: jax.Array) -> jax.Array:
    logits = self.embedder.decode(x.astype(jnp.bfloat16))
    return jnp.argmax(logits, axis=-1).astype(jnp.int32)


def main():
  jax.config.update("jax_use_shardy_partitioner", False)
  print("Loading model and params...")
  model = SplitGemmaKV()
  params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_270M_IT)
  print("Loaded.")

  out_dir = "./"
  os.makedirs(out_dir, exist_ok=True)

  dummy_model = gm.nn.Gemma3_270M()
  dict_cache = dummy_model.init_cache(batch_size=1,
                                      dtype=jnp.bfloat16,
                                      cache_length=128)
  c1_init = [dict_cache[f"layer_{i}"] for i in range(9)]
  c2_init = [dict_cache[f"layer_{i}"] for i in range(9, 18)]

  dummy_token_dec = jnp.zeros((1, 1), dtype=jnp.int32)
  pos_dec = jnp.zeros((1, 1), dtype=jnp.int32)
  mask_dec = jnp.zeros((1, 1, 128), dtype=jnp.int8)
  dummy_x_dec = jnp.zeros((1, 1, model.config.embed_dim), dtype=jnp.float32)

  full_params = {"params": params}

  @jax.jit
  def p1_dec(p, token, pos, cache, mask):
    return model.apply(p, token, pos, cache, mask, method=model.apply_part1)

  @jax.jit
  def p2_dec(p, x, pos, cache, mask):
    return model.apply(p, x, pos, cache, mask, method=model.apply_part2)

  @jax.jit
  def p3_dec(p, x):
    return model.apply(p, x, method=model.apply_part3)

  print("Lowering Part 1...")
  t0 = time.time()
  l_p1_dec = p1_dec.lower(full_params, dummy_token_dec, pos_dec, c1_init,
                          mask_dec)
  with open(os.path.join(out_dir, "gemma3_270m_part1.mlir"), "w") as f:
    f.write(str(l_p1_dec.compiler_ir(dialect="stablehlo")))
  print(f"Part 1 MLIR exported in {time.time() - t0:.2f}s")

  print("Lowering Part 2...")
  t0 = time.time()
  l_p2_dec = p2_dec.lower(full_params, dummy_x_dec, pos_dec, c2_init, mask_dec)
  with open(os.path.join(out_dir, "gemma3_270m_part2.mlir"), "w") as f:
    f.write(str(l_p2_dec.compiler_ir(dialect="stablehlo")))
  print(f"Part 2 MLIR exported in {time.time() - t0:.2f}s")

  print("Lowering Part 3...")
  t0 = time.time()
  l_p3_dec = p3_dec.lower(full_params, dummy_x_dec)
  with open(os.path.join(out_dir, "gemma3_270m_part3.mlir"), "w") as f:
    f.write(str(l_p3_dec.compiler_ir(dialect="stablehlo")))
  print(f"Part 3 MLIR exported in {time.time() - t0:.2f}s")


if __name__ == "__main__":
  main()
