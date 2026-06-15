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

from dataclasses import dataclass
import os
import time
import typing
import gemma.gm.text._prefill
import gemma.gm.text._sampler_loop
import gemma.gm.text._sampling
import gemma.gm.utils._types
from gemma import gm
import jax
import jax.numpy as jnp


def load_model_and_params() -> tuple[gm.nn.Gemma3_270M, typing.Any]:
  model = gm.nn.Gemma3_270M()
  print("Loading params...")
  params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_270M_IT)
  print("Params loaded.")
  return model, params


@dataclass
class ExportConfig:
  cache_length: int = 512
  max_out_length: int = 128
  prompt_length: int = 128
  batch_size: int = 1
  output_dir: str = "./"
  output_filename: str = "gemma3_270m.mlir"


class GemmaAotExporter:
  """Handles lowering and export of Gemma 3 270M generation loop to StableHLO MLIR."""

  def __init__(self, config: ExportConfig) -> None:
    self.config = config
    self.tokenizer = gm.text.Gemma3Tokenizer()
    self.model, self.params = load_model_and_params()

  def build_generate_tokens_fn(self) -> typing.Callable[..., jax.Array]:
    cache_length = self.config.cache_length
    max_out_length = self.config.max_out_length
    model = self.model

    special_tokens = self.tokenizer.special_tokens
    end_tokens = (
        special_tokens.EOS,
        special_tokens.END_OF_TURN,
    )
    forbidden_tokens = self.tokenizer.FORBIDDEN_TOKENS
    sampling_method = gemma.gm.text._sampling.Greedy()

    @jax.jit
    def generate_tokens(params: typing.Any, tokens: jax.Array,
                        rng: jax.Array) -> jax.Array:
      inputs = gemma.gm.utils._types.Input(
          text=tokens,
          images=None,
          config=model.config.input_config,
      )

      init_state = gemma.gm.text._prefill.prefill(
          model=model,
          params=params,
          input=inputs,
          last_state=None,
          cache_length=cache_length,
          max_out_length=max_out_length,
          pad_length=None,
          rng=rng,
          sharding=None,
      )

      sampler_loop_inst = gemma.gm.text._sampler_loop.SamplerLoop(
          model=model,
          end_tokens=end_tokens,
          forbidden_tokens=forbidden_tokens,
          sampling=sampling_method,
          cache_length=cache_length,
          special_tokens=special_tokens,
      )

      final_state = sampler_loop_inst.sample(
          params=params,
          init_state=init_state,
          max_new_tokens=jnp.asarray(max_out_length),
          stream=False,
      )

      return final_state.predicted_tokens

    return generate_tokens

  def lower_to_stablehlo(self) -> str:
    generate_tokens = self.build_generate_tokens_fn()

    key = jax.random.PRNGKey(0)
    dummy_tokens = jnp.zeros(
        (self.config.batch_size, self.config.prompt_length), dtype=jnp.int32)
    dummy_tokens = dummy_tokens.at[:, 0].set(self.tokenizer.special_tokens.BOS)

    print("Lowering to StableHLO...")
    t0 = time.time()
    lowered = generate_tokens.lower(self.params, dummy_tokens, key)
    stablehlo_ir = lowered.compiler_ir(dialect="stablehlo")
    print(f"Lowered in {time.time() - t0:.2f}s")
    return str(stablehlo_ir)

  def export(self) -> str:
    stablehlo_ir_text = self.lower_to_stablehlo()

    os.makedirs(self.config.output_dir, exist_ok=True)
    output_path = os.path.join(self.config.output_dir,
                               self.config.output_filename)

    print(f"Writing MLIR to {output_path}...")
    t0 = time.time()
    with open(output_path, "w") as f:
      f.write(stablehlo_ir_text)
    print(f"Wrote MLIR in {time.time() - t0:.2f}s")

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"MLIR file size: {file_size_mb:.2f} MB")
    return output_path


def main() -> None:
  jax.config.update("jax_use_shardy_partitioner", False)
  config = ExportConfig()
  exporter = GemmaAotExporter(config)
  exporter.export()


if __name__ == "__main__":
  main()
