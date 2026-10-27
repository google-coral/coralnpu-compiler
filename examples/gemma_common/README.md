# Gemma 3 & Gemma 4 JAX Examples

The Gemma examples and their shared helpers:

| Directory                                       | Model                      | Mode                  | Description                                                                                                                                                                                                              |
| :---------------------------------------------- | :------------------------- | :-------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`examples/gemma_common/`](.)                   | Shared                     | Library               | Step exporter, `K x N` einsum rewrite, KV-cache generation schedule and multi-turn chat loop ([`gemma_common.py`](gemma_common.py)); IREE runtime helpers for the AOT examples ([`gemma_iree_runtime.py`](gemma_iree_runtime.py)). |
| [`examples/gemma3-jax-aot/`](../gemma3-jax-aot) | Gemma 3 270M IT            | AOT (`iree.runtime`)  | Exports StableHLO `.mlir`, compiles `.vmfb` modules with `iree-compile`, and runs multi-turn chat across CPU + CoralNPU.                                                                                                 |
| [`examples/gemma3-jax-pjrt/`](../gemma3-jax-pjrt) | Gemma 3 270M IT          | JIT (`pjrt_plugin`)   | Runs the same model, weight layout, and step schedule directly from JAX via `libiree_pjrt_coralnpu_dylib.so`.                                                                                                            |
| [`examples/gemma4-jax-aot/`](../gemma4-jax-aot) | Gemma 4 E2B IT (text-only) | AOT (`iree.runtime`)  | Exports StableHLO `.mlir` + `*_kept_indices.json`, compiles `.vmfb` modules, and runs multi-turn chat across CPU + CoralNPU.                                                                                             |

## Quick Start

Each example provides an end-to-end `test_chat.sh` script that builds the
required compiler/runtime targets, compiles the model, and runs a scripted
two-turn chat:

```shell
# Gemma 3 270M (AOT, CPU + CoralNPU)
./examples/gemma3-jax-aot/test_chat.sh

# Gemma 3 270M (JAX PJRT JIT, CPU + CoralNPU)
./examples/gemma3-jax-pjrt/test_chat.sh

# Gemma 4 E2B text-only (AOT, CPU + CoralNPU)
./examples/gemma4-jax-aot/test_chat.sh
```

Useful flags forwarded by `test_chat.sh` to `chat.py`:

- `--cpu`: Compile and run on host CPU only (`local-sync` in AOT; device 0 in
  PJRT).
- `--max_new_tokens N`: Maximum tokens generated per turn (default `128`; `16`
  for Gemma 4 on CoralNPU).
- `--no-full-history` (also `--no-format_chat` in Gemma 4): Disable Gemma
  `<start_of_turn>` / `<|turn>` multi-turn prompt formatting.
- `--dummy_weights` *(Gemma 4 AOT)*: Use synthetic zero weights instead of
  downloading the checkpoint.

Once the AOT `.vmfb` files are built, you can run interactive chat directly
without recompiling:

```shell
bazel run --config=dev //examples/gemma3-jax-aot:chat -- \
    --prefill_vmfb_path "$(pwd)/examples/gemma3-jax-aot/gemma3_270m_prefill.vmfb" \
    --decode_vmfb_path "$(pwd)/examples/gemma3-jax-aot/gemma3_270m_decode.vmfb"
```

---

## What Is Exported

`export_step_mlirs()` in [`gemma_common.py`](gemma_common.py) exports
(`jax.export`) a single autoregressive step function
`(params, tokens, positions, cache, attention_mask) -> (next_token, new_cache)`
to StableHLO MLIR, where `tokens` and `positions` are `int32[1, seq_len]` and
`next_token` is `int32[1]`:

- **Weight-free abstract tracing:** Both
  [`gemma3_util.get_abstract_params()`](../gemma3-jax-aot/gemma3_util.py) and
  [`gemma4_util.get_abstract_inputs()`](../gemma4-jax-aot/gemma4_util.py) derive
  parameter PyTree shapes via `jax.eval_shape`, so exporting `.mlir` does not
  download model checkpoints. Weights are passed as external runtime inputs
  (`bfloat16`).
- **Fixed KV cache (`CACHE_LEN = 128`):** Each attention layer holds static
  128-slot cache buffers (`bfloat16` `k` and `v`, `int32[1, 128]` `positions`
  for sliding-window masks, and an `int32[1]` `end_index` cursor), returned as
  updated cache buffers on every step.
- **Three static sequence-length modules:** To avoid single-token prompt prefill
  while keeping static shapes for CoralNPU tiling, `export_step_mlirs()` exports
  three `.mlir` files from the same `step_fn`, each under its own function name,
  which JAX uses as the module name, so one IREE `SystemContext` can load all
  three `.vmfb` files side by side (`load_step_funcs()`):
  - `<stem>.mlir` (`SEQ_LEN = 16`, module `@jit_step_fn`): medium prefill tail
    and single-module fallback (used alone in AOT `--cpu` mode).
  - `<stem>_prefill.mlir` (`PREFILL_SEQ_LEN = 32`, module `@jit_prefill_fn`):
    bulk prompt prefill.
  - `<stem>_decode.mlir` (`DECODE_SEQ_LEN = 1`, module `@jit_decode_fn`):
    1-token autoregressive generation.
- **`int8` causal mask at the ABI boundary:** `attention_mask` has shape
  `[1, seq_len, 128]` with `int8` elements (`1` where
  `cache_col <= pos_idx + row`, `0` otherwise) and is cast to `bool` (`i1`)
  inside the step function, avoiding sub-byte `i1` buffers across the IREE HAL
  ABI.
- **Dead-argument filtering (`module_kept_var_idx`):** JAX's StableHLO export
  drops unused flattened inputs. `export_step_mlirs()` verifies all three step
  lengths keep the exact same input subset and returns those indices. Gemma 3
  uses every leaf; Gemma 4 E2B shares KV caches across its upper 20 layers,
  leaving those layers' `kv_einsum`/`key_norm` weights and `k`/`v`/`positions`
  cache buffers unused, so `export_gemma.py` writes `<stem>_kept_indices.json`
  and `chat.py` filters the flattened argument list before calling into IREE
  (`load_kept_indices()` falls back to re-deriving them with `jax.export` if the
  JSON file is absent).

---

## Optimizations

### Model & Export Optimizations

- **2D `K x N` projections (`make_fast_einsum`, `transpose_params`):** Upstream
  Gemma stores `q_einsum` `(N, D, H)` (`BTD,NDH->BTNH`), `kv_einsum`
  `(C, K, D, H)` (`BSD,CKDH->CBSKH`), and `gating_einsum` `(2, H_ffw, F)`
  (`...F,NHF->...NH`) with a non-leading contraction axis (`D` or `F`) and
  contracts via multi-axis `jnp.einsum`. `make_fast_einsum()` moves each
  contraction axis ahead of that projection's output axes (`(D, N, H)`,
  `(C, D, K, H)` with `(D, K, H)` per `C` slice, `(F, 2, H_ffw)`) and lowers
  `q`, `kv` (`C = 2` slices), `gating`, attention-output (`BTNH,NHD->BTD`), and
  Gemma 4's scaled per-layer input (`...td,dnp->...tnp`) projections to 2D
  `jnp.dot` matmuls. At load time, `transpose_params()` permutes checkpoint
  weights once into contiguous `K x N` layout so IREE lowers them directly to
  Zvt/RVV 2D matmuls without runtime weight transposes.
- **Gemma 3 RoPE & mask overrides (`patch_fast_rope`):**
  [`gemma3_util.py`](../gemma3-jax-aot/gemma3_util.py) overrides `apply_rope`
  with `fast_rope()` (explicit `float32` `sin`/`cos` rotation that skips
  `jnp.pad` when `nope_angles == 0`), changes `K_MASK` from `-2.38e38` to
  `-1e4`, and installs `make_fast_einsum`.
- **Last-token-only LM head:** Both Gemma 3 (`GemmaKV.apply_step`) and Gemma 4
  (`return_last_only=True`) index the hidden state of only the last non-pad
  token (`sum(tokens != 0) - 1`) before running the vocabulary projection
  (`embedder.decode`) and `argmax`, so `16`- and `32`-token prefill steps
  project 1 row instead of `seq_len` rows to the vocabulary. Token ID `0` is
  reserved as the pad token for partially filled chunks.

### Runtime & Scheduling Optimizations

- **One-time immutable weight upload (`to_device_arrays`):** In AOT, flattened
  `bfloat16` weights are uploaded once at startup with
  `SHARING_IMMUTABLE | SHARING_REPLICATE`. Because the MPACT backend keeps a
  single simulator instance per process and caches read-only bindings in
  simulated DDR, weights stay resident across all steps and chat turns instead
  of being re-copied per call, as long as they fit in its read-only window
  (1 GiB of simulated DDR minus 128 MiB reserved for staging). PJRT uploads its
  weight PyTree to HAL device memory once via `jax.device_put`.
- **Tiered prefill/decode schedule (`generate`):** Prompts are prefilled in
  `32`-token chunks while more than `16` tokens remain (and
  `pos <= 128 - 32`); the tail uses the `32`-token step (`17..32` tokens),
  `16`-token step (`2..16` tokens), or `1`-token step (`1` token). Subsequent
  tokens are generated one at a time with the `1`-token decode module. When
  `--prefill_vmfb_path` or `--decode_vmfb_path` is omitted (such as AOT
  `--cpu`), prefill and/or decode fall back to the `16`-token module; generation
  stops once `curr_len > 128 - decode_len` (after slot `127` for `1`-token
  decode, or slot `112` for the `16`-token fallback so padded 16-slot cache
  writes do not overrun or wrap the 128-slot cache).
- **Persistent multi-turn KV cache & `end_index` reset:** `cached_tokens`
  records the token sequence currently in the KV cache across turns. On a new
  turn, `generate()` finds the longest matching prompt prefix (excluding the
  final prompt token so `next_token` is still computed), aligns `start_pos` down
  to a multiple of `SEQ_LEN = 16` so fixed `16`-/`32`-token cache writes fit
  within `CACHE_LEN = 128`, and prefills only the remaining suffix. Because
  zero-padded chunks advance each layer's internal `end_index` by the full
  `seq_len` and multi-turn prefix reuse may rewind `start_pos`, the step runner
  resets every layer's `end_index` leaf to `pos_idx` before each step.

---

## How Everything Is Put Together

- **1. Heterogeneous CPU + CoralNPU compilation:** AOT `test_chat.sh` passes
  `--iree-hal-target-device=local` (`llvm-cpu`) and
  `--iree-hal-target-device=coralnpu` explicitly, while the PJRT plugin
  (`CoralNPUClientInstance::SetCompilerFlags`) injects both targets
  automatically when inputs are uncommitted (`target_device = None`, vs pinned
  to device 0 for `--cpu`). Both flows configure CoralNPU with a 1 MiB DTCM
  (`--coralnpu-dtcm-size-kb=1024` and `//crt:coralnpu_tcm_highmem_ld`, passed
  via `IREE_PJRT_IREE_COMPILER_OPTIONS` in PJRT). The compiler's roofline
  affinity pass places supported compute dispatches on CoralNPU while keeping
  host-bound or oversized I/O dispatches (such as vocabulary
  lookups/projections) on the CPU.
- **2. AOT flow (`gemma3-jax-aot`, `gemma4-jax-aot`):**
  - **Phase 1 (`export_gemma.py`):** Traces the abstract model and writes
    `<stem>{,_prefill,_decode}.mlir` (plus `<stem>_kept_indices.json` for
    Gemma 4).
  - **Phase 2 (`iree-compile`):** Compiles the three `.mlir` files to `.vmfb` in
    parallel background jobs (or compiles only `<stem>_cpu.vmfb` when `--cpu` is
    passed).
  - **Phase 3–4 (runtime build & `chat.py`):** Builds the Python runtime
    bindings and `:chat` binary, creates a multi-device
    (`local-sync` + `coralnpu`) `iree.runtime.SystemContext` (`local-sync` only
    with `--cpu`), loads the `.vmfb` modules, uploads pre-transposed checkpoint
    weights once as immutable device buffers, allocates the 128-token KV cache,
    and runs `chat_loop()`.
- **3. PJRT JIT flow (`gemma3-jax-pjrt`):**
  - `test_chat.sh` builds `libIREECompiler.so`,
    `libiree_pjrt_coralnpu_dylib.so`, `iree-compile`,
    `coralnpu_tcm_highmem.ld`, and the `//examples/gemma3-jax-pjrt:chat` binary
    (with the `requirements_lock.txt` packages), creates
    `bazel-bin/external/{crt,toolchain_rv32}` symlinks so `libIREECompiler.so`
    resolves relative CRT/RV32 toolchain paths, and runs the binary with
    `PJRT_NAMES_AND_LIBRARY_PATHS`, `IREE_PJRT_COMPILER_LIB_PATH`, and
    `ENABLE_PJRT_COMPATIBILITY=1`.
  - [`examples/gemma3-jax-pjrt/chat.py`](../gemma3-jax-pjrt/chat.py) imports
    `gemma3_util` and `gemma_common` directly (sharing the exact same `GemmaKV`
    model, `FastEinsum`/`fast_rope` patches, `transpose_params`, and
    `make_run_turn`/`chat_loop` helpers) and selects
    `jax_platforms="coralnpu_plugin"` (CPU on device 0, CoralNPU on device 1).
  - At startup, `compile_steps()` lowers `step_fn` for sequence lengths
    `(16, 32, 1)` and compiles all three concurrently via a
    `ThreadPoolExecutor`, so interactive per-turn latency reflects execution
    only.
