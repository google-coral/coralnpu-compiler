# Add PyTorch AOT Model Examples (`DeiT-Tiny` and `SmolLM2-135M`)

- **Author(s)**: Lun Dong
- **Component Doc(s)**: [aot_examples.md](../aot_examples.md) *(new in CL 2a)*
- **Bug(s)**: None

---

## 1. Summary & Motivation

The repository provides ahead-of-time (AOT) examples for JAX/Flax, Keras, and
TFLite models in [`examples/`](../../../examples/), but has no PyTorch examples.
IREE's C++ `input_torch` plugin is disabled in CMake
([`CMakeLists.txt`](../../../CMakeLists.txt)) and absent from IREE's Bazel
build, and CoralNPU's preprocessing, Data Tightly-Coupled Memory (`DTCM`)
tiling, and `16x16` Zvt matrix codegen require static-shape StableHLO or TOSA
inputs.

This change traces PyTorch `nn.Module` graphs on the host into StableHLO MLIR
via `torch` and `torchax`, then compiles and runs `.vmfb` modules across
CoralNPU (`coralnpu`) and host CPU (`local-sync`). It adds two AOT examples:
**`DeiT-Tiny`** (`facebook/deit-tiny-patch16-224`, single-pass Vision
Transformer (`ViT`) classification) and **`SmolLM2-135M`**
(`HuggingFaceTB/SmolLM2-135M-Instruct`, multi-turn Grouped-Query Attention
(`GQA`) chat).

- **Goals**:
  - Lower concrete and `"meta"`-device PyTorch `nn.Module` graphs to static
    StableHLO on the host using the CUDA-free `torch` wheel (`whl/cpu`, for
    host-side export and weight loading) and `torchax`, without adding PyTorch
    runfiles to existing JAX/TFLite examples.
  - Run `DeiT-Tiny` image classification across CoralNPU + host CPU on the
    default 32 KB DTCM build in CI (`atol=1e-2` vs. `reference.npy`).
  - Run 3-bucket (`16`-, `32`-, and `1`-token) `SmolLM2-135M` multi-turn chat
    across CoralNPU + host CPU within the default 8 KB ITCM / 32 KB DTCM budget
    with zero runtime weight transposes.
- **Non-Goals**:
  - Enabling C++ `IREE_INPUT_TORCH` / `torch-mlir` plugins in `iree-compile`:
    host-side `torchax` tracing emits static StableHLO directly without new C++
    compiler dependencies.
  - Dynamic sequence-length or dynamic-batch compilation: CoralNPU affinity
    annotation, DTCM tiling, and Zvt codegen require static tensor dimensions.
  - Direct PyTorch PJRT JIT execution via `libiree_pjrt_coralnpu_dylib.so`:
    deferred as a separate cross-component follow-up (`§4`).

---

## 2. Proposed Change & Component Doc Delta

An exporter script lifts an evaluation-mode `torch.nn.Module` on the host into a
JAX-traceable `(params, apply_fn)` pair via `torchax` and emits static-shape
StableHLO `.mlir`. `iree-compile` then compiles the `.mlir` into heterogeneous
`local-sync` + `coralnpu` `.vmfb` modules executed through the IREE Python
runtime, documented in [`aot_examples.md`](../aot_examples.md) (indexed in
[`INDEX.md`](../INDEX.md)). Below, `[M, K] @ [K, N]` denotes a 2D contraction of
`M` rows and reduction `K` into `N` output columns, `B` batch size, and `L` step
sequence length:

### Isolated PyTorch Bridge (`examples/torch_utils.py`)

Keeping `torch` and `torchax` in `//examples:torch_utils` leaves
[`examples/utils.py`](../../../examples/utils.py) free of PyTorch runfiles. In
`BUILD.bazel`, `pip_compile` passes `--no-config` and
`--index-strategy=unsafe-best-match` (`size = "medium"`) so `uv` resolves the
CUDA-free `whl/cpu` `torch` wheel alongside PyPI without leaking workstation
`pip.conf` URLs. `lift_torch_module()` converts concrete or `"meta"`-device
modules into `(params, apply_fn)` pairs for `export_torch_stablehlo()` and
`gemma_common.export_step_mlirs()`.

### `DeiT-Tiny` Padding & Patch Matmul (`examples/deit-tiny-torch-aot/`)

`DeiT-Tiny` has `197` tokens (`1` `[CLS]` + `14x14` patches) and 12 encoder
blocks (`48` projections after merging `Q, K, V`). Because `197` is prime,
`Q @ K^T` (`[3, 197, 64] @ [3, 64, 197]`) fails Zvt column alignment and runs on
RVV. Zvt requires $N$ divisible by `16`, or by `8`/`12` when $M$ has a divisor
in `[3, 15]` ([`matrix_codegen.md`](../matrix_codegen.md)). Under the default
`24 KB` DTCM tile budget (`32 KB` DTCM minus `4 KB` stack and `4 KB` reserve;
[`dtcm_tiling.md`](../dtcm_tiling.md)), unsplit $M = 197$ shrinks 24 `qkv`/`fc1`
projections from `16x64` to `16x16` Zvt. It also forces 24 residual `proj`/`fc2`
projections onto RVV (`832 B` LHS + `2 * 12,608 B` `[197, 16]` residual/output
slices + `1,024 B` scratch = `27,072 B > 24,576 B`).

Zero-padding `197` to `208` (`13 x 16`) with `-1e4` key masking and transposing
projection weights to `(K, N)` (`_transpose_linear()`) restores `16x64` Zvt
across all encoder blocks. Reshaping the `16x16` patch `Conv2d` into
`(B, 196, 768) @ (768, 192)` (since `ConvolutionToIGEMM` is inactive;
[`matrix_codegen.md`](../matrix_codegen.md) §2.0) and slicing token `0`
preserves host FP32 probabilities within rounding. On the default 32 KB DTCM
build, CoralNPU output matches `reference.npy` within `atol=1e-2` under default
`--coralnpu-fast-math`.

### `SmolLM2-135M` Step & `(K, N)` Weights (`examples/smollm2-torch-aot/`)

`SmolLM2Step` implements the 30-layer GQA decoder (`hidden = 576`, SwiGLU
`1536`, 9 Q and 3 KV heads of dim `64`). Zvt and `vecmat` contractions stream
along `K` without promoting operands into DTCM, allocating at most one `4 KB`
`workgroup_local_memory` scratch drain tile. Under default compiler flags
(`24 KB` tile budget, `--coralnpu-fast-math`), all `1`-, `16`-, and `32`-token
dispatches fit within the default 8 KB ITCM / 32 KB DTCM budget (`3,712 B` peak
ITCM and `4 KB` `workgroup_local_memory`). Full `32x32` Zvt tiles (`8.1 KiB`
modeled footprint) step down to `16x32` at an `8 KB` tile budget and `16x16` at
`4 KB`.

`load_and_flatten_params()` transposes the 7 per-layer linear weights to
`(K, N)` `bfloat16`, while `SmolLM2Attention` groups Q heads by KV head
(`[B * 3, L * 3, 64]`, $M = 3L \ge 3$). `apply_rope()` uses
broadcast-multiply-add to avoid `stablehlo.concatenate` host fallbacks on
16/32-token prefill (1-token elementwise RoPE stays on CPU). A 2D-by-1D `matvec`
(`[49152, 576] @ [576]` on the last non-pad hidden state) keeps native
`[49152, 576]` `embed_tokens` on host CPU as a contiguous row-wise reduction
without `49152x576` weight transposes. Its `54.0 MiB` footprint
(`49152 * 576 * 2 B`) exceeds the `16 MiB` non-Zvt I/O cap in
`compiler/Transforms/CoralNPUAffinityAnnotation.cpp`.
Sequence-major `[1, 128, 3, 64]` KV caches update via `dynamic_update_slice` to
match `gemma_common`, accepting per-layer `[128, 3, 64] -> [3, 128, 64]` cache
transposes (`§4`).

### Invariants, Fallbacks & Edge Cases

- **`"meta"` PyTree & Buffer ABI**: `lift_torch_module()` lifts parameters and
  buffers into a `dict` flattened in lexicographic key order by
  `jax.tree_util.tree_flatten` (`"layers.10..."` before `"layers.2..."`),
  preserving tied aliases via `torch.func.functional_call`. External loaders
  (`load_and_flatten_params()`) flatten matching keys identically (`SmolLM2Step`
  registers zero buffers).
- **Import & Sandbox Guards**: `_import_torchax()` stubs missing `aten` ops
  during `import torchax` while raising `AttributeError` on `__dunder__` and
  non-`aten` lookups. `hf_download()` disables `hf_xet` by default, checks the
  `pwd`-resolved local cache (`local_files_only=True`), and falls back to
  `TEST_TMPDIR`/temp storage on misses or read-only `OSError`s (`DeiT-Tiny`
  export needs network or a warm cache; `SmolLM2-135M` export is weight-free).
- **Offline ChatML & Dummy Weights**: `SmolLM2Tokenizer` includes an offline
  regex/CRC32 fallback under `--dummy_weights`, ignores `add_bos=True`, and
  reserves `0` (`<|endoftext|>`) as pad and `(0, 2)` as `STOP_TOKENS`.
  Deterministic non-zero `--dummy_weights` avoid argmaxing to stop token `0`
  during prefill so offline runs exercise the 1-token decode step (`decode_fn`
  in `gemma_common.py`).

### Alternatives Considered

- **Exporting HuggingFace `LlamaForCausalLM` + `StaticCache` Directly**: Avoids
  maintaining a custom `SmolLM2Step` module (the main trade-off accepted here).
  However, `StaticCache.update` emits `stablehlo.scatter`, `LlamaForCausalLM`
  keeps `(N, K)` weights and ungrouped $M = 1$ decode GQA, and concrete
  `from_pretrained` lifting inlines 270 MB of weights.
- **Adding `timm` and `torchvision` to `requirements.txt`**: Supplies stock ViT
  builders, but adds extra vision wheels when `pillow` is already direct.
  Promoting `huggingface-hub`, `safetensors`, and `tokenizers` (already locked
  transitively via `kenning`) avoids adding `@pip_deps//transformers` to example
  targets.
- **Unpadded `197`-Token `DeiT-Tiny` Graph**: Avoids ~5.6% sequence padding, but
  prime $M = N = 197$ blocks Zvt `Q @ K^T` lowering and Zvt DTCM $M$-loop tiling
  on the `24 KB` tile budget.

---

## 3. Implementation & Verification Plan

### Stage 1 (`CL 2a`): PyTorch Lowering Utility & `DeiT-Tiny` AOT Example

- [ ] **Dependencies (`BUILD.bazel`, `requirements.txt`,
  `requirements_lock.txt`)**: Add the CUDA-free host `torch` wheel
  (`https://download.pytorch.org/whl/cpu`) and `torchax`, promote
  `huggingface-hub`, set `size = "medium"` and
  `extra_args = ["--no-config", "--index-strategy=unsafe-best-match"]` on
  `pip_compile` in `BUILD.bazel`, and run
  `bazel run --config=dev //:requirements`.
- [ ] **Shared Bridge & Docs (`examples/torch_utils.py`, `examples/BUILD.bazel`,
  `docs/design/aot_examples.md`, `docs/design/INDEX.md`, `README.md`)**:
  Implement `//examples:torch_utils`, add `aot_examples.md` + its `INDEX.md`
  row, and document `DeiT-Tiny` usage in `README.md`.
- [ ] **`DeiT-Tiny` Example & Tests (`examples/deit-tiny-torch-aot/BUILD.bazel`,
  `export_deit_tiny.py`, `classify_deit_tiny.py`, `test_deit_tiny.sh`,
  `reference.npy`, `tests/examples/BUILD.bazel`,
  `tests/examples/torch_examples_test.py`)**: Export `pytorch_model.bin` via
  `export_deit_tiny.py`, run `cat.jpg` inference in `classify_deit_tiny.py`, and
  verify:
  - *Positive*: `deit_tiny_test` matches `reference.npy` (`atol=1e-2`) and
    `torch_examples_test` covers `torch_utils` and `DeiTTiny` (`B = 2`).
  - *Negative / Fallback*: `test_deit_tiny.sh` asserts `Affinity: coralnpu` and
    `Affinity: local`, and `torch_examples_test` checks `ValueError`/dunder
    rejections.

### Stage 2 (`CL 2b`): `SmolLM2-135M` Multi-Turn Chat AOT Example

- [ ] **Dependencies & `"meta"` Lifting (`requirements.txt`,
  `requirements_lock.txt`, `examples/torch_utils.py`,
  `tests/examples/torch_examples_test.py`)**: Promote `safetensors` and
  `tokenizers` to direct pins and extend `lift_torch_module()` to map `"meta"`
  parameters and buffers to `jax.ShapeDtypeStruct` leaves (with tied-weight
  support), verified by `torch_examples_test`.
- [ ] **`SmolLM2-135M` Example & Verification
  (`examples/smollm2-torch-aot/BUILD.bazel`, `smollm2_util.py`,
  `export_smollm2.py`, `chat.py`, `test_chat.sh`,
  `examples/gemma_common/gemma_common.py`)**: Implement 3-bucket step export in
  `export_smollm2.py`, interactive `chat.py`, generalized `--full-history` help
  text in `gemma_common.py`, and verify via `test_chat.sh`:
  - *Positive*: `test_chat.sh` (`--dummy_weights` and `--cpu --dummy_weights`)
    exports `16`/`32`/`1`-token `.mlir` steps, compiles CPU and CoralNPU
    `.vmfb`s, and completes 2-turn non-stop chat runs (`>= 2` `Model:` lines).
  - *Negative / Fallback*: `test_chat.sh` asserts `Affinity: coralnpu` and
    `Affinity: local` (in CoralNPU mode), zero `transpose_49152x576` dispatches
    in compiled `.vmfb`s, and no `^ERROR:` lines in `chat.log`.
- [ ] **Doc & Archive Sync (`README.md`, `examples/gemma_common/README.md`,
  `docs/design/aot_examples.md`)**: Update docs and `## 5. TODOs` in
  `aot_examples.md` for `SmolLM2-135M`, verify `docs/design/INDEX.md`, archive
  this brief via `git mv` to `docs/design/archive/`, and run
  `./scripts/format-code.sh`.

---

## 4. Follow-Up Items (`aot_examples.md` TODOs / Buganizer `2161972`)

- [ ] **Head-Major GQA KV-Cache Layout** (`aot_examples.md` |
      `examples/smollm2-torch-aot/smollm2_util.py`): Evaluate storing
      multi-KV-head caches (`KV_H` KV heads of dim `D`) as `[1, KV_H, 128, D]`
      to avoid per-layer `[128, KV_H, D] -> [KV_H, 128, D]` cache transpose
      dispatches.
- [ ] **PyTorch PJRT JIT Example via `torchax`** (`Buganizer 2161972` | `P3` |
      `examples/`, `pjrt_plugin/`): Run a `torchax`-wrapped `torch.nn.Module`
      directly via `libiree_pjrt_coralnpu_dylib.so` without ahead-of-time
      `.mlir` export.
