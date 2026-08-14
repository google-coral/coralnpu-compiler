# Reproducing and Debugging Louhi CI Presubmit Locally

This document explains how to reproduce Louhi's CI environment locally using a clean `git worktree` and Podman to debug test failures, build issues, or environment discrepancies that occur in presubmit for a Gerrit change.

Assumes the starting point is in your local `coralnpu-compiler` repository root.

For instructions on installing and setting up rootless Podman on Cloudtop, see [aaronyu/podman.md](https://g3doc.corp.google.com/company/users/aaronyu/podman.md?cl=head).

---

## 1. Overview of Louhi Presubmit Workflow

Louhi CI runs presubmits for `coralnpu-compiler` via Google Cloud Build (GCB) containers defined in `flow-test_coralnpu-compiler.yml`. The pipeline consists of:

1. **Stage 1 (`clone_coralnpu_repo`)**: Clones the repository manifest, checks out the change under test, symlinks `hw/coralnpu` into `third_party/coralnpu`, and applies patches to submodules.
2. **Stage 2 (Parallel test stages)**:
   - `test_coralnpu-compiler_dev` (`bazel test --config=dev //tests:ci`)
   - `test_coralnpu-compiler_release` (`bazel test --config=release //tests:ci`)
   - `format_code_check` (`scripts/format-code.sh`)
   - `cmake_build` (CMake Ninja build with ccache)
   - `cmake_test` (CMake build and `ctest -L ci`)

### How to Find the Louhi Flows and Stage Definitions

Louhi pipeline definitions are stored in an internal Git configuration repository:

```bash
git clone sso://louhi-config-internal/cerebra-hw-shodan
```

Within that repository:
- **Flow Definition**: `.louhi/flows/flow-test_coralnpu-compiler.yml`
  Lists all pipeline stages, triggers, and parameters (including inline scripts for presubmit checks).
- **Stage Definitions**: `.louhi/stages/`
  Contains reusable stage implementations (e.g. `stage_type-clone_coralnpu_repo.yml`, `stage_type-bazel-test.yml`, `stage_type-presubmit_script.yml`).

If CI configurations change upstream, inspect `flow-test_coralnpu-compiler.yml` and `.louhi/stages/` to update the local commands.

---

## 2. Create an Isolated `git worktree` for the Gerrit Change

From your `coralnpu-compiler` repository root, fetch the Gerrit change and create an isolated worktree for it:

```bash
# Fetch the Gerrit change ref (replace <XX>, <CHANGE_NUMBER>, and <PATCHSET>):
# e.g., git fetch spacebeaker refs/changes/85/91885/3
git fetch spacebeaker refs/changes/<XX>/<CHANGE_NUMBER>/<PATCHSET>

# Create and switch to the clean worktree:
git worktree add ../coralnpu-compiler-ci FETCH_HEAD
cd ../coralnpu-compiler-ci
```

---

## 3. Initialize Submodules and Apply Patches

*(Corresponds to Stage 1 `clone_coralnpu_repo` in `flow-test_coralnpu-compiler.yml`, defined in `.louhi/stages/stage_type-clone_coralnpu_repo.yml`).*

In the newly created worktree, initialize all submodules (including `third_party/coralnpu`) and apply patches:

```bash
# 1. Initialize and update submodules recursively
git submodule update --init --recursive --depth=1

# 2. Apply third-party patches
./scripts/patch-third_party.sh --restore-first all
```

---

## 4. Build the Container Image

Build the container image using the Dockerfile from `third_party/coralnpu` inside the new worktree:

```bash
cd third_party/coralnpu
podman build --tag coralnpu --file utils/coralnpu.dockerfile .
cd ../..
```

> **Note on Rootless Podman:** Do not pass `--build-arg _UID=$(id -u) --build-arg _GID=$(id -g)`. Rootless user namespaces map up to UID/GID 65535, so building with the default `_UID=1000` / `_GID=1000` is required. Podman will map your host user to container UID 1000 at runtime via `--userns=keep-id:uid=1000,gid=1000`.

---

## 5. Run Stages in the Container

You only need to run the stage that fails, if you know which one it is.

### A. Run Bazel Dev CI (`test_coralnpu-compiler_dev`)

*(Corresponds to stage `test_coralnpu-compiler_dev` in `flow-test_coralnpu-compiler.yml`, implemented via `.louhi/stages/stage_type-bazel-test.yml` with `_TEST_FLAGS: "--config=dev"`).*

```bash
WORKTREE_DIR="$(pwd)"

podman run --rm \
  --userns=keep-id:uid=1000,gid=1000 \
  -v "${WORKTREE_DIR}":"${WORKTREE_DIR}" \
  -w "${WORKTREE_DIR}" \
  localhost/coralnpu:latest \
  bazel test \
    --config=dev \
    --test_output=errors \
    --verbose_failures \
    --noshow_progress \
    --test_summary=detailed \
    -- //tests:ci
```

### B. Run Bazel Release CI (`test_coralnpu-compiler_release`)

*(Corresponds to stage `test_coralnpu-compiler_release` in `flow-test_coralnpu-compiler.yml`, implemented via `.louhi/stages/stage_type-bazel-test.yml` with `_TEST_FLAGS: "--config=release"`).*

```bash
WORKTREE_DIR="$(pwd)"

podman run --rm \
  --userns=keep-id:uid=1000,gid=1000 \
  -v "${WORKTREE_DIR}":"${WORKTREE_DIR}" \
  -w "${WORKTREE_DIR}" \
  localhost/coralnpu:latest \
  bazel test \
    --config=release \
    --test_output=errors \
    --verbose_failures \
    --noshow_progress \
    --test_summary=detailed \
    -- //tests:ci
```

### C. Run Code Formatting Check (`format_code_check`)

*(Corresponds to stage `format_code_check` in `flow-test_coralnpu-compiler.yml`, defined in `parameters._SCRIPT`).*

```bash
WORKTREE_DIR="$(pwd)"

podman run --rm \
  --userns=keep-id:uid=1000,gid=1000 \
  -v "${WORKTREE_DIR}":"${WORKTREE_DIR}" \
  -w "${WORKTREE_DIR}" \
  localhost/coralnpu:latest \
  bash -c "scripts/format-code.sh && git diff --exit-code --ignore-submodules=all"
```

### D. Run CMake Build and Tests (`cmake_test`)

*(Corresponds to stage `cmake_test` in `flow-test_coralnpu-compiler.yml`, defined in `parameters._SCRIPT`).*

```bash
WORKTREE_DIR="$(pwd)"

podman run --rm \
  --userns=keep-id:uid=1000,gid=1000 \
  -v "${WORKTREE_DIR}":"${WORKTREE_DIR}" \
  -w "${WORKTREE_DIR}" \
  localhost/coralnpu:latest \
  bash -c '
    set -eux -o pipefail
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements_lock.txt
    BUILD_DIR=build
    cmake -G Ninja -B "${BUILD_DIR}" -S . \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DIREE_BUILD_TESTS=ON
    cmake --build "${BUILD_DIR}" -j $(nproc)
    cmake --build "${BUILD_DIR}" --target iree-check-module -j $(nproc)
    ctest --test-dir "${BUILD_DIR}" -L "ci" --output-on-failure -j $(nproc)
  '
```

---

## 6. Interactive Debugging Shell

To manually inspect generated artifacts, run Bazel commands, or debug tests interactively:

```bash
WORKTREE_DIR="$(pwd)"

podman run --rm -it \
  --userns=keep-id:uid=1000,gid=1000 \
  -v "${WORKTREE_DIR}":"${WORKTREE_DIR}" \
  -w "${WORKTREE_DIR}" \
  localhost/coralnpu:latest \
  /bin/bash
```

---

## 7. Cleanup

Once debugging is complete, remove the temporary worktree:

```bash
cd /path/to/coralnpu-compiler
git worktree remove ../coralnpu-compiler-ci
```

---

## 8. Key Tips

1. **Self-Contained Submodules**: Using `third_party/coralnpu` directly inside the worktree keeps the compiler checkout completely self-contained and avoids needing the full Shodan manifest.
2. **Tool Caching (Bazel & ccache)**: If you want caching across container runs without polluting your host's personal `~/.cache`, mount a dedicated cache directory (e.g. `-v "${HOME}/.cache/podman_cache:/home/builder/.cache"`). Container user `builder` stores both Bazel and ccache artifacts under `/home/builder/.cache`.
