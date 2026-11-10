---
name: implement-design
description: >-
  Implements an approved change design document from
  ./docs/design/changes/<change>.md on top of the initial CL 2 commit (which
  already contains the ./docs/design/<component>.md and .h/.td interface
  updates). Supports four descriptive steps: `code` (implement tasks, sync
  build/doc files, and update `<change>.md` and `<component>.md`), `audit`
  (spawn an isolated narrow-view subagent with no chat history for a code,
  style, and semantic-drift audit), `archive` (verify `<component>.md` sync, git
  mv `<change>.md` to `archive/`, and format code), and `file-followups` (file
  approved Section 4 follow-ups in Buganizer component 2161972 without
  implementing them).
---

# Design Implementation & Finalization Workflow (`/implement-design`)

Use this skill during **CL 2** (or staged `CL 2a`, `CL 2b`, ...) to implement,
audit, and finalize an approved design from `./docs/design/changes/<change>.md`
and `./docs/design/<component>.md`.

> [!IMPORTANT]
> **Reference Files (Load Only What the Active Step Needs)**:
> - [`references/standards.md`](references/standards.md) &
>   [`../../AGENTS.md`](../../AGENTS.md) — Spec-driven implementation,
>   adaptation tiers, audit rubric, and archival standards (read in `code`,
>   `audit`, and `archive`).
> - [`../design-doc/references/interview.md`](../design-doc/references/interview.md)
>   & [`../design-doc/references/lenses.md`](../design-doc/references/lenses.md)
>   — Load on demand only when escalating a 3-strike blocker, an architectural
>   pivot, or a non-trivial design trade-off surfaced by `audit`.

---

## Resolving `<change>` (No Full Path Required)

- **Short Name**: Resolve `<change>` or `<change>.md` (e.g., `zvt_conv_igemm`)
  to `./docs/design/changes/<change>.md` (or `./docs/design/archive/<change>.md`
  if already archived in CL 2).
- **Omitted Name**: List `./docs/design/changes/*.md` (excluding `TEMPLATE.md`).
  Use the sole active doc automatically, or check `git status` /
  `git diff --name-only` for a modified doc under `./docs/design/archive/*.md`
  (asking the developer if multiple exist).

---

## Named Steps

| Invocation                                       | Purpose                                                                                               |
| :----------------------------------------------- | :---------------------------------------------------------------------------------------------------- |
| `/implement-design code [<change>]`              | Implement tasks on top of the initial CL 2 commit, sync build/doc files, and update `<change>.md`.    |
| `/implement-design audit [<change>] [<commits>]` | Spawn an isolated narrow-view subagent (no chat history) for a code, style, and semantic-drift audit. |
| `/implement-design archive [<change>]`           | Verify `<component>.md` sync, `git mv` `<change>.md` to `archive/`, and format code.                  |
| `/implement-design file-followups [<change>]`    | Deduplicate and file approved Section 4 follow-ups into Buganizer component `2161972`.                |

If invoked without a step keyword, default to `code` (unless the prompt
requests `audit`, `archive`, or `file-followups`).

---

## Step 1: `code` — Implementation & Developer Review

- **1. Load Approved Design Context**:
  - Read `./docs/design/changes/<change>.md`, the updated
    `./docs/design/<component>.md` living docs (via `./docs/design/INDEX.md`),
    [`references/standards.md`](references/standards.md), and
    [`../../AGENTS.md`](../../AGENTS.md). Do **not** run `audit` or `archive`
    automatically during `code`.
- **2. Execute Tasks & Maintain Doc Synchronicity**:
  - Implement the tasks in **Section 3** following **Section 1** of
    [`references/standards.md`](references/standards.md). For a multi-CL chain
    (`CL 2a .. CL 2n`), implement the current `[WIP]` stage commit `CL 2k`,
    amend `CL 2k` (removing `[WIP]` while preserving its `Change-Id`), and
    rebase `CL 2(k+1) .. CL 2n` on top.
  - Keep `<change>.md` and `<component>.md` synchronized with the code across
    follow-up prompts (including after `archive` moved the doc to
    `archive/<change>.md`).
- **3. Next Step**:
  - Suggest `/implement-design audit` for an isolated code, style, and
    semantic-drift audit, or `/implement-design archive` to finalize CL 2 (or
    `/implement-design code` for the next stage `CL 2(k+1)`).

---

## Step 2: `audit` — Isolated Code, Style & Semantic-Drift Audit

Repeatable; each call spawns a fresh, stateless subagent (for a broader codebase
review, the developer can ask the main agent in the current session).

- **1. Resolve the Implementation Diff Command (`<diff_cmd>`)**:
  - Determine `<diff_cmd>` before spawning the subagent (never rely on bare
    `git diff`, which is empty when code is committed):
    - **Explicit `<commits>` argument**: `git diff <base_sha>..<tip_sha>` (or
      `git diff <commit>~1..<commit>`).
    - **Committed or uncommitted code on top of CL 1**: Find `<cl1_sha>` via
      `git log -n 1 --diff-filter=A --format=%H -- docs/design/changes/<change>.md`
      (falling back to `BASE_SHA` in `./agent-temp/<change>.commit` or
      `HEAD~1`), and use `git diff <cl1_sha>` so the diff includes both the
      initial CL 2 `<component>.md` changes and all committed/working-tree code
      edits.
- **2. Spawn an Isolated Audit Subagent (No Chat History, Narrow View)**:
  - Launch a zero-history subagent that inherits local file-reading and
    shell/git capabilities in read-only mode, reading **only**:
    - `./docs/design/changes/<change>.md` (or `archive/<change>.md`) and
      `./docs/design/<component>.md`
    - `./.agents/skills/implement-design/references/standards.md` and
      `./.agents/AGENTS.md`
    - The output of `<diff_cmd>`, the modified source/test files, any
      repository files explicitly referenced in `<change>.md` or
      `<component>.md`, and `#include`d MLIR/IREE `*Utils.h` / `*Interfaces.h`
      headers needed to check utility reuse under **Section 2.1** of
      [`references/standards.md`](references/standards.md)
  - Instruct it to perform a read-only audit (no file edits) using **Section
    2.1** of [`references/standards.md`](references/standards.md).
- **3. Fact-Check & Report Findings (No Automatic Edits)**:
  - Follow **Section 2.2** of
    [`references/standards.md`](references/standards.md), then end with a
    **`Next Step:`** prompt to specify which findings to fix or run
    `/implement-design archive`.

---

## Step 3: `archive` — Coherence Pass on `<component>.md` & Archive `<change>.md`

Run in the final implementation CL (`CL 2`, or the last stage `CL 2n` of a
multi-CL chain) before Gerrit upload:

- **1. Coherence Pass & Archive via `git mv`**:
  - Complete the `<component>.md` (`+ INDEX.md`) coherence pass per **Section
    3** of [`references/standards.md`](references/standards.md), then move
    `<change>.md` using `git mv`:

    ```bash
    mkdir -p docs/design/archive
    git mv ./docs/design/changes/<change>.md ./docs/design/archive/<change>.md
    ```

  - Mark the final sync item in **Section 3** as `- [x]`, remove
    `./agent-temp/<change>.commit` if present
    (`rm -f ./agent-temp/<change>.commit`), and run `./scripts/format-code.sh`.
- **2. Next Step**:
  - Show the command to amend CL 2 (`git commit --amend`, removing `[WIP]` and
    the initial-review note while preserving `Change-Id`) and push Ready for
    Review, plus `/implement-design file-followups <change>` once CL 2 receives
    `+2` (if Section 4 has unchecked items):

    ```bash
    git push -o ready origin HEAD:refs/for/master
    ```

---

## Step 4: `file-followups` — File Approved Follow-Ups in Buganizer (`2161972`)

> [!IMPORTANT]
> **Do NOT implement Section 4 follow-up items in code.** This step only files
> deferred follow-ups in Buganizer after CL 2 receives `+2` in Gerrit.

- **1. Deduplicate, Confirm & File**:
  - Read unchecked `- [ ]` items in **Section 4: Follow-Up Items** of
    `./docs/design/archive/<change>.md` (or `changes/<change>.md`).
  - If `/google/bin/releases/issues-cli/issues` is available (pass
    `BypassSandbox: true` if present in the tool schema), search for open
    duplicates via `issues readonly search "componentid:2161972 status:open
    <keywords>" --limit 5`, present a summary table (Title, Priority, Duplicate
    check result, and 1–2 sentence description referencing the Gerrit CL or
    archive doc path) for **explicit developer authorization**, and create
    approved tickets via `issues mutate create --component_id=2161972
    --title="<title>" --priority=<P1|P2|P3> --type=BUG
    --description_file=./agent-temp/.bug_desc.md` (or present them for manual
    filing if `issues` is unavailable or fails with a LOAS/ACL restriction such
    as `RPC_RESTRICTIONS_VIOLATION` or `LabelACLRejected`).
