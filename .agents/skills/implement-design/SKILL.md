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
  approved Section 4 follow-ups in Buganizer component 564546727 without
  implementing them).
---

# Design Implementation & Finalization Workflow (`/implement-design`)

Use this skill during **CL 2** (or staged `CL 2a`, `CL 2b`, ...) to implement,
audit, and finalize an approved design from `./docs/design/changes/<change>.md`
and `./docs/design/<component>.md`.

> [!IMPORTANT]
> **Standards & Project Rules**:
> - Read and apply
>   [`references/standards.md`](references/standards.md)
>   and [`../../AGENTS.md`](../../AGENTS.md) across all steps.

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

| Invocation                                       | Purpose                                                                                                   |
| :----------------------------------------------- | :-------------------------------------------------------------------------------------------------------- |
| `/implement-design code [<change>]`              | Implement tasks on top of the initial CL 2 commit, sync build/doc files, and update `<change>.md`.        |
| `/implement-design audit [<change>] [<commits>]` | Spawn an isolated narrow-view subagent (no chat history) for a code, style, and semantic-drift audit.     |
| `/implement-design archive [<change>]`           | Verify `<component>.md` sync, `git mv` `<change>.md` to `archive/`, and format code.                      |
| `/implement-design file-followups [<change>]`    | Deduplicate and file approved Section 4 follow-ups into Buganizer component `564546727`.                  |

If invoked without a step keyword, default to `code` (unless the prompt
requests `audit`, `archive`, or `file-followups`).

---

## Step 1: `code` — Implementation & Developer Review

1. **Load Only the Approved Design Context**:
   - Read `./docs/design/changes/<change>.md`, the updated
     `./docs/design/<component>.md` living docs (via `./docs/design/INDEX.md`),
     [`references/standards.md`](references/standards.md),
     and [`../../AGENTS.md`](../../AGENTS.md).
   - Do **not** run `audit` or `archive` automatically during `code`.
2. **Execute Tasks & Maintain Doc Synchronicity**:
   - Implement the tasks in **Section 3** (for a multi-CL chain `CL 2a .. CL 2n`,
     work bottom-up on the current `[WIP]` stage commit `CL 2k`, amend `CL 2k`
     removing `[WIP]` while preserving its `Change-Id`, and rebase
     `CL 2(k+1) .. CL 2n` on top), verify tests, handle tactical vs.
     architectural adaptations, log out-of-scope discoveries to **Section 4**,
     and check off verified tasks (`- [x]`) following **Section 1 (Spec-Driven
     Implementation & Adaptation Standards)** of
     [`references/standards.md`](references/standards.md)
     and [`../../AGENTS.md`](../../AGENTS.md).
   - On any follow-up prompts in the same session (or when re-invoked in a new
     session, including after `archive` moved the doc to `archive/<change>.md`),
     keep `<change>.md` and `<component>.md` synchronized with the code.
3. **Next Step**:
   - End your response with a **`Next Step:`** prompt suggesting:
     - Run `/implement-design audit` for an isolated code, style, and
       semantic-drift audit, or
     - Run `/implement-design archive` to finalize CL 2 for Gerrit (or
       `/implement-design code` for the next stage `CL 2(k+1)` in a multi-CL
       chain).

---

## Step 2: `audit` — Isolated Code, Style & Semantic-Drift Audit

Run this step when requested (repeatable; each call spawns a fresh, stateless
subagent; for a broader codebase review, the developer can ask the main agent
in the current session).

1. **Resolve the Implementation Diff Command (`<diff_cmd>`)**:
   - Determine the `git diff` command before spawning the subagent (do not rely
     on bare `git diff`, which is empty when code is committed):
     - **Explicit `<commits>` argument**: `git diff <base_sha>..<tip_sha>` (or
       `git diff <commit>~1..<commit>`).
     - **Uncommitted code changes on top of the initial CL 2 commit**: If
       `git diff HEAD` touches source/test files outside `docs/design/`, find
       `<cl1_sha>` via
       `git log -n 1 --diff-filter=A --format=%H -- docs/design/changes/<change>.md`
       (falling back to `HEAD~1`) and use `git diff <cl1_sha>` so the diff
       includes both the initial CL 2 `<component>.md` changes and the working
       tree code edits.
     - **Committed code on top of the CL 1 design doc**: Find `<cl1_sha>` via
       `git log -n 1 --diff-filter=A --format=%H -- docs/design/changes/<change>.md`
       (falling back to `BASE_SHA` in `./agent-temp/<change>.commit` or
       `HEAD~1`), and use `git diff <cl1_sha>`.
2. **Spawn an Isolated Audit Subagent (No Chat History, Narrow View)**:
   - Launch a fresh, zero-history subagent (`invoke_subagent`) reading **only**:
     - `./docs/design/changes/<change>.md` (or `archive/<change>.md`)
     - `./docs/design/<component>.md`
     - `./.agents/skills/implement-design/references/standards.md`
     - `./.agents/AGENTS.md`
     - The output of `<diff_cmd>`, the modified source/test files, and any
       repository files explicitly referenced in `<change>.md` or
       `<component>.md`
   - Instruct it to perform a read-only audit (without editing files) using
     **Section 2.1** of
     [`references/standards.md`](references/standards.md).
3. **Fact-Check & Report Findings (No Automatic Edits)**:
   - Follow **Section 2.2 (Main-Agent Fact-Checking Protocol)** of
     [`references/standards.md`](references/standards.md).
   - End your response with a **`Next Step:`** prompt suggesting:
     - Specify which findings to fix, or
     - Run `/implement-design archive` when ready to finalize CL 2 for Gerrit.

---

## Step 3: `archive` — Coherence Pass on `<component>.md` & Move `<change>.md` to Archive

Run this step in the final implementation CL (`CL 2`, or the last stage `CL 2n`
of a multi-CL chain) before Gerrit upload:

1. **Coherence Pass & Adaptation Sync in `./docs/design/<component>.md` (`+ INDEX.md`)**:
   - Do a final pass over `./docs/design/<component>.md` to merge and smooth any
     separate per-stage paragraphs/bullets added across `CL 2a .. CL 2n` (and
     any adaptations made during `code`) into unified, cohesive `HEAD` prose,
     following **Section 3 (Living Component Doc Verification & Archival
     Standards)** of
     [`references/standards.md`](references/standards.md).
2. **Archive the Change Doc via `git mv`, Format & Suggest Next Step**:
   - Move `<change>.md` using `git mv` so Gerrit preserves rename history:
     ```bash
     mkdir -p docs/design/archive
     git mv ./docs/design/changes/<change>.md ./docs/design/archive/<change>.md
     ```
   - Mark the final sync item in **Section 3** as `- [x]`, remove
     `./agent-temp/<change>.commit` if present
     (`rm -f ./agent-temp/<change>.commit`), and run `./scripts/format-code.sh`.
   - End your response with a **`Next Step:`** prompt showing:
     - Amend the CL 2 commit (`git commit --amend`, removing the `[WIP]` prefix
       and initial-review note while preserving its Gerrit `Change-Id`) and push
       Patchset 2+ as Ready for Review:
       ```bash
       git push -o ready origin HEAD:refs/for/master
       ```
     - Once CL 2 receives `+2` in Gerrit (if **Section 4** has unchecked
       follow-up items): `/implement-design file-followups <change>`.

---

## Step 4: `file-followups` — File Approved Follow-Ups in Buganizer (`564546727`)

> [!IMPORTANT]
> **Do NOT implement Section 4 follow-up items in code.** This step only files
> deferred follow-ups in Buganizer after CL 2 receives `+2` in Gerrit.

1. **Deduplicate, Confirm & File**:
   - Read unchecked `- [ ]` items in **Section 4: Follow-Up Items** of
     `./docs/design/archive/<change>.md` (or `changes/<change>.md`).
   - If `/google/bin/releases/issues-cli/issues` is available (pass
     `BypassSandbox: true` if present in the tool schema), search for open
     duplicates via `issues readonly search "componentid:564546727 status:open
     <keywords>" --limit 5`, present a summary table (Title, Priority, Duplicate
     check result, and 1–2 sentence description referencing the Gerrit CL or
     archive doc path) for **explicit developer authorization**, and create the
     approved tickets via `issues mutate create --component_id=564546727
     --title="<title>" --priority=<P1|P2|P3> --type=BUG
     --description_file=./agent-temp/.bug_desc.md` (or present the suggested
     tickets for manual filing if the `issues` CLI is unavailable or fails with
     a LOAS/ACL restriction such as `RPC_RESTRICTIONS_VIOLATION` or
     `LabelACLRejected`).
