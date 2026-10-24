---
name: implement-design
description: >-
  Implements an approved change design document from
  ./docs/design/changes/<change>.md (CL 2 of the project CL workflow).
  Supports four descriptive steps: `code` (implement tasks, sync build/doc
  files, and update the design doc), `audit` (spawn an isolated narrow-view
  subagent with no chat history for a code, style, and spec-drift audit),
  `archive` (fold lasting architecture into <component>.md and git mv
  <change>.md to archive/), and `file-followups` (file approved Part 3
  follow-ups in Buganizer component 564546727 without implementing them).
---

# Design Implementation & Finalization Workflow (`/implement-design`)

Use this skill during **CL 2** (or staged `CL 2a`, `CL 2b`, ...) to implement,
audit, and finalize an approved design document from
`./docs/design/changes/<change>.md`.

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

| Invocation                               | Purpose                                                                                                |
| :--------------------------------------- | :----------------------------------------------------------------------------------------------------- |
| `/implement-design code [<change>]`      | Implement tasks, adapt flexibly, sync build/doc files, and update `<change>.md`.                       |
| `/implement-design audit [<change>]`     | Spawn an isolated narrow-view subagent (no chat history) for a code, style, and spec-drift audit.      |
| `/implement-design archive [<change>]`   | Fold lasting design into `<component>.md`, `git mv` `<change>.md` to `archive/`, and format code.      |
| `/implement-design file-followups [<change>]` | Deduplicate and file approved Part 3 follow-ups into Buganizer component `564546727`.             |

If invoked without a step keyword, default to `code` (unless the prompt
requests `audit`, `archive`, or `file-followups`).

---

## Step 1: `code` — Implementation & Developer Review

1. **Load Only the Approved Design Context**:
   - Read `./docs/design/changes/<change>.md`, linked
     `./docs/design/<component>.md` living docs (via `./docs/design/INDEX.md`),
     [`references/standards.md`](references/standards.md),
     and [`../../AGENTS.md`](../../AGENTS.md).
   - Do **not** run `audit` or `archive` automatically during `code`.
2. **Execute Tasks & Maintain Spec Synchronicity**:
   - Implement the tasks in **Section 2.2** (or the current stage `CL 2a`,
     `CL 2b`, ...), verify tests, handle tactical vs. architectural adaptations,
     log out-of-scope discoveries to **Part 3**, and check off verified tasks
     (`- [x]`) following **Section 1 (Spec-Driven Implementation & Adaptation
     Standards)** of
     [`references/standards.md`](references/standards.md)
     and [`../../AGENTS.md`](../../AGENTS.md).
   - On any follow-up prompts in the same session (or when re-invoked in a new
     session, including after `archive` moved the doc to `archive/<change>.md`),
     keep `<change>.md` (and `<component>.md` if already archived) synchronized
     with the code.

---

## Step 2: `audit` — Isolated Code, Style & Spec-Drift Audit

Run this step when requested (repeatable; each call spawns a fresh, stateless
subagent; for a broader codebase review, the developer can ask the main agent
in the current session).

1. **Spawn an Isolated Audit Subagent (No Chat History, Narrow View)**:
   - Launch a fresh, zero-history subagent (`invoke_subagent`) reading **only**:
     - `./docs/design/changes/<change>.md` (or `archive/<change>.md`)
     - `./.agents/skills/implement-design/references/standards.md`
     - `./.agents/AGENTS.md`
     - The current `git diff`, the modified source/test files, and any
       repository files explicitly referenced in `<change>.md`
   - Instruct it to perform a read-only audit (without editing files) using
     **Section 2.1** of
     [`references/standards.md`](references/standards.md).
2. **Fact-Check & Report Findings (No Automatic Edits)**:
   - Follow **Section 2.2 (Main-Agent Fact-Checking Protocol)** of
     [`references/standards.md`](references/standards.md).

---

## Step 3: `archive` — Fold into Living Doc & Move to Archive

Run this step in the final implementation CL before Gerrit upload:

1. **Fold Lasting Architecture into `./docs/design/<component>.md`**:
   - Update `./docs/design/<component>.md` and `./docs/design/INDEX.md`
     following **Section 3 (Living Component Doc Promotion Standards)** of
     [`references/standards.md`](references/standards.md).
2. **Archive the Change Doc via `git mv` & Format**:
   - Move `<change>.md` using `git mv` so Gerrit preserves rename history:
     ```bash
     mkdir -p docs/design/archive
     git mv ./docs/design/changes/<change>.md ./docs/design/archive/<change>.md
     ```
   - Mark **Section 2.4** items as `- [x]` and run `./scripts/format-code.sh`.

---

## Step 4: `file-followups` — File Approved Follow-Ups in Buganizer (`564546727`)

> [!IMPORTANT]
> **Do NOT implement Part 3 follow-up items in code.** This step only files
> deferred follow-ups in Buganizer after CL 2 receives `+2` in Gerrit.

1. **Deduplicate, Confirm & File**:
   - Read unchecked `- [ ]` items in **Part 3: Follow-Up Items** of
     `./docs/design/archive/<change>.md` (or `changes/<change>.md`).
   - Follow the **Buganizer Component `564546727` (`issues` CLI)** convention in
     [`../../AGENTS.md`](../../AGENTS.md) to search for open duplicates, present
     a summary table (Title, Priority, Duplicate check result, and 1–2 sentence
     description referencing the Gerrit CL or archive doc path) for **explicit
     developer authorization**, and create the approved tickets.
