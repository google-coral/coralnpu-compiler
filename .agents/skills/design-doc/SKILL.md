---
name: design-doc
description: >-
  Authors, audits, and refines change design documents in
  ./docs/design/changes/<change>.md (CL 1 of the project CL workflow).
  Supports three named steps: `draft` (query Buganizer 564546727, verify
  codebase understanding, challenge assumptions, right-size scope, and draft
  the design doc), `audit` (spawn a repeatable isolated narrow-view subagent
  with no chat history for blind comprehension and specification verification),
  and `revise` (update the design doc from inline A: comments or Gerrit
  feedback).
---

# Change Design Document Workflow (`/design-doc`)

Use this skill during **CL 1** of the project CL workflow to draft, audit, or
revise a change design document in `./docs/design/changes/<change>.md`.

> [!IMPORTANT]
> **Standards, Rules & Design-Phase Boundary**:
> - Read and apply
>   [`references/standards.md`](references/standards.md)
>   and [`../../AGENTS.md`](../../AGENTS.md) across all steps.
> - During CL 1, do **not** implement production code changes: only create or
>   edit `./docs/design/changes/<change>.md` (and optional interface/test stubs
>   if explicitly requested), keeping all **Part 2** tasks unchecked (`- [ ]`).

---

## Resolving `<change>` (No Full Path Required)

- **Short Name**: Resolve `<change>` or `<change>.md` (e.g., `zvt_conv_igemm`)
  to `./docs/design/changes/<change>.md`.
- **Omitted Name**:
  - `draft`: Derive a concise `snake_case` name from the feature description or
    bug title.
  - `audit` / `revise`: List `./docs/design/changes/*.md` (excluding
    `TEMPLATE.md`). Use the sole active doc automatically, or check `git status`
    / ask the developer if multiple exist.

---

## Named Steps

| Invocation                         | When to Use                                                                                         |
| :--------------------------------- | :-------------------------------------------------------------------------------------------------- |
| `/design-doc draft [<change>] ...` | Co-design with the developer, author `changes/<change>.md`, and stop for developer review.          |
| `/design-doc audit [<change>]`     | Spawn an isolated narrow-view subagent with no chat history (run **only after** review; repeatable).|
| `/design-doc revise [<change>] ...`| Update the doc from inline `A:` notes or Gerrit reviewer comments.                                  |

If invoked without a step keyword, infer `draft` or `revise` from the prompt,
but **never** run `audit` automatically during `draft`.

---

## Step 1: `draft` — Co-Design & Author the Initial Design Document

1. **Identify `<change>` and Related Living Component Doc(s)**:
   - Resolve `./docs/design/changes/<change>.md`.
   - Read `./docs/design/INDEX.md` and the relevant
     `./docs/design/<component>.md` living docs before scanning source files.
2. **Query Buganizer Component `564546727` (Read-Only)**:
   - Follow the Buganizer `564546727` CLI convention in
     [`../../AGENTS.md`](../../AGENTS.md) to read any cited `b/<id>` and search
     for up to 5 related open issues, highlighting candidates for **Goals**,
     **Non-Goals**, or **Part 3: Follow-Up Items**.
3. **Alignment Check & Scope Right-Sizing**:
   - Inspect relevant files and search (`code_search`) for reusable utilities.
   - Follow **Section 1.2 (Pre-Draft Alignment Check & Scope Right-Sizing)** of
     [`references/standards.md`](references/standards.md)
     to align with the developer before generating the full document.
4. **Write `./docs/design/changes/<change>.md` and Stop for Review**:
   - Copy `./docs/design/changes/TEMPLATE.md` and populate **Parts 1–3**
     following **Sections 2–3** of
     [`references/standards.md`](references/standards.md)
     and [`../../AGENTS.md`](../../AGENTS.md).
   - **Do NOT spawn the audit subagent yet.** Present the file link so the
     developer can review and add inline `A:` comments first.

---

## Step 2: `audit` — Run the Isolated Narrow-View Audit Subagent

Run this step **only when explicitly requested** after developer review (for a
broad critique across the wider codebase, the developer can ask the main agent
in the current session). This step is repeatable; each call spawns a fresh,
stateless subagent.

1. **Spawn an Isolated Subagent (No Chat History, Narrow View)**:
   - Launch a fresh subagent with **zero chat history** (or ask the developer to
     run `/design-doc audit` in a separate session if subagents are
     unavailable).
   - Instruct the subagent to read **only**:
     - `./docs/design/changes/<change>.md`
     - `./.agents/skills/design-doc/references/standards.md`
     - `./.agents/AGENTS.md`
     - Any repository files explicitly referenced in `<change>.md`
   - Instruct it to perform a read-only audit (without editing files) using
     **Section 4.1** of
     [`references/standards.md`](references/standards.md).
2. **Fact-Check & Report Findings (No Automatic Edits)**:
   - Follow **Section 4.2 (Main-Agent Fact-Checking Protocol)** of
     [`references/standards.md`](references/standards.md).

---

## Step 3: `revise` — Address Developer `A:` Comments or Gerrit Feedback

1. **Contradiction Gate & Edits**:
   - Check for contradictions first per **Section 1.3** of
     [`references/standards.md`](references/standards.md).
   - Apply inline `A:` notes (removing resolved `A:` lines) or Gerrit reviewer
     feedback in `./docs/design/changes/<change>.md` following
     [`references/standards.md`](references/standards.md),
     moving valid out-of-scope reviewer ideas to **Part 3: Follow-Up Items**.
2. **Transition to CL 2**:
   - Once CL 1 receives `+2` in Gerrit, remind the developer to start a **fresh
     session** and invoke `/implement-design code <change>`.
