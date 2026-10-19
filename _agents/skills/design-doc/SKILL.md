---
name: design-doc
description: >-
  Authors, critiques, and refines change design documents in
  ./docs/design/changes/<change>.md (CL 1 of the project CL workflow).
  Supports three named steps: `draft` (query Buganizer 564546727, verify
  codebase understanding, challenge assumptions, right-size scope, and draft
  the design doc), `critique` (spawn a repeatable isolated critic subagent with
  no chat history for blind comprehension, critical review, and codebase
  verification), and `revise` (update the design doc from inline A: comments
  or Gerrit feedback).
---

# Change Design Document Workflow (`/design-doc`)

Use this skill during **CL 1** of the project CL workflow to draft, critique,
or revise a change design document in `./docs/design/changes/<change>.md`.

> [!IMPORTANT]
> **Design Phase & Git Boundary**:
> - During CL 1, do NOT implement production code changes. Only create or edit
>   `./docs/design/changes/<change>.md` (and optional header, `.td`, or test
>   stubs if explicitly requested by the developer). Keep all tasks in
>   **Part 2** unchecked (`- [ ]`).
> - Operate directly on the current Git worktree. Do NOT switch branches,
>   create Git commits, or push to Gerrit unless explicitly asked by the
>   developer.

---

## Resolving `<change>` (No Full Path Required)

The developer never needs to type the full path `./docs/design/changes/...`:
- **Short Name**: If the developer passes `<change>` or `<change>.md` (for
  example, `zvt_conv_igemm`), resolve it to
  `./docs/design/changes/<change>.md`.
- **Omitted Name**:
  - During `draft`: derive a concise `snake_case` name `<change>` from the
    feature description or bug title.
  - During `critique` or `revise`: list `./docs/design/changes/*.md` (excluding
    `TEMPLATE.md`). If only one active change doc exists, use it automatically.
    If multiple exist, check `git status` for the modified doc or ask the
    developer which one to target.

---

## Named Steps

Invoke `/design-doc` with a descriptive step name:

| Invocation                          | When to Use                                                                                         |
| :---------------------------------- | :-------------------------------------------------------------------------------------------------- |
| `/design-doc draft [<change>] ...`  | Co-design with the developer, author `changes/<change>.md`, and stop for developer review.          |
| `/design-doc critique [<change>]`   | Spawn an isolated critic subagent with no chat history (run **only after** review; repeatable).     |
| `/design-doc revise [<change>] ...` | Update the doc from inline `A:` notes or Gerrit reviewer comments.                                  |

If the developer invokes `/design-doc` without a step keyword, infer the step
from their prompt (for example, a new feature description runs `draft`; asking
to check `A:` comments or reviewer feedback runs `revise`), but **never** run
`critique` automatically during `draft`.

---

## Step 1: `draft` — Co-Design & Author the Initial Design Document

1. **Identify `<change>` and Related Living Component Doc(s)**:
   - Resolve `./docs/design/changes/<change>.md`.
   - Read `./docs/design/INDEX.md` first to locate which living component
     doc(s) in `./docs/design/<component>.md` (such as `matrix_codegen.md` or
     `tiling_levels_explained.md`) this change affects, and read those component
     docs before scanning raw source files.

2. **Query Buganizer Component `564546727` (Read-Only)**:
   - Use `/google/bin/releases/issues-cli/issues readonly` (run with
     `BypassSandbox: true` per Google CLI auth rules):
     - If the developer provides `b/<id>`, read the issue and its comments:
       ```bash
       /google/bin/releases/issues-cli/issues readonly render <id>
       /google/bin/releases/issues-cli/issues readonly list-updates <id>
       ```
     - Search for up to 5 related open issues in component `564546727`:
       ```bash
       /google/bin/releases/issues-cli/issues readonly search \
         "componentid:564546727 status:open <keywords>" --limit 5
       ```
   - Briefly highlight any closely related open bugs so the developer can
     decide whether to include them in **Goals**, **Non-Goals**, or **Part 3:
     Follow-Up Items**.

3. **Context Verification, Assumption Check & Scope Right-Sizing**:
   - Inspect the relevant files in `compiler/`, `runtime/`, `crt/`,
     `pjrt_plugin/`, or `tests/` to ground the design in existing classes,
     MLIR passes, and IREE hooks.
   - Unless the developer explicitly asks you to write the full `<change>.md`
     file immediately in one shot, **pause before generating the full
     document** and present a concise alignment check:
     1. **Baseline Understanding**: Summarize the current code flow and
        hardware/IR constraints so the developer can correct any
        misunderstandings early.
     2. **Challenge Assumptions**: Do not passively agree with suboptimal
        assumptions. Push back on unhandled edge cases, register/DTCM/ITCM
        pressure, or third-party patch bloat, and propose 2–3 concrete design
        options (including simpler upstream IREE/MLIR alternatives) with
        trade-offs.
     3. **Scope Right-Sizing & CL Splitting**: If the requested change is large
        (for example, >300–400 LOC or spanning multiple independent ops,
        passes, or components), proactively propose one of two ways to keep CLs
        small and reviewable:
        - **Option A — Tighten the Spec & Defer Follow-Ups (Preferred)**:
          Narrow the current design doc to a minimal, end-to-end vertical
          slice, and place the remaining slices of the original requirement
          into **Part 3: Follow-Up Items** so they are reviewed in Gerrit and
          filed in Buganizer (`564546727`) for subsequent design cycles.
        - **Option B — Multi-CL Implementation Plan (`CL 2a`, `CL 2b`, ...)**:
          If the whole architecture must be designed and reviewed together in
          CL 1, structure **Section 2.2 (Implementation Tasks)** into separate,
          independently testable implementation CLs (`CL 2a`, `CL 2b`, ...).
     4. **Clarifying Questions**: Ask any remaining scope or design questions.

4. **Write `./docs/design/changes/<change>.md` and Stop for Review**:
   - Once the approach is aligned, copy the structure from
     `./docs/design/changes/TEMPLATE.md` and populate **Part 1 (Design
     Specification)**, **Part 2 (Implementation Plan)**, and **Part 3
     (Follow-Up Items)** following `./_agents/AGENTS.md` and Google Markdown
     style guidelines (active voice, concise prose wrapped at 80 columns, and
     space-aligned single-line-per-row tables).
   - **Do NOT spawn the critic subagent yet.** Present the drafted file link to
     the developer so they can review it and add inline `A:` comments first.

---

## Step 2: `critique` — Run the Isolated Critic Subagent

Run this step **only when explicitly requested** after the developer has
reviewed the draft. This step can be invoked **multiple times** across the life
of the design doc; each invocation spawns a fresh, stateless subagent.

1. **Spawn an Isolated Subagent (No Chat History)**:
   - Call `invoke_subagent` (`TypeName: "self"`, `Model: "inherit"`). Because
     the subagent starts in a new conversation, it has **zero chat history**
     and cannot rely on unstated assumptions from earlier turns.
   - Instruct the subagent to read **only**:
     - `./docs/design/changes/<change>.md`
     - `./_agents/AGENTS.md`
     - The repository source files explicitly referenced in `<change>.md`
   - Instruct the subagent to perform a read-only audit (it must **not** edit
     `<change>.md`) covering three sections:
     1. **Blind Comprehension Summary ("Explain It Back")**: In 1 short
        paragraph, explain in its own words how the current system works and
        what the proposed change does, based *strictly* on what is written in
        `<change>.md`. Flag any place where it had to guess unstated context.
     2. **Critical Design & Codebase Audit**: Actively test the design for
        weaknesses:
        - Do all file paths, C++ symbols, MLIR ops, passes, and Bazel targets
          referenced in `<change>.md` actually exist at `HEAD`?
        - Are there unhandled edge cases, fallback gaps, pass-ordering bugs, or
          DTCM/ITCM/register-spill risks?
        - Does the design violate `./_agents/AGENTS.md` (such as unnecessary
          third-party patches, hardcoded op lists instead of MLIR interfaces,
          sub-byte truncation, or long verbatim C++ bodies in Markdown)?
     3. **Scope & 1-Pass Implementation Readiness**:
        - Does **Part 2** specify every affected file, signature, task order,
          and verification command needed to implement the change without
          guessing?
        - Is the change too large for a single implementation CL? If so,
          recommend either tightening the spec (moving later slices to
          **Part 3: Follow-Up Items**) or splitting **Section 2.2** into
          staged implementation CLs (`CL 2a`, `CL 2b`, ...).

2. **Verify Critic Findings Against the Codebase (No Automatic Edits)**:
   - Do **not** automatically modify `./docs/design/changes/<change>.md` when
     the critic subagent finishes.
   - **Fact-Check the Critic**: Before reporting to the developer, verify the
     critic's objections against the actual codebase:
     - Check if the **Blind Comprehension Summary** misunderstood any part of
       the design (which indicates `<change>.md` is missing context).
     - Label each critic finding (`#1`, `#2`, ...) as **[Verified]** or
       **[Refuted by `<file>:<line>`]** (if the critic missed an existing
       guarantee in the codebase; for refuted points, suggest whether to
       clarify `<change>.md` or note the rebuttal under *Alternatives
       Considered*).
   - Present the **Blind Comprehension Summary** prominently at the top of your
     response (so the developer can read it to verify that a zero-context
     reader's understanding matches their actual intent), followed by the
     verified findings list, and wait for their confirmation on which findings
     to apply.

---

## Step 3: `revise` — Address Developer `A:` Comments or Gerrit Feedback

1. **Inline `A:` Comments**:
   - Read `./docs/design/changes/<change>.md`, locate all lines starting with
     `A:`, apply the requested changes, and remove the resolved `A:` lines.
2. **Gerrit Reviewer Feedback**:
   - Update `./docs/design/changes/<change>.md` to address the reviewer
     comments provided by the developer.
   - Move any valid but out-of-scope reviewer ideas into **Part 3: Follow-Up
     Items** so they are preserved and visible in the next Gerrit patchset.
3. **Transition to CL 2**:
   - Once CL 1 receives `+2` in Gerrit, remind the developer to start a
     **fresh session** and invoke `/implement-design code <change>`.
