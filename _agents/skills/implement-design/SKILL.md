---
name: implement-design
description: >-
  Implements an approved change design document from
  ./docs/design/changes/<change>.md (CL 2 of the project CL workflow).
  Supports four descriptive steps: `code` (implement tasks, sync build/doc
  files, and update the design doc), `critique` (spawn an isolated subagent
  with no chat history for a critical code, style, and spec-drift review),
  `archive` (fold lasting architecture into <component>.md and git mv
  <change>.md to archive/), and `file-followups` (file approved Part 3
  follow-ups in Buganizer component 564546727 without implementing them).
---

# Design Implementation & Finalization Workflow (`/implement-design`)

Use this skill during **CL 2** (or staged `CL 2a`, `CL 2b`, ...) to implement,
critique, and finalize an approved design document from
`./docs/design/changes/<change>.md`.

> [!IMPORTANT]
> **Worktree & Git Boundary**: Operate directly on the current Git worktree. Do
> NOT create or switch Git branches, create commits, or push to Gerrit
> automatically (except running `git mv` during the `archive` step) unless
> explicitly requested by the developer.

---

## Resolving `<change>` (No Full Path Required)

The developer does not need to specify the full path:
- **Short Name**: If the developer passes `<change>` or `<change>.md` (for
  example, `zvt_conv_igemm`), resolve it to
  `./docs/design/changes/<change>.md` (or fall back to
  `./docs/design/archive/<change>.md` if the doc was already moved to
  `archive/` in CL 2).
- **Omitted Name**: List `./docs/design/changes/*.md` (excluding `TEMPLATE.md`).
  If only one active change doc exists, use it automatically. If none exist in
  `changes/`, check `git status` / `git diff --name-only` for a modified or
  renamed doc under `./docs/design/archive/*.md`. If multiple exist, ask the
  developer which one to use.

---

## Named Steps

Invoke `/implement-design` with a descriptive step name:

| Invocation                                    | Purpose                                                                                                |
| :-------------------------------------------- | :----------------------------------------------------------------------------------------------------- |
| `/implement-design code [<change>]`           | Implement tasks, adapt flexibly, sync build/doc files, and update `<change>.md`.                       |
| `/implement-design critique [<change>]`       | Spawn an isolated subagent (no chat history) for a critical code, style, and spec-drift review.        |
| `/implement-design archive [<change>]`        | Fold lasting design into `<component>.md`, `git mv` `<change>.md` to `archive/`, and format code.      |
| `/implement-design file-followups [<change>]` | Deduplicate and file approved Part 3 follow-ups into Buganizer component `564546727`.                  |

If the developer invokes `/implement-design` without a step keyword, default to
`code` (unless their prompt explicitly asks for `critique`, `archive`, or
`file-followups`).

---

## Step 1: `code` — Implementation & Developer Review

1. **Load Only the Approved Design Context**:
   - Read `./docs/design/changes/<change>.md`, any linked living component docs
     in `./docs/design/<component>.md` (using `./docs/design/INDEX.md` as
     needed), and `./_agents/AGENTS.md`.
   - Do **not** run `critique` or `archive` automatically during the `code`
     step.
   - If **Section 2.2 (Implementation Tasks)** is split into multiple CLs
     (`CL 2a`, `CL 2b`, ...), implement only the tasks for the current CL stage
     (the first stage with unchecked `- [ ]` tasks, unless the developer
     specifies otherwise).

2. **Execute Tasks with "Adapt, Inform, Document" Flexibility**:
   - Following `./_agents/AGENTS.md`, develop and verify the implementation
     using Bazel (`bazel build --config=dev ...` and
     `bazel test --config=dev ...`), and once the solution is reached (or when
     approved by the developer), sync `CMakeLists.txt`,
     `refresh_compile_commands` in `BUILD.bazel`, and `README.md` as needed.
   - Treat `<change>.md` as your blueprint, not a rigid constraint. You have
     freedom to adapt to real code or adopt a cleaner solution:
     - **Local / Tactical Adaptations** (such as helper signatures, touching an
       additional caller or build target, or handling a local IR edge case):
       Implement the improvement, update **Part 1 / Part 2** (including
       **Section 2.3: Implementation Notes & Adaptations**) of
       `./docs/design/changes/<change>.md`, and highlight the adaptation in
       your response to the developer.
     - **Architectural Pivots** (such as changing pass pipeline placement,
       replacing the core algorithm, adding a new `third_party/` patch, or
       changing user-facing flags): Pause code edits, explain the newly
       discovered option and trade-offs to the developer, and update
       `<change>.md` once agreed.

3. **Log Out-of-Scope Discoveries to Part 3 (Follow-Up Items)**:
   - If you discover an adjacent bug, optimization opportunity, or cleanup that
     is outside the scope of the current change, do **not** bloat the CL.
   - Add a concise entry under **Part 3: Follow-Up Items** in
     `./docs/design/changes/<change>.md` and mention it to the developer so
     both the developer and Gerrit reviewers can inspect it.

4. **Check Off Verified Tasks (`- [x]`)**:
   - Mark a task `- [x]` in `./docs/design/changes/<change>.md` only after its
     code is written, its test verification passes, and any design adaptations
     are documented in `<change>.md`.
   - When the tasks are complete (or if the developer adds `A:` comments in
     `<change>.md`), summarize the changes and pause for developer review.

5. **Follow-Up Prompts & Code Review Iterations**:
   - **Same Session (Regular Prompts)**: Once `/implement-design code` has been
     invoked in a session, the developer can request further code changes using
     ordinary prompts without re-typing `/implement-design code`. On every
     follow-up code change in the session, you **must** continue to keep
     `<change>.md` (Part 1, Section 2.3 Adaptations, and Part 3 Follow-Up
     Items) synchronized with the code.
   - **New Session or Post-`archive` Gerrit Review on CL 2**: If the developer
     starts a new session or addresses Gerrit review comments on CL 2 after
     `archive` has already moved the file to
     `./docs/design/archive/<change>.md`, re-invoking
     `/implement-design code [<change>] <feedback>` updates the code,
     `./docs/design/archive/<change>.md`, and `./docs/design/<component>.md`.

---

## Step 2: `critique` — Isolated Code, Style & Spec-Drift Review

Run this step when the developer invokes `/implement-design critique [<change>]`
to review the implementation before or during local review. This step can be
run **multiple times**; each call spawns a fresh, stateless subagent.

1. **Spawn an Isolated Critic Subagent (No Chat History)**:
   - Call `invoke_subagent` (`TypeName: "self"`, `Model: "inherit"`).
   - Instruct the subagent to read:
     - `./docs/design/changes/<change>.md` (or `archive/<change>.md`)
     - `./_agents/AGENTS.md`
     - The full `git diff` and modified source/test files
   - Instruct the subagent to perform a strict, critical, read-only audit (it
     must **not** modify any files) across three dimensions:
     1. **Spec-to-Code Drift & Blind Summary**: Summarize in 1 short paragraph
        what the `git diff` actually implements, and compare it against
        `<change>.md`. Are there undocumented behavior changes, missing
        requirements, or unrecorded adaptations?
     2. **Correctness, Safety & Test Integrity (Anti-Test-Hacking)**:
        - Are there logic bugs, unhandled IR shapes/types, sub-byte bit-width
          truncation risks, missing fallback remarks, or register/DTCM/ITCM
          hazards?
        - **Test Integrity**: Do the tests genuinely exercise the new code path
          (rather than silently taking a fallback or vacuous assertion), and
          were any existing test assertions or `CHECK` lines weakened?
     3. **Code Style & Comment Density (`./_agents/AGENTS.md`)**:
        - Is every added or modified function documented (on the `.h`
          declaration if declared in a header)?
        - Does every non-obvious `if`, `for`, or `while` block have a brief
          comment explaining *why* (rationale and intent)?
        - Does the code use early guard clauses, labeled `#include` groups,
          `/*paramName=*/` literal comments, and MLIR interfaces instead of
          hardcoded op lists?

2. **Fact-Check & Report Findings (No Automatic Code Edits)**:
   - Do **not** automatically edit code or `<change>.md` when the subagent
     finishes.
   - Verify each finding against the codebase, filter out any false positives,
     and present the subagent's 1-paragraph implementation summary alongside
     the numbered findings (`#1`, `#2`, ...) so the developer can verify
     alignment and choose which findings to apply.

---

## Step 3: `archive` — Fold into Living Doc & Move to Archive

Run this step when preparing the final implementation CL for Gerrit upload
(if a change is split across `CL 2a` and `CL 2b`, leave `<change>.md` in
`changes/` during `CL 2a` and run `archive` only in the final `CL 2b`):

1. **Fold Lasting Architecture & Rejected-Alternative Guardrails into
   `./docs/design/<component>.md`**:
   - Update the target living component doc(s) in `./docs/design/<component>.md`
     (or create a new one if introducing a new subsystem) so it accurately
     describes how the component works at `HEAD`.
   - **Preserve "Why Not X" Rationales**: In addition to describing what was
     built, fold in any non-obvious rejected-alternative rationales from
     **Section 1.3 (Alternatives Considered)** or **Section 2.3 (Adaptations)**
     so future developers and agents know why an apparently simpler approach
     does not work.
   - **Update `./docs/design/INDEX.md`**: If a new `<component>.md` was created
     or an existing component doc's scope changed, update the table in
     `./docs/design/INDEX.md`.
   - Do **not** copy transient task checklists into `<component>.md`.
2. **Archive the Change Doc via `git mv`**:
   - Ensure `./docs/design/archive/` exists (`mkdir -p docs/design/archive`) and
     move the completed change doc using `git mv` so Gerrit preserves rename
     history and displays a clean diff against CL 1:
     ```bash
     git mv ./docs/design/changes/<change>.md ./docs/design/archive/<change>.md
     ```
   - Mark the items in **Section 2.4 (Documentation Archive Checklist)** of
     `./docs/design/archive/<change>.md` as `- [x]`.
3. **Format & Audit**:
   - Run `./scripts/format-code.sh`.
   - Verify that `git diff` is consistent with the finalized design in
     `./docs/design/archive/<change>.md` and `./docs/design/<component>.md`.

---

## Step 4: `file-followups` — File Approved Follow-Ups in Buganizer (`564546727`)

> [!IMPORTANT]
> **Do NOT implement Part 3 follow-up items in code.** This step only files the
> deferred follow-up items as Buganizer tickets in component `564546727` after
> the implementation CL receives `+2` in Gerrit.

Run this step when the developer asks to file the reviewed **Part 3: Follow-Up
Items** from `<change>.md` in Buganizer (after CL 2 receives `+2` in Gerrit):

1. **Read & Dedup-Check Follow-Up Items**:
   - Read all unchecked `- [ ]` items in **Part 3: Follow-Up Items** of
     `./docs/design/archive/<change>.md` (or `changes/<change>.md`).
   - For each item, run a read-only search in Buganizer component `564546727`
     (`BypassSandbox: true`) to check if an open issue already covers it:
     ```bash
     /google/bin/releases/issues-cli/issues readonly search \
       "componentid:564546727 status:open <keywords>" --limit 5
     ```

2. **Present Draft Tickets for Explicit Developer Authorization (MANDATORY)**:
   - Creating Buganizer issues (`issues mutate create`) is a mutating write
     action that **requires explicit user authorization**.
   - Present a summary table of the follow-ups to file (Title, Priority,
     Duplicate check result, and 1–2 sentence description referencing the
     Gerrit CL or archive doc path) and ask the developer to confirm filing.

3. **Create Approved Issues in Component `564546727`**:
   - Once authorized by the developer, write each multi-line description to a
     scratch file under `./agent-temp/.bug_desc.md` and run
     (`BypassSandbox: true`):
     ```bash
     /google/bin/releases/issues-cli/issues mutate create \
       --component_id=564546727 \
       --title="<title>" \
       --priority=<P1|P2|P3> \
       --type=BUG \
       --description_file=./agent-temp/.bug_desc.md
     ```
   - Report the created `b/<id>` links back to the developer.
