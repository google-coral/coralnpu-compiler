---
name: design-doc
description: >-
  Authors, audits, refines, and commits change design documents in
  ./docs/design/changes/<change>.md alongside living component doc updates in
  ./docs/design/<component>.md and boundary interface changes (.h / .td).
  Supports five named steps: `draft` (query Buganizer 564546727, verify codebase
  understanding, challenge assumptions, right-size scope, and draft
  `<change>.md` + `<component>.md` + interface updates), `draft-from-commits`
  (abstract and critique pre-code design doc(s) from an existing commit or
  commit range), `audit` (spawn a repeatable isolated narrow-view subagent with
  no chat history for blind comprehension, readability, and specification
  verification), `revise` (update the docs and interfaces from inline A:
  comments or Gerrit feedback), and `commit` (create or amend the chained CL 1
  `<change>.md` commit and initial WIP CL 2 `<component>.md` + interface
  commit).
---

# Change Design Document Workflow (`/design-doc`)

Use this skill during the **Design Phase (CL 1 + initial CL 2)** of the project
CL workflow to draft, audit, revise, and commit a change design brief in
`./docs/design/changes/<change>.md` alongside its living component architecture
updates in `./docs/design/<component>.md` and boundary interface changes (`.h`
headers and `.td` TableGen files).

> [!IMPORTANT]
> **Standards, Rules & Design-Phase Boundary**:
> - Read and apply
>   [`references/standards.md`](references/standards.md)
>   and [`../../AGENTS.md`](../../AGENTS.md) across all steps.
> - During the design phase, do **not** implement `.cpp`/`.cc`/`.c` production
>   bodies or tests: only create or edit `./docs/design/changes/<change>.md`
>   (for CL 1), `./docs/design/<component>.md` (`+ ./docs/design/INDEX.md` if
>   needed), and cross-file boundary interface files (`.h` headers and `.td`
>   TableGen definitions, with full doc comments, for the initial `[WIP]` CL 2),
>   keeping all **Section 3** tasks unchecked (`- [ ]`).

---

## Resolving `<change>` (No Full Path Required)

- **Short Name**: Resolve `<change>` or `<change>.md` (e.g., `zvt_conv_igemm`)
  to `./docs/design/changes/<change>.md`.
- **Omitted Name**:
  - `draft` / `draft-from-commits`: Derive a concise `snake_case` name from the
    feature description, bug title, or commit subject(s).
  - `audit` / `revise` / `commit`: List `./docs/design/changes/*.md` (excluding
    `TEMPLATE.md`). Use the sole active doc automatically, or check `git status`
    / ask the developer if multiple exist.

---

## Named Steps

| Invocation                                                  | When to Use                                                                                                                                      |
| :---------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------- |
| `/design-doc draft [<change>] ...`                          | Co-design with the developer, author `changes/<change>.md`, update `<component>.md` + `.h`/`.td` interfaces, and stop for review.                |
| `/design-doc draft-from-commits [<commits>] [<change>] ...` | Abstract and critique pre-code `changes/<change>.md` + `<component>.md` updates from an existing commit range (`<commits>`).                     |
| `/design-doc audit [<change>]`                              | Spawn an isolated narrow-view subagent with no chat history (run **only after** review; repeatable).                                             |
| `/design-doc revise [<change>] ...`                         | Update `changes/<change>.md`, `<component>.md`, and `.h`/`.td` interfaces from inline `A:` notes or Gerrit reviewer comments.                    |
| `/design-doc commit [<change>]`                             | Commit `changes/<change>.md` as CL 1 and `<component>.md` (+ `INDEX.md` and `.h`/`.td` interfaces) as the chained `[WIP]` CL 2 (or amend CLs).   |

If invoked without a step keyword, infer `draft`, `draft-from-commits`,
`revise`, or `commit` from the prompt, but **never** run `audit` or `commit`
automatically during `draft`, `draft-from-commits`, or `revise`.

---

## Step 1: `draft` (and `draft-from-commits`) — Co-Design & Author the Initial Design Docs

> [!NOTE]
> **When Running `draft-from-commits`**: Read and apply
> [`references/draft-from-commits.md`](references/draft-from-commits.md) on top
> of the steps below.

1. **Identify `<change>` and Related Living Component Doc(s)**:
   - Resolve `./docs/design/changes/<change>.md`.
   - Read `./docs/design/INDEX.md` and the relevant
     `./docs/design/<component>.md` living docs before scanning source files.
2. **Query Buganizer Component `564546727` (Read-Only)**:
   - If `/google/bin/releases/issues-cli/issues` exists, run it (passing
     `BypassSandbox: true` if present in the tool schema): read any cited
     `b/<id>` (`issues readonly render <id>` and `issues readonly list-updates
     <id>`) and search for up to 5 related open issues (`issues readonly search
     "componentid:564546727 status:open <keywords>" --limit 5`), highlighting
     candidates for **Goals**, **Non-Goals**, or **Section 4: Follow-Up Items**.
     If the `issues` CLI is unavailable or fails with a LOAS/ACL restriction
     (`RPC_RESTRICTIONS_VIOLATION` or `LabelACLRejected`), skip the lookup and
     note in `<change>.md` that Buganizer was not searched.
3. **Alignment Check & Scope Right-Sizing**:
   - Inspect relevant files and search for reusable utilities.
   - Follow **Section 1.2 (Pre-Draft Alignment Check & Scope Right-Sizing)** of
     [`references/standards.md`](references/standards.md)
     to align with the developer before generating the documents.
4. **Update `./docs/design/<component>.md` & `.h`/`.td` Interfaces, Write `./docs/design/changes/<change>.md`, and Stop for Review**:
   - Update `./docs/design/<component>.md` (and `./docs/design/INDEX.md` if
     adding a component doc or expanding its scope) in-place with the timeless
     post-change component architecture, pipeline table, and IR examples.
   - If the design adds, modifies, or deletes cross-file boundary interfaces
     (`.h` headers or `.td` TableGen definitions), apply those interface-only
     changes (with complete header documentation comments per
     [`../../AGENTS.md`](../../AGENTS.md); never `.cpp` implementation bodies or
     pass-internal `.cpp` helpers).
   - Copy `./docs/design/changes/TEMPLATE.md` to
     `./docs/design/changes/<change>.md` and populate **Sections 1–4** as a
     concise (80–150 line) change brief following **Sections 2–3** of
     [`references/standards.md`](references/standards.md)
     and [`../../AGENTS.md`](../../AGENTS.md) without duplicating
     `<component>.md` or pasting `.h`/`.td` code blocks.
   - **Do NOT spawn the audit subagent or commit yet.** Present links to
     `changes/<change>.md`, `<component>.md`, and any modified `.h`/`.td` files,
     and end your response with a **`Next Step:`** prompt suggesting:
     - Review the files (optionally adding inline `A:` comments and running
       `/design-doc revise`), or
     - Run `/design-doc audit` for an isolated narrow-view audit.

---

## Step 2: `audit` — Run the Isolated Narrow-View Audit Subagent

Run this step **only when explicitly requested** after developer review (for a
broad critique across the wider codebase, the developer can ask the main agent
in the current session). This step is repeatable; each call spawns a fresh,
stateless subagent.

> [!IMPORTANT]
> **Commit-Sourced Design Docs (`./agent-temp/<change>.commit`)**: If
> `./agent-temp/<change>.commit` exists, follow **Section 3** of
> [`references/draft-from-commits.md`](references/draft-from-commits.md) so the
> subagent runs inside an isolated `<base_sha>` snapshot directory.

1. **Spawn an Isolated Subagent (No Chat History, Narrow View)**:
   - Launch a fresh subagent with **zero chat history** (or ask the developer to
     run `/design-doc audit` in a separate session if subagents are
     unavailable).
   - Instruct the subagent to read **only**:
     - `./docs/design/changes/<change>.md`
     - Any modified `./docs/design/<component>.md` (and `./docs/design/INDEX.md`)
     - Any modified `.h` or `.td` interface files
     - `./.agents/skills/design-doc/references/standards.md`
     - `./.agents/AGENTS.md`
     - Any repository files explicitly referenced in `<change>.md` or
       `<component>.md`
   - Instruct it to perform a read-only audit (without editing files) using
     **Section 4.1** of
     [`references/standards.md`](references/standards.md).
2. **Fact-Check & Report Findings (No Automatic Edits)**:
   - Follow **Section 4.2 (Main-Agent Fact-Checking Protocol)** of
     [`references/standards.md`](references/standards.md).
   - End your response with a **`Next Step:`** prompt suggesting:
     - Specify which findings to apply (or add inline `A:` comments and run
       `/design-doc revise`), or
     - Run `/design-doc commit` if the design is ready to commit.

---

## Step 3: `revise` — Address Developer `A:` Comments or Gerrit Feedback

1. **Contradiction Gate & Edits**:
   - Check for contradictions first per **Section 1.3** of
     [`references/standards.md`](references/standards.md).
   - Apply inline `A:` notes (removing resolved `A:` lines) or Gerrit reviewer
     feedback across `./docs/design/changes/<change>.md`,
     `./docs/design/<component>.md`, and any modified `.h`/`.td` interface files
     following [`references/standards.md`](references/standards.md), moving
     valid out-of-scope reviewer ideas to **Section 4: Follow-Up Items**.
2. **Next Step**:
   - End your response with a **`Next Step:`** prompt suggesting:
     - Run `/design-doc audit` to re-verify after major edits, or
     - Run `/design-doc commit` to create or amend the CL 1 and `[WIP]` CL 2
       commits.

---

## Step 4: `commit` — Create or Update Chained CL 1 & Initial CL 2 Commits

Run this step when explicitly requested via `/design-doc commit [<change>]`:

1. **Pre-Commit Sanity Check**:
   - Verify that no unresolved inline `A:` comments remain in
     `./docs/design/changes/<change>.md`, `./docs/design/<component>.md`, or
     modified `.h`/`.td` interface files.
2. **If `./agent-temp/<change>.commit` Exists (`draft-from-commits` Flow)**:
   - Follow **Section 4** of
     [`references/draft-from-commits.md`](references/draft-from-commits.md):
     commit `docs/design/changes/<change>.md` as **CL 1** on `<base_sha>` with
     the existing implementation commit(s) rebased on top, inform the developer
     that `docs/design/<component>.md` (+ `INDEX.md`) has changes to fold into
     their implementation commit(s), suggest the specific target commit, and
     offer to fold it in for them.
3. **Standard Greenfield Flow (Chained CL 1 + Initial `[WIP]` CL 2 / `CL 2a..2n`)**:
   - **First-Time Commit**:
     1. Stage and commit `./docs/design/changes/<change>.md` as **Commit 1
        (CL 1)** with title `[<tag>][design] Add design doc for <change>`.
     2. Create the initial `[WIP]` implementation commit(s) on top of Commit 1
        (1 commit `CL 2` for a single-CL change, or `n` stacked commits `CL 2a`,
        `CL 2b`, ..., `CL 2n` when **Section 3** of `<change>.md` is grouped
        into multiple CLs):
        - Partition the `./docs/design/<component>.md` (`+ INDEX.md`) and
          `.h`/`.td` boundary interface changes across the stage(s) (using
          `git commit --allow-empty` if a stage only touches `.cpp` files). For
          multi-CL changes, keep each stage's `<component>.md` additions in
          separate paragraphs/bullets/rows so the commits rebase cleanly without
          merge conflicts; `/implement-design archive` in the final stage
          (`CL 2n`) will merge them into cohesive text.
        - Format each initial implementation commit message with a `[WIP]`
          prefix on the title line and an opening body paragraph stating that it
          must be amended with the code implementation:
          ```text
          [WIP][<tag>] <Title of Change or Stage>

          Initial CL 2 containing only the `<component>.md` and interface
          changes for CL 1 design review. Amend this commit with the actual code
          implementation, archive `docs/design/changes/<change>.md` to
          `docs/design/archive/` (in the final stage), and remove the `[WIP]`
          prefix before submitting.
          ```
   - **Re-Commit After `revise` (Amending In-Place)**:
     - If the CL 1 and initial `[WIP]` CL 2 (`CL 2a..2n`) commits already exist,
       fold any working-tree updates to `changes/<change>.md` into CL 1 and any
       updates to `<component>.md` (+ `INDEX.md` and `.h`/`.td` interfaces) into
       the corresponding CL 2 commit(s), **preserving existing commit messages
       and `Change-Id` trailers**.
   - **Report Commits, Gerrit Push Command & Next Step (`-o wip` for CL 2)**:
     - Show `git log -n <n+1> --oneline` and end your response with a **`Next
       Step:`** prompt showing:
       - Push CL 1 as Ready for Review and the initial CL 2 commit(s) (`n = 1`
         for a single CL 2) as Work-In-Progress (`-o wip`):
         ```bash
         git push origin HEAD~<n>:refs/for/master && git push -o wip origin HEAD:refs/for/master
         ```
       - To address Gerrit review comments on CL 1: `/design-doc revise`
       - Once CL 1 receives `+2` and is submitted, start a **fresh session** on
         CL 2 and run `/implement-design code <change>` (or
         `/implement-design audit <change>` for `draft-from-commits`).
