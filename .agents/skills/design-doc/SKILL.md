---
name: design-doc
description: >-
  Authors, challenges, audits, refines, and commits change design documents in
  ./docs/design/changes/<change>.md alongside living component doc updates in
  ./docs/design/<component>.md and boundary interface changes (.h / .td).
  Supports six named steps: `draft`, `draft-from-commits`, `challenge`, `audit`,
  `revise`, and `commit`.
---

# Change Design Document Workflow (`/design-doc`)

Use this skill during the **Design Phase (CL 1 + initial CL 2)** to draft,
challenge, audit, revise, and commit `./docs/design/changes/<change>.md`
alongside `./docs/design/<component>.md` and boundary interface changes (`.h`
headers and `.td` TableGen files).

> [!IMPORTANT]
> **Design-Phase Boundary**:
> - Always apply [`../../AGENTS.md`](../../AGENTS.md) and the step-specific
>   `references/*.md` files linked below.
> - Never implement `.cpp`/`.cc`/`.c` production bodies or tests during
>   `/design-doc`: only create or edit `docs/design/changes/<change>.md` (CL 1),
>   `docs/design/<component>.md` (`+ docs/design/INDEX.md`), and cross-file
>   `.h`/`.td` boundary interfaces (with complete doc comments, for initial
>   `[WIP]` CL 2), keeping all **Section 3** tasks unchecked (`- [ ]`).

---

## Resolving `<change>` & Named Steps

- **Short / Omitted `<change>`**: Resolve `<change>` to
  `./docs/design/changes/<change>.md`. If omitted, derive a `snake_case` name
  for `draft` / `draft-from-commits`, or use the sole active doc in
  `./docs/design/changes/*.md` (excluding `TEMPLATE.md`) for `challenge` /
  `audit` / `revise` / `commit` (asking if multiple exist).

| Invocation                                                  | When to Use                                                                                                                    |
| :---------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------- |
| `/design-doc draft [<change>] ...`                          | Co-design via parallel lenses & interview, update `<component>.md` + `.h`/`.td`, write `changes/<change>.md`, stop for review. |
| `/design-doc draft-from-commits [<commits>] [<change>] ...` | Abstract and critique pre-code `changes/<change>.md` + `<component>.md` updates from an existing commit range (`<commits>`).   |
| `/design-doc challenge [<change>]`                          | Spawn parallel adversarial Design Lens subagents (with wide codebase search) to red-team the drafted proposal (repeatable).    |
| `/design-doc audit [<change>]`                              | Spawn an isolated narrow-view subagent with zero chat history for blind comprehension and readability checks (repeatable).     |
| `/design-doc revise [<change>] ...`                         | Update `changes/<change>.md`, `<component>.md`, and `.h`/`.td` interfaces from inline `A:` notes or Gerrit reviewer comments.  |
| `/design-doc commit [<change>]`                             | Commit `changes/<change>.md` as CL 1 and `<component>.md` (+ `INDEX.md` and `.h`/`.td`) as chained `[WIP]` CL 2 (or amend).    |

If invoked without a step keyword, infer `draft`, `draft-from-commits`,
`revise`, or `commit` from the prompt, but **never** run `challenge`, `audit`,
or `commit` automatically during `draft`, `draft-from-commits`, or `revise`.

---

## Step 1: `draft` (and `draft-from-commits`) — Co-Design & Author Docs

> [!NOTE]
> **When Running `draft-from-commits`**: Also read and apply **Sections 1–2** of
> [`references/draft-from-commits.md`](references/draft-from-commits.md).

- **1. Reconnaissance & Buganizer Lookup (`2161972`)**:
  - Read `./docs/design/INDEX.md`, the relevant `./docs/design/<component>.md`
    living docs, and target source files.
  - If `/google/bin/releases/issues-cli/issues` exists (pass `BypassSandbox:
    true` if present in the tool schema), read cited `b/<id>` (`issues
    readonly render|list-updates <id>`) and search related open issues
    (`issues readonly search "componentid:2161972 status:open <keywords>"
    --limit 5`); if `issues` is unavailable or fails with a LOAS/ACL
    restriction (`RPC_RESTRICTIONS_VIOLATION` or `LabelACLRejected`), skip it
    and note in `<change>.md` that Buganizer was not searched.
- **2. Baseline Triage, Optional Design-Lens Exploration & Co-Design Interview**:
  - Read [`references/standards.md`](references/standards.md) (§1–3),
    [`references/lenses.md`](references/lenses.md) (§1–3), and
    [`references/interview.md`](references/interview.md).
  - Unless asked to write the docs immediately in one shot:
    1. **Turn 1 — Baseline Primer, Sub-Problem Triage & Exploration Choice**:
       Present the self-contained **Baseline Understanding** primer and
       **Sub-Problem Triage** table (`Straightforward` items with proposed
       defaults vs. `Open` architectural items, without inventing pre-lens
       options for hard open items), then ask **Question 1**
       ([`references/interview.md`](references/interview.md) §1.3) so the
       developer can choose to: (1) lock in straightforward items and run
       targeted Design Lenses on the open items, (2) lock in straightforward
       items and write in their own design outline/draft for the open items,
       (3) run full Design Lenses across all items, or any custom combination.
    2. **If Design Lenses Are Selected (Targeted or Full)**: Spawn the parallel
       isolated **Design Lens** subagents on the selected items and curate the
       top 3–4 `[Lens: ...]` options + Recommended Synthesis + 1-line filtered
       list per [`references/lenses.md`](references/lenses.md) §2–3.
    3. **1-at-a-Time Co-Design Alignment**: Walk down the remaining decision
       tree (from the curated lens options or from any open forks in the
       developer's written outline) **one question at a time** per
       [`references/interview.md`](references/interview.md) §2 until aligned.
- **3. Author `<component>.md`, `.h`/`.td` Interfaces & `changes/<change>.md`**:
  - Update `./docs/design/<component>.md` (`+ INDEX.md`), any cross-file `.h` /
    `.td` boundary interfaces, and `./docs/design/changes/<change>.md` (copied
    from `./docs/design/changes/TEMPLATE.md`) following **Sections 2–3** of
    [`references/standards.md`](references/standards.md).
  - **Stop for review** (do not run `challenge`, `audit`, or `commit`). Link
    the modified files and end with a **`Next Step:`** prompt suggesting
    `/design-doc revise` (after inline `A:` comments), `/design-doc challenge`,
    or `/design-doc audit`.

---

## Step 2: `challenge` — Red-Team with Adversarial Design Lenses

Run **only when explicitly requested** (repeatable):

1. Read and apply **Section 4 (`Adversarial Mode`)** of
   [`references/lenses.md`](references/lenses.md) and
   [`references/interview.md`](references/interview.md) (plus **Section 3** of
   [`references/draft-from-commits.md`](references/draft-from-commits.md) if
   `./agent-temp/<change>.commit` exists) to spawn 4–6 parallel adversarial
   lens subagents with wide codebase search, fact-check findings (`[Verified]`
   vs. `[Refuted]`), and walk through resulting choices via `interview.md`.

---

## Step 3: `audit` — Isolated Narrow-View Blind-Reader Audit

Run **only when explicitly requested** (repeatable):

1. If `./agent-temp/<change>.commit` exists, follow **Section 3** of
   [`references/draft-from-commits.md`](references/draft-from-commits.md) so the
   subagent runs inside an isolated `<base_sha>` snapshot directory.
2. Spawn a fresh, zero-history subagent that inherits local file-reading and
   shell/git capabilities in read-only mode, instructed to read **only**
   `./docs/design/changes/<change>.md`, modified `./docs/design/<component>.md`
   (`+ INDEX.md`), modified `.h`/`.td` interface files,
   `./.agents/skills/design-doc/references/standards.md`, `./.agents/AGENTS.md`,
   and repository files explicitly referenced in `<change>.md` or
   `<component>.md`, and evaluate **Section 4.1** of
   [`references/standards.md`](references/standards.md) without editing files.
3. Follow **Section 4.2** of
   [`references/standards.md`](references/standards.md) to fact-check and report
   findings, ending with a **`Next Step:`** prompt (`/design-doc revise` or
   `/design-doc commit`).

---

## Step 4: `revise` — Address Developer `A:` Comments or Gerrit Feedback

1. Read [`references/standards.md`](references/standards.md) (§1–3) and
   [`references/interview.md`](references/interview.md). Check for
   contradictions first (`standards.md` §1.3), use `interview.md` if feedback
   opens a multi-branch design question, and apply inline `A:` notes (removing
   resolved `A:` lines) or Gerrit feedback across `changes/<change>.md`,
   `<component>.md`, and `.h`/`.td` files (moving out-of-scope ideas to
   **Section 4: Follow-Up Items**).
2. End with a **`Next Step:`** prompt suggesting `/design-doc challenge`,
   `/design-doc audit`, or `/design-doc commit`.

---

## Step 5: `commit` — Create or Update Chained CL 1 & Initial CL 2 Commits

- **1. Sanity Check**: Verify no unresolved inline `A:` comments remain in
  `changes/<change>.md`, `<component>.md`, or modified `.h`/`.td` files.
- **2. If `./agent-temp/<change>.commit` Exists**: Follow **Section 4** of
  [`references/draft-from-commits.md`](references/draft-from-commits.md).
- **3. Standard Greenfield Flow (Chained CL 1 + Initial `[WIP]` CL 2 /
  `CL 2a..2n`)**:
  - **First-Time Commit**:
    1. Commit `./docs/design/changes/<change>.md` as **CL 1** with title
       `[<tag>][design] Add design doc for <change>`.
    2. On top of CL 1, create the initial `[WIP]` commit(s) (`CL 2`, or
       `CL 2a .. CL 2n` when **Section 3** is split into multiple CLs),
       partitioning `./docs/design/<component>.md` (`+ INDEX.md`) and
       `.h`/`.td` interface changes across stages (`git commit --allow-empty`
       if a stage only touches `.cpp` files; keep per-stage `<component>.md`
       additions in separate paragraphs/bullets/rows so rebases stay
       conflict-free):

       ```text
       [WIP][<tag>] <Title of Change or Stage>

       Initial CL 2 containing only the `<component>.md` and interface
       changes for CL 1 design review. Amend this commit with the actual code
       implementation, archive `docs/design/changes/<change>.md` to
       `docs/design/archive/` (in the final stage), and remove the `[WIP]`
       prefix before submitting.
       ```

  - **Re-Commit After `revise`**: Fold working-tree updates to
    `changes/<change>.md` into CL 1 and `<component>.md` (`+ INDEX.md` and
    `.h`/`.td`) into the corresponding CL 2 commit(s), **preserving commit
    messages and `Change-Id` trailers**.
  - **Report & Next Step**: Show `git log -n <n+1> --oneline` and the Gerrit
    push command (`n = 1` for single CL 2):

    ```bash
    git push origin HEAD~<n>:refs/for/master && git push -o wip origin HEAD:refs/for/master
    ```

    Remind the developer to use `/design-doc revise` for CL 1 review comments,
    and once CL 1 lands, start a **fresh session** on CL 2 with
    `/implement-design code <change>`.
