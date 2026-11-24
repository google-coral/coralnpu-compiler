---
name: roadmap
description: >-
  Authors, challenges, exports to Google Docs, and merges back multi-change
  topic roadmaps in ./docs/design/<topic>_roadmap.md (one level above
  individual change design briefs in ./docs/design/changes/<change>.md).
  Supports five named steps: `draft`, `draft-from-commits`, `challenge`,
  `to-doc`, and `from-doc`.
---

# Topic Roadmap Workflow (`/roadmap`)

Use this skill to plan, verify, and collaborate on multi-change engineering
efforts in `./docs/design/<topic>_roadmap.md` **one level above individual
change design briefs** (`./docs/design/changes/<change>.md`). Each planned
`<change>` later feeds into `/design-doc draft <change>`.

> [!IMPORTANT]
> **Roadmap-Phase Boundary**:
> - Always apply [`../../AGENTS.md`](../../AGENTS.md) and
>   [`references/standards.md`](references/standards.md).
> - Never modify source code, tests, `.h`/`.td` boundary interfaces, or
>   `docs/design/changes/<change>.md` during `/roadmap`: only create or update
>   `./docs/design/<topic>_roadmap.md` (plus temporary export/baseline files in
>   `./agent-temp/`).
> - Never execute `git commit` or `git push` during `/roadmap`; suggest
>   copy-pasteable commands when ready.

---

## Resolving `<topic>`, Lifecycle & Named Steps

- **Resolving `<topic>`**: Strip `_roadmap` or `.md` to resolve
  `./docs/design/<topic>_roadmap.md`. If omitted, use the sole
  `./docs/design/*_roadmap.md` file for `challenge` / `to-doc` / `from-doc`—or
  for `draft` when it holds inline `A:` comments or matches `<prompt>`—and
  otherwise derive a `snake_case` topic name (asking if multiple exist).
- **Updating & Lifecycle**:
  - Update `<topic>_roadmap.md` directly from chat feedback within a session, or
    add inline `A:` notes and re-run `/roadmap draft [<topic>]`.
  - Keep `<topic>_roadmap.md` decoupled from individual `CL 2` commits so
    parallel tracks never conflict during `/implement-design archive`. Once all
    tracks finish, delete (`git rm`) or archive `<topic>_roadmap.md`.

| Invocation                                          | When to Use                                                                                                                               |
| :-------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------- |
| `/roadmap draft [<topic>] [<sources/prompt>]`       | Gather sources, investigate `HEAD`, align via interview, and write or update `./docs/design/<topic>_roadmap.md`.                          |
| `/roadmap draft-from-commits [<commits>] [<topic>]` | Abstract and critique a pre-code roadmap anchored at `<base_sha>` from an existing prototype commit range (`<commits>`).                  |
| `/roadmap challenge [<topic>]`                      | Spawn **one** isolated zero-history subagent with wide codebase search to verify completeness, `HEAD` grounding, sizing, and boundaries.  |
| `/roadmap to-doc [<topic>] [<doc_url_or_id>]`       | Export `<topic>_roadmap.md` to a Google Doc for team review, or ask how to update an existing Doc *(if Google Docs tools are available)*. |
| `/roadmap from-doc [<topic>] [<doc_url_or_id>]`     | Merge edits from the linked Google Doc back into `<topic>_roadmap.md` without `HEAD` checks *(if Google Docs tools are available)*.       |

Run only the requested step; **never** chain steps automatically in the same
turn.

---

## Step 1: `draft` (and `draft-from-commits`) — Gather, Align & Author

> [!NOTE]
> **When Running `draft-from-commits` (or When `.commit` Exists)**: If running
> `draft-from-commits` or if `./agent-temp/<topic>_roadmap.commit` exists, also
> read and apply **Sections 1–2** of
> [`references/draft-from-commits.md`](references/draft-from-commits.md) so
> reconnaissance and Section 1 stay anchored at `BASE_SHA`.

- **1. Gather Sources & Investigate `HEAD`**:
  - Read [`references/standards.md`](references/standards.md) (§1–3) and
    `./docs/design/INDEX.md` (plus `./docs/design/<topic>_roadmap.md` if it
    already exists), then gather inputs and investigate `HEAD` (or `BASE_SHA`
    when `.commit` exists) per **Section 1.4** of
    [`references/standards.md`](references/standards.md).
- **2. Baseline Synthesis & 1-at-a-Time Interview**:
  - Check for contradictions first
    ([`references/standards.md`](references/standards.md) §1.2).
  - If creating a new roadmap—or if the prompt or inline `A:` notes leave open
    scope, track, or ordering choices (and the developer did not ask to write in
    one shot)—present the **Sources Read & `HEAD` Synthesis** together with
    **Question 1** of [`interview.md`](../design-doc/references/interview.md) §2
    (inviting additional sources before answering) and align one question at a
    time. Apply straightforward targeted `A:` updates directly.
- **3. Author or Update `./docs/design/<topic>_roadmap.md`**:
  - Write or update `./docs/design/<topic>_roadmap.md` per **Sections 2–3** of
    [`references/standards.md`](references/standards.md) (removing resolved `A:`
    notes).
  - **Stop for review** and end with a **`Next Step:`** prompt suggesting
    `/roadmap challenge <topic>`, `/roadmap to-doc <topic>` *(if Google Docs
    tools are available)*, inline `A:` refinements (`/roadmap draft <topic>`),
    or `/design-doc draft <change>`.

---

## Step 2: `challenge` — Isolated Zero-History Subagent Verification

Run **only when explicitly requested** (repeatable):

- **1. Snapshot Isolation (If Prototype Baseline Exists)**:
  - If `./agent-temp/<topic>_roadmap.commit` exists, follow **Section 3** of
    [`references/draft-from-commits.md`](references/draft-from-commits.md) so
    the subagent runs inside an isolated `<base_sha>` snapshot directory.
- **2. Spawn an Isolated Zero-History Subagent**:
  - Launch **one** fresh, zero-history `self` subagent in **read-only mode**
    (per [`../../AGENTS.md`](../../AGENTS.md)) with wide codebase search to read
    `./docs/design/<topic>_roadmap.md`, `./docs/design/INDEX.md`, referenced
    `./docs/design/<component>.md` docs,
    [`references/standards.md`](references/standards.md), and
    [`../../AGENTS.md`](../../AGENTS.md), and evaluate the **5-Point Challenge
    Rubric** (**Section 4.1** of
    [`references/standards.md`](references/standards.md)) without editing files.
- **3. Verify, Apply Clear-Cut Fixes & Resolve Open Trade-Offs**:
  - Follow **Section 4.2** of
    [`references/standards.md`](references/standards.md) and
    [`interview.md`](../design-doc/references/interview.md) to fact-check
    findings, apply clear-cut `<topic>_roadmap.md` updates directly, resolve
    open trade-offs interactively, and end with the conditional **`Next Step:`**
    prompt (suggesting `/roadmap to-doc <topic>` *(if Google Docs tools are
    available)*, `/design-doc draft <change>`, and optional commit/push commands
    per **Section 5.3**).

---

## Step 3: `to-doc` — Export to Google Doc for Team Review

Run **only when explicitly requested** *(if Google Docs tools are available)*:

- Follow **Section 5.1** of
  [`references/standards.md`](references/standards.md) (and
  [`interview.md`](../design-doc/references/interview.md) §2 if a linked Doc
  already exists) to export `./docs/design/<topic>_roadmap.md` to a converted
  Google Doc, record the `**Review Doc**:` link in Section 1, and end with a
  **`Next Step:`** prompt to share the Doc and later run `/roadmap from-doc
  <topic>` *(if Google Docs tools are available)*.

---

## Step 4: `from-doc` — Merge Edits from Google Doc

Run **only when explicitly requested** *(if Google Docs tools are available)*:

- Follow **Section 5.2** of
  [`references/standards.md`](references/standards.md) (and
  [`interview.md`](../design-doc/references/interview.md) §2 for unaccepted
  suggestions, open comment threads, or merge conflicts) to merge remote edits
  into `./docs/design/<topic>_roadmap.md` without running `HEAD` checks, then
  end with a **`Next Step:`** prompt recommending `/roadmap challenge <topic>`
  and printing optional commit/push commands per **Section 5.3**.
