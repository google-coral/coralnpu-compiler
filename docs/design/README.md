# CoralNPU Compiler Design Documents & CL Workflow

This directory holds the architectural documentation and change design plans for
the CoralNPU Compiler. For non-trivial changes (new compiler passes, operation
enablements, third-party patches, or changes larger than ~150 LOC), the project
uses a **Two-CL Spec-Driven Development (SDD)** workflow supported by the
`/design-doc` and `/implement-design` agent skills.

The workflow and authoring/audit standards in
[`.agents/skills/design-doc/references/standards.md`](../../.agents/skills/design-doc/references/standards.md)
and
[`.agents/skills/implement-design/references/standards.md`](../../.agents/skills/implement-design/references/standards.md)
build on three engineering and AI skill sources:
- **Elephants & Goldfish Model (`go/egm-plugin`)**: Dave Rensin's paper,
  *Elephants, Goldfish, and the New Golden Age of Software Engineering*
  (`go/design-is-the-new-code` |
  [Google Research](https://research.google/pubs/elephants-goldfish-and-the-new-golden-age-of-software-engineering/)
  |
  [Medium](https://drensin.medium.com/elephants-goldfish-and-the-new-golden-age-of-software-engineering-c33641a48874))
  — separating context-rich design sessions from isolated zero-history audit
  subagents, blind comprehension checks, the Zoom-Out Standard
  (`go/egm:standard:zoom-out`), and anti-sycophancy protocols.
- **Design Doc Readability (`go/writing-a-design-doc`, `go/tott/773`,
  `go/bluedoc`, `go/contracts`)**: Strict objective isolation, the rational
  decision formula, design by contract, and falsifiable language.
- **Natural-Language-First Technical Writing (`go/speak-like-tl`,
  `go/good-design-docs`, `go/how-to-write-a-design-document`)**: Explaining
  system flows in plain natural language first without code-symbol dumping.

---

## 1. Directory Layout

```
docs/design/
├── README.md               # This workflow guide
├── INDEX.md                # Map of living component docs to source paths
├── <component>.md          # Tier 1: Living component architecture at HEAD
├── changes/
│   ├── TEMPLATE.md         # Template for new change design documents
│   └── <change>.md         # Tier 2: Active, unmerged change designs & plans
└── archive/
    └── <change>.md         # Tier 3: Completed change plans (historical record)
```

| Path                            | Purpose & Lifecycle                                                                                                              |
| :------------------------------ | :------------------------------------------------------------------------------------------------------------------------------- |
| `docs/design/INDEX.md`          | **Component Map**: Maps each living `<component>.md` doc to the source directories and compiler passes it covers.                |
| `docs/design/<component>.md`    | **Living Component Docs**: Describe how a subsystem works at `HEAD` (e.g., `matrix_codegen.md`). No transient task checklists.   |
| `docs/design/changes/<name>.md` | **Active Change Docs**: Created in CL 1 from `changes/TEMPLATE.md` with the design spec, implementation plan, and follow-ups.    |
| `docs/design/archive/<name>.md` | **Archived Change Docs**: Moved from `changes/` via `git mv` in CL 2 to preserve historical options, trade-offs, and notes.      |

---

## 2. Document Structure (`changes/<change>.md`)

Each change document in `docs/design/changes/<change>.md` has three parts (see
[changes/TEMPLATE.md](changes/TEMPLATE.md)):

1. **Part 1 — Design Specification**:
   - Problem statement (isolated from the implementation mechanism), explicit
     Goals, and Non-Goals with 1-line reasons.
   - Plain-language background (linking to `<component>.md`) and preserved
     system/hardware invariants.
   - Alternatives considered (non-strawman options with pros, cons, estimated
     LOC delta, the 3-part rational decision formula, and reconsideration
     triggers).
   - Detailed technical design (steady-state flow, imposed preconditions and
     offered postconditions, signatures, representative examples, and edge
     cases/fallbacks) and positive + negative/fallback verification plan.
2. **Part 2 — Implementation Plan**:
   - Expected affected files.
   - Ordered `- [ ]` implementation tasks for the `code` step (grouped into
     `CL 2a`, `CL 2b`, ... if the implementation is split across multiple CLs).
   - Implementation notes and adaptations recorded during coding.
   - Checklist for the `archive` step (folding into `<component>.md`, updating
     `INDEX.md`, and `git mv` to `archive/`).
3. **Part 3 — Follow-Up Items**:
   - Deferred work (including scope pared back during design to keep the CL
     focused) or adjacent issues discovered during coding.
   - Reviewers inspect and approve these items in Gerrit before the agent files
     them in Buganizer (component `564546727`) during the `file-followups`
     step.

---

## 3. Step-by-Step Workflow & Skill Invocations

```
===============================================================================
                        CL 1: DESIGN CL (`/design-doc`)
===============================================================================
  1. `/design-doc draft`     -> Co-design (verify context, challenge
          │                     assumptions, right-size scope) -> Write
          │                     `changes/<change>.md` (Developer reviews)
          ▼
  2. `/design-doc audit`     -> Spawn isolated narrow-view subagent (no chat
          │                     history; developer reads blind summary &
          │                     findings to verify spec alignment; repeatable)
          ▼
  3. Commit & Push CL 1      -> Gerrit Design Review
          │
          ▼
  4. `/design-doc revise`    -> Address Gerrit comments -> Receive +2 & Submit

===============================================================================
                   CL 2: IMPLEMENTATION CL (`/implement-design`)
===============================================================================
  1. `/implement-design code`           -> Implement tasks & update `<change>.md`
          │                                (Developer reviews locally)
          ▼
  2. `/implement-design audit`          -> Optional isolated narrow-view subagent
          │                                (Code, test integrity, & drift audit)
          ▼
  3. `/implement-design archive`        -> Fold into `<component>.md`, update
          │                                `INDEX.md`, `git mv` to `archive/`
          ▼
  4. Push CL 2 to Gerrit                -> Reviewers inspect code,
          │                                `<component>.md`, and rename-diff on
          │                                `archive/<change>.md` -> Receive +2
          ▼
  5. `/implement-design file-followups` -> File approved Part 3 follow-ups in
                                           Buganizer (`564546727`)
```

---

### Phase 1: CL 1 — Design CL (`/design-doc`)

#### 1. `draft` — Co-Design & Author the Initial Design Document
```text
/design-doc draft zvt_conv_igemm Enable 2D NHWC convolution lowering via
implicit GEMM (b/123456789).
```
- Reads [INDEX.md](INDEX.md) to locate relevant living component docs and
  queries Buganizer component `564546727` (`issues readonly`) for the target
  bug and up to 5 related open issues (or notes in `<change>.md` that issues
  were not searched if the `issues` CLI is unavailable).
- **Context Verification, Assumption Check & Scope Right-Sizing**:
  - Summarizes its understanding of the current code flow back to you and
    challenges assumptions (pushing back on edge cases, register pressure, or
    simpler upstream IREE alternatives).
  - If the requested change is large, proposes either **tightening the spec**
    to a minimal vertical slice (moving the remaining slices of the original
    requirement into **Part 3: Follow-Up Items**) or splitting **Section 2.2**
    into staged implementation CLs (`CL 2a`, `CL 2b`, ...).
- **Stops for your review** before running the audit subagent.

#### 2. `audit` — Run the Isolated Narrow-View Audit Subagent
After you have reviewed the draft, invoke `audit`:
```text
/design-doc audit
```
- Spawns a fresh subagent with **zero chat history** that reads only
  `docs/design/changes/<change>.md`,
  `.agents/skills/design-doc/references/standards.md`, `.agents/AGENTS.md`, and
  any repository files explicitly referenced in `<change>.md`.
- **Focused Document Audit vs. Broad Design Critique**:
  - `/design-doc audit` intentionally gives the isolated subagent a **narrow
    view** (only `<change>.md`,
    `.agents/skills/design-doc/references/standards.md`, `.agents/AGENTS.md`,
    and the files explicitly referenced in `<change>.md`). Its purpose is to
    verify that `<change>.md` is self-contained for a zero-context reader and
    that its cited files, symbols, contracts, and tasks are accurate.
  - It does **not** perform an open-ended critique across the rest of the
    repository. If you want a broad architectural review or want the agent to
    search the wider codebase for better hooks, uncited callers, or simpler
    alternatives, ask the **main agent in your current session** to do that.
- **Developer Verification of the Blind Comprehension Summary (Critical)**:
  - The audit subagent begins its report with a 1-paragraph **Blind
    Comprehension Summary** explaining in its own words how the current system
    works and what the proposed change does based *only* on `<change>.md`.
  - **Your role as the developer is to carefully read this summary and the
    audit findings to verify that they match your actual intent.** Because the
    audit subagent has no memory of your earlier chat discussion, any
    misunderstanding or wrong assumption in its summary proves that
    `<change>.md` is still ambiguous or missing context that an implementer or
    Gerrit reviewer would also miss.
- **Critical Audit & Codebase Verification**: The subagent checks referenced
  files and symbols, edge cases, rule compliance, scope sizing, and 1-pass
  implementation readiness. The main agent then verifies every objection
  against the actual codebase, labeling each finding as **[Verified]** or
  **[Refuted by `<file>:<line>`]**.
- **No automatic edits & repeatable**: Reports numbered findings and waits for
  your direction; can be run as many times as needed until the blind summary
  and readiness checks align with your intent.

#### 3. `revise` — Update from Inline `A:` Notes or Gerrit Comments
```text
/design-doc revise
```
or with Gerrit reviewer feedback:
```text
/design-doc revise Address Gerrit feedback: <paste comments>. Move dilated
convolution support to Part 3: Follow-Up Items.
```

Once CL 1 receives `+2` in Gerrit, start a **fresh agent session** for CL 2.

---

### Phase 2: CL 2 — Implementation CL (`/implement-design`)

#### 1. `code` — Implementation & Local Review
In a fresh session (with your CL 2 branch checked out), run:
```text
/implement-design code zvt_conv_igemm
```
- Implements the tasks in **Section 2.2** (working in Bazel `--config=dev`
  first and syncing `CMakeLists.txt`, `refresh_compile_commands` in
  `BUILD.bazel`, and `README.md` once the solution is approved, per
  `.agents/AGENTS.md`).
- **Adapt, Inform, Document**: The agent has freedom to adapt to the real code
  or propose a cleaner solution, informing you of changes and updating
  `<change>.md` (including **Section 2.3: Implementation Notes & Adaptations**).
- Logs any newly discovered out-of-scope issues under **Part 3: Follow-Up
  Items** in `<change>.md`.

#### 2. `audit` — Isolated Code, Style & Spec-Drift Audit
Before or during your local code review, you can run a stateless code audit:
```text
/implement-design audit [<change>]
```
- Spawns a fresh subagent (with no chat history and a narrow view) that
  inspects the current working-tree diff (`git diff`), `<change>.md`,
  `.agents/skills/implement-design/references/standards.md`,
  `.agents/AGENTS.md`, and any referenced files to produce a 1-paragraph summary
  of what the diff actually implements and check:
  1. **4-Way Bidirectional Spec-to-Code Drift**: Specified but missing/shortcut
     requirements (including completed tasks in **Section 2.2** still left
     unchecked `- [ ]`), implemented behavior that diverges from `<change>.md`
     (or adaptations missing from **Section 2.3**), undocumented observable
     scope creep (vs. internal refactor invariance), and symbol/terminology
     drift. Read the subagent's summary to verify that the implemented behavior
     matches the specification's intent.
  2. **Contract-Derived Test Coverage & Correctness**: Positive tests for every
     **Offered Postcondition**, negative/fallback tests for every **Imposed
     Precondition**, logic/boundary/resource safety, and anti-test-hacking
     compliance.
  3. **Code Style & Documentation Compliance**: Compliance with all build,
     C/C++, MLIR/LLVM, and comment-density rules in `.agents/AGENTS.md`.
- Reports verified numbered findings without modifying files automatically.
  (For a broader code review across the rest of the repository, ask the main
  agent in your current session to double-check itself.)

#### 3. `archive` — Fold into Living Doc & Move to Archive
When ready to prepare the final implementation CL for Gerrit:
```text
/implement-design archive
```
- Folds lasting architectural changes (including non-obvious **"Why Not X"**
  rejected-alternative rationales) into `docs/design/<component>.md`, and
  updates [INDEX.md](INDEX.md) if a component doc was added or its scope
  changed.
- Runs `git mv docs/design/changes/<change>.md docs/design/archive/<change>.md`
  so Gerrit shows a clean rename-diff against CL 1.
- Runs `./scripts/format-code.sh`.

#### 4. `file-followups` — File Approved Part 3 Follow-Ups in Buganizer (`564546727`)
Once CL 2 receives `+2` in Gerrit (including **Part 3: Follow-Up Items**):
```text
/implement-design file-followups zvt_conv_igemm
```
- Checks Buganizer component `564546727` (`issues readonly search`) for
  existing duplicates.
- Presents a draft table of tickets for your explicit confirmation, then files
  them via `issues mutate create` and reports the `b/<id>` links (or presents
  the suggested issues to open manually if the `issues` CLI is unavailable).

---

## 4. General Usage & Git Conventions

1. **Current Worktree & Git Responsibility**:
   - Both `/design-doc` and `/implement-design` operate directly on the
     **current Git worktree**. They do **not** create or switch branches, make
     Git commits, or push to Gerrit automatically.
   - Check out the target branch or worktree (for example, a branch for CL 1,
     or a branch based on CL 1 / `master` for CL 2) before invoking a skill,
     and handle committing and pushing to Gerrit yourself.

2. **How to Interact Within vs. Across Sessions**:
   - **Within the same session (Regular Prompts)**: Once you invoke a skill step
     (such as `/design-doc draft` or `/implement-design code`), the skill stays
     loaded in the session context. On subsequent turns in that session, you can
     request edits to the design doc or code using **regular prompts** without
     re-typing the `/design-doc` or `/implement-design` command, or by adding
     inline `A:` comments in `<change>.md`. The agent will keep `<change>.md`
     synchronized with any requested code changes.
   - **In a new session (or after `/compact`)**: Invoke the skill and step name
     (for example, `/design-doc revise` or
     `/implement-design code [<change>] <feedback>`) on the first turn so the
     new session loads the skill and the target design document (even if it was
     already moved to `docs/design/archive/<change>.md` during CL 2 review).

3. **Short Name / Automatic Path Resolution**:
   - When invoking `/design-doc` or `/implement-design`, you **never** need to
     type the full path `docs/design/changes/<change>.md`.
   - Pass just the short `<change>` name (for example, `zvt_conv_igemm`), or
     omit the name completely when only one active change doc exists in
     `docs/design/changes/`.
