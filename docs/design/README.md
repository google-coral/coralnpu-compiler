# CoralNPU Compiler Design Documents & CL Workflow

This directory holds the architectural documentation and change design plans for
the CoralNPU Compiler. For non-trivial changes (new compiler passes, operation
enablements, third-party patches, or changes larger than ~150 LOC), the project
uses a **Two-CL Spec-Driven Development (SDD)** workflow supported by the
`/design-doc` and `/implement-design` agent skills.

The separation between a context-rich design session and isolated, zero-history
critic subagents is inspired by Dave Rensin's paper, *Elephants, Goldfish, and
the New Golden Age of Software Engineering*
([Google Research](https://research.google/pubs/elephants-goldfish-and-the-new-golden-age-of-software-engineering/)
| [Medium](https://drensin.medium.com/elephants-goldfish-and-the-new-golden-age-of-software-engineering-c33641a48874)).

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
   - Problem statement, explicit Goals, and Non-Goals (including related
     Buganizer issues left out of scope).
   - Background (linking to `<component>.md`) and hardware/IR invariants
     (DTCM/ITCM limits, 32 vector registers, zero register spills).
   - Alternatives considered (with pros, cons, and estimated LOC delta).
   - Detailed technical design (pipeline diagrams, C++ signatures or TableGen
     definitions, and before/after MLIR IR examples) and verification plan.
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
  2. `/design-doc critique`  -> Spawn isolated critic subagent (no chat history)
          │                     (Developer reads blind summary & findings to
          │                     verify spec alignment; repeatable)
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
  2. `/implement-design critique`       -> Optional isolated critic subagent
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
  bug and up to 5 related open issues.
- **Context Verification, Assumption Check & Scope Right-Sizing**:
  - Summarizes its understanding of the current code flow back to you and
    challenges assumptions (pushing back on edge cases, register pressure, or
    simpler upstream IREE alternatives).
  - If the requested change is large, proposes either **tightening the spec**
    to a minimal vertical slice (moving the remaining slices of the original
    requirement into **Part 3: Follow-Up Items**) or splitting **Section 2.2**
    into staged implementation CLs (`CL 2a`, `CL 2b`, ...).
- **Stops for your review** before running the critic subagent.

#### 2. `critique` — Run the Isolated Critic Subagent
After you have reviewed the draft, invoke `critique`:
```text
/design-doc critique
```
- Spawns a fresh subagent with **zero chat history** that reads only
  `docs/design/changes/<change>.md`, `_agents/AGENTS.md`, and cited files.
- **Developer Verification of the Blind Comprehension Summary (Critical)**:
  - The critic subagent begins its report with a 1-paragraph **Blind
    Comprehension Summary** explaining in its own words how the current system
    works and what the proposed change does based *only* on `<change>.md`.
  - **Your role as the developer is to carefully read this summary and the
    critique findings to verify that they match your actual intent.** Because
    the critic subagent has no memory of your earlier chat discussion, any
    misunderstanding or wrong assumption in its summary proves that
    `<change>.md` is still ambiguous or missing context that an implementer or
    Gerrit reviewer would also miss.
- **Critical Audit & Codebase Verification**: The subagent checks codebase
  references, edge cases, rule compliance, scope sizing, and 1-pass
  implementation readiness. The main agent then verifies every critic objection
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
  `_agents/AGENTS.md`).
- **Adapt, Inform, Document**: The agent has freedom to adapt to the real code
  or propose a cleaner solution, informing you of changes and updating
  `<change>.md` (including **Section 2.3: Implementation Notes & Adaptations**).
- Logs any newly discovered out-of-scope issues under **Part 3: Follow-Up
  Items** in `<change>.md`.

#### 2. `critique` — Isolated Code, Style & Spec-Drift Review
Before or during your local code review, you can run a stateless code critic:
```text
/implement-design critique
```
- Spawns a fresh subagent (with no chat history) that inspects `git diff`,
  `<change>.md`, and `_agents/AGENTS.md` to produce a 1-paragraph summary of
  what `git diff` actually implements and check:
  1. **Spec-to-Code Drift**: Any undocumented behavior or unimplemented design
     requirements. Read the critic's summary to verify that the implemented
     behavior matches the specification's intent.
  2. **Correctness, Edge Cases & Test Integrity**: Logic flaws, sub-byte bit
     math, fallback remarks, register/DTCM/ITCM invariants, and verification
     that tests genuinely exercise the new code path without weakening existing
     `CHECK` assertions.
  3. **Code Style & Comment Density**: Header comments on declarations,
     *why*-comments on non-obvious `if`/`for` blocks, early guard clauses, and
     MLIR interfaces over op lists.
- Reports verified numbered findings without modifying files automatically.

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
  them via `issues mutate create` and reports the `b/<id>` links.

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
