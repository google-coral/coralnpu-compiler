# Design Document Authoring, Interaction & Audit Standards

This document defines the authoring, structural, prose, and review standards
used across `/design-doc` and `/implement-design`. Project-specific rules live
in [`../../../AGENTS.md`](../../../AGENTS.md).

---

## 1. Interaction, Scope Right-Sizing & Contradiction Gate (`draft` & `revise`)

1. **Default to Stress-Testing (Critique First, No Flattery)**:
   - Never agree by default or use validating praise (*"great idea"*,
     *"brilliant"*, *"spot-on"*, *"excellent"*) or filler apologies.
   - Lead with what is missing, risky, or could fail, and present the strongest
     counter-argument before affirming a design choice.
2. **Scope Right-Sizing**:
   - During `draft` and `draft-from-commits` (alongside
     [`lenses.md`](lenses.md) and [`interview.md`](interview.md)), if a change
     is large (>300–400 LOC or spanning multiple independent features), propose:
     - **Option A — Tighten the Spec & Defer Follow-Ups (Preferred)**: Narrow
       the change to a minimal end-to-end vertical slice and move remaining
       slices to **Section 4: Follow-Up Items**.
     - **Option B — Multi-CL Implementation Plan (`CL 2a`, `CL 2b`, ...)**:
       Review the full architecture in CL 1 + initial CL 2 and group **Section
       3** into independently testable implementation CLs.
3. **Contradiction Auditing Gate**:
   - If a new developer request or inline `A:` note contradicts an earlier
     design goal or an established system invariant, **do not modify any files
     in that turn**. Point out the contradiction in plain text and resolve it
     via [`interview.md`](interview.md) first.

---

## 2. Two-CL Division of Responsibility & Structural Standards

During the design phase, the agent updates **two complementary sets of files**
committed as a 2-CL chain by `/design-doc commit`:

1. **Change Design Brief (`docs/design/changes/<change>.md` — CL 1)**:
   - Target length: **100–200 lines** (hard ceiling **300 lines**). Follow the
     4-section layout in
     [`TEMPLATE.md`](../../../docs/design/changes/TEMPLATE.md) without
     duplicating `<component>.md` or copy-pasting `.h`/`.td` code blocks:
     - **Section 1 (Summary & Motivation — *Today -> Gap -> Core Idea ->
       Outcome Goals*)**:
       - *Paragraph 1 (Today & Gap)*: Explain how the system works today on a
         concrete scenario and why it breaks or falls short
         (`go/writing-a-design-doc`).
       - *Paragraph 2 (Core Idea)*: State the conceptual fix in 2–3 plain
         sentences without dumping pass, helper, or field inventories.
       - *Goals (Observable Outcomes) & Non-Goals*: State what capabilities,
         guarantees, or metrics the change achieves (never a file-by-file or
         symbol-by-symbol task list), followed by plausible **Non-Goals** with
         1-line reasons.
     - **Section 2 (Proposed Change & Component Doc Delta — *End-to-End Story
       First, Details Second*)**:
       - *Opening End-to-End Walkthrough*: Start Section 2 with 1–2 short prose
         paragraphs tracing **one concrete input or example end to end** (entry
         point -> transformation -> output) so the reader gets the complete
         mental model before any topic bullets.
       - *Key Mechanisms & Component Boundaries*: Follow with 2–4 focused
         bullets or sub-paragraphs explaining each stage and why it works that
         way, linking to the updated sections of `<component>.md` and `.h`/`.td`
         files in CL 2.
       - *Invariants, Fallbacks & Edge Cases*: Group strict invariants, fallback
         paths, and corner-case rules at the end of the proposal (or in a
         compact table) so they never derail the main walkthrough prose.
       - *Alternatives Considered*: Place *after* the proposal (`go/tott/773`),
         stating why the chosen design wins, what downside it accepts, and why
         realistic alternatives were rejected.
     - **Section 3 (Implementation & Verification Plan)**: One unified checklist
       (`egm-draft` Step 5.1) of implementation steps (with affected files
       inline), positive and negative/fallback tests, and the final sync step.
       Never split affected files and tasks into separate lists.
     - **Section 4 (Follow-Up Items)**: Single tracker for all follow-ups from
       this change during `draft`, `revise`, and `code`. Tag each item for
       `<component>.md` TODOs by default (merged into `<component>.md` during
       `/implement-design archive`), and reserve Buganizer (`2161972`) for
       cross-cutting tasks, bugs with an unknown source, multi-component
       architectural changes, or work needing an assignee, priority, or
       `b/<id>`.
2. **Living Component Doc & Boundary Interfaces (`docs/design/<component>.md` +
   `.h`/`.td` — Chained Initial `[WIP]` CL 2 / `CL 2a..2n`)**:
   - **`docs/design/<component>.md`** (`+ INDEX.md` if updated): Holds the
     timeless post-change architecture (summary, pipeline table/diagram, rules
     and invariants, IR examples, test files, and a `## TODOs` section of `- [ ]`
     items). Integrate changes in-place (`"One Contract, One Place"`) without
     temporal phrasing (*"This CL adds..."*, *"Previously..."*). During `draft`,
     `revise`, and `code`, tick (`- [x]`) any existing TODO item in
     `<component>.md` resolved by the change (unticking back to `- [ ]` if scope
     shrinks), and do not add new TODO items or delete ticked `- [x]` lines
     until `/implement-design archive`. For multi-CL chains (`CL 2a .. CL 2n`),
     keep each stage's `<component>.md` additions in separate paragraphs,
     bullets, or rows so per-stage commits rebase without conflicts (merged by
     `/implement-design archive` in `CL 2n`).
   - **Boundary Interface Files (`.h` Headers & `.td` TableGen Definitions)**:
     Update added, modified, or deleted cross-file `.h` declarations or `.td`
     definitions directly (with full doc comments per
     [`../../../AGENTS.md`](../../../AGENTS.md)) for Initial CL 2. Never touch
     `.cpp`/`.cc`/`.c` files during `/design-doc` or move pass-internal `.cpp`
     helpers into `.h` headers just to expose them in CL 2.

---

## 3. Technical Altitude & Prose Craft (`go/speak-like-tl`)

1. **Altitude Control: Principles & Examples by Default, Compact Tables
   Opt-In**:
   - **Never Transcribe C++ Control Flow into Prose**: Do not serialize C++
     boolean expressions, variable assignments, or call trees into English
     sentences (e.g., never write *"When `!outputChannel.empty() ||
     isChannelLast`, set `vectorParallelSizes[secondPar] = ...`"*).
   - **Default (Prose Principles, Invariants & Examples)**: State *what* rule or
     invariant holds, *why* it is needed, how unsupported cases fall back, and
     illustrate with 1–2 concrete examples (such as an IR snippet or tile trace
     in `<component>.md`).
   - **Opt-In (Compact Tables or Short Pseudocode)**: When requested or when a
     design hinges on a multi-category heuristic matrix, use a compact Markdown
     table (`< 15` words per cell) or a `<= 10`-line pseudocode block instead of
     multi-bullet prose paragraphs.
2. **Short Sentences, Concrete Examples First, Minimal Backticks**:
   - Aim for **15–25 words** per sentence (split any sentence over ~30–35
     words).
   - Lead with the concrete concept or example (*"Support multi-op reductions
     such as decomposed `linalg.softmax` (`max` -> `exp` + `sum` -> `div`)"*)
     before stating structural conditions.
   - Explain flows by logical component role (*"the DTCM pass"*, *"a contraction
     with reuse"*) with at most 1–2 parenthetical symbol/file pointers per
     paragraph (using symbol names rather than line numbers).
3. **Say It Once & Grounded Natural Voice**:
   - State each requirement, invariant, or file change **once** across the doc;
     never repeat table contents in prose or restate Motivation in Design.
   - Back performance, memory, or code-size claims with concrete bounds (cycles,
     KB, registers) rather than vague adjectives (*"fast"*, *"scalable"*).
   - Name specific actors, use active verbs, allow natural `"we"` and plain
     everyday words (*"simple"*, *"small"*, *"thin tile"*), and avoid AI filler
     (*leverage*, *utilize*, *streamline*, *orchestrate*, *facilitate*, *delve*,
     *holistic*, *seamless*, *robust*, *overarching*, *pivotal*, *paradigm*).

---

## 4. Design Doc Audit Rubric & Shared Fact-Checking Protocol

### 4.1 Audit Rubric for the Isolated Narrow-View Subagent (`/design-doc audit`)

Evaluate `changes/<change>.md`, modified `<component>.md` (`+ INDEX.md`),
modified `.h`/`.td` interfaces, and referenced files across three dimensions:

1. **Blind Comprehension & Readability Check (Human-First Gate)**:
   - **1-Paragraph Blind Summary**: Explain in 1 short paragraph how the current
     system works and what the proposed change does, based *strictly* on
     `<change>.md` and `<component>.md`.
   - **Readability, Narrative Flow, Altitude & Length Violations**: Flag
     `Goals` that read like symbol/file inventories, Section 2 drafts that jump
     into disconnected feature bullets without an opening end-to-end
     walkthrough, corner cases crammed into the main narrative flow,
     C++-in-prose transcription, cross-section repetition, sentences > 35 words
     or walls of backticks, duplication between `<change>.md` and
     `<component>.md`, length exceeding 100–200 lines (300 hard ceiling), or
     undefined terms that forced guessing.
2. **Zoom-Out, Technical Accuracy & Trade-Off Audit**:
   - Verify cited file paths, symbols, interfaces, and build targets exist;
     check whether the design solves the root problem without YAGNI or missed
     upstream/codebase utility reuse (or whether commits in `draft-from-commits`
     can be simplified); and verify realistic alternatives, invariants, and
     graceful fallbacks ([`../../../AGENTS.md`](../../../AGENTS.md)).
3. **Scope, Implementation Plan & TODOs Check**:
   - Verify **Section 3** is a single unified checklist (with target files
     inline) including positive and negative/fallback tests, verify that every
     ticked (`- [x]`) TODO in `<component>.md` is genuinely covered by the
     proposed design (and flag any resolved `- [ ]` item left unticked or any
     new follow-up added prematurely to `<component>.md` instead of Section 4),
     and recommend Scope Right-Sizing (§1.2) if too large for a single CL.

### 4.2 Shared Main-Agent Verification & Two-Tier Resolution Protocol

Subagents remain strictly **read-only** (required for parallel runs and
`<snapshot_dir>` isolation). After subagents finish, the main agent runs the
skill-agnostic core protocol below paired with the active skill's scope block.

#### 4.2.1 Skill-Agnostic Core Protocol

Applies to `/design-doc challenge`, `/design-doc audit`, `/implement-design
audit`, and `/roadmap challenge`:

1. **Codebase & Persisted-Decision Verification**:
   - Fact-check every subagent claim against the codebase, silently drop minor
     low-signal noise, and record refuted claims as `[Refuted by <file>
     (<symbol>)]` for the compact footer.
   - Check every verified finding against the active skill's **persisted
     decision records first** (so anti-oscillation holds across fresh sessions
     and `/compact`) and in-session chat history when present:
     - *Re-litigated choice with no new technical evidence*: Skip it (note
       briefly as `[Already Settled]` in the footer) rather than reversing an
       agreed decision.
     - *Prior-decision conflict backed by new evidence, or iterative
       back-and-forth across passes*: Route to **Tier 2** (never auto-apply).
2. **Partition Verified Findings: Tier 1 (Apply) vs. Tier 2 (Interview)**:
   - **Tier 1 (Apply Directly in Working Tree)**: Edit files strictly within the
     active skill's allowed boundary **only when the change is
     complexity-neutral or simplifying** and preserves agreed goals, contracts,
     and [`../../../AGENTS.md`](../../../AGENTS.md) hard gates.
   - **Coupling Hold Rule**: Hold any Tier 1 edit targeting a section, helper,
     or interface contested by a queued Tier 2 question until that question
     resolves.
   - **Tier 2 (Escalate via [`interview.md`](interview.md) §2)**: Hold file
     edits for open architectural trade-offs, additive machinery, scope shifts,
     contract changes, or prior-decision/oscillation conflicts, and resolve them
     one question at a time.
   - **Convergence / Diminishing-Returns Signal**: When verified findings across
     repeat passes drop to zero—or consist solely of speculative/defensive
     additions or already-settled trade-offs with no concrete invariant breaks,
     correctness bugs, or meaningful simplifications—note convergence and
     recommend advancing to the next workflow step.
3. **Fixed Chat Output Layout**:
   Present results in this order so Tier 2 options are never buried:
   - **1. 1-Paragraph Blind Summary** (`/design-doc audit`, `/implement-design
     audit`, and `/roadmap challenge`) or 1-line **`Lenses Run`** note
     (`/design-doc challenge`).
   - **2. Applied Updates** (omit if none): Compact bulleted summary of Tier 1
     edits applied in the working tree, naming touched files/sections with
     clickable file links and preserving any `[Lens: ...]` or category tags.
   - **3. Refuted & Already-Settled** (omit if empty): Compact 1-line bullets
     (`- **[Refuted by <file> (<symbol>)]**: ...` or `- **[Already Settled]**:
     ...`).
   - **4. Resolution or Next Step**:
     - *If Tier 2 has `>= 1` items*: List the queued Tier 2 topics upfront (if
       `>= 2`) and present **Question 1** per [`interview.md`](interview.md)
       §2, **withholding** any terminal **`Next Step:`** prompt until the final
       Tier 2 answer is applied.
     - *If Tier 2 is empty (or after the final Tier 2 edit)*: State any
       convergence note and end with a terminal **`Next Step:`** prompt
       inviting `git diff` review (and overrides if needed) before advancing to
       the next workflow step.

#### 4.2.2 `/design-doc` Scope Block (`challenge` & `audit`)

1. **Persisted Decision Records**: `### Alternatives Considered`, `Non-Goals`,
   and documented invariants in `changes/<change>.md`.
2. **Allowed Tier 1 Edits** (preserving the Design-Phase Boundary—never touch
   `.cpp`/`.cc`/`.c`):
   - Accuracy, clarity, and structural fixes in `changes/<change>.md` and
     `<component>.md` (`+ INDEX.md`): broken file/symbol citations, C++-in-prose
     rewrites, `<change>.md` vs. `<component>.md` deduplication, `- [x]` TODO
     sync, and adding Section 3 test bullets covering already-specified
     invariants.
   - Simplifications: removing redundant prose/checks or swapping bespoke logic
     for an **exact semantic drop-in** upstream MLIR/IREE utility (identical
     preconditions, outputs, and fallback behavior; no new dependencies).
   - Fail-closed guards **only** for inputs already outside `Goals` (or making
     an already-documented fallback explicit), and appending verified
     out-of-scope follow-ups to Section 4 while leaving `Goals` and Section 2
     untouched.
   - `.h` / `.td` edits **strictly** limited to doc comments, `#include`s,
     typos, and syncing drift already agreed in `<change>.md`.
3. **Mandatory Tier 2 Escalations**: Adding, deleting, or altering any
   cross-file `.h`/`.td` declaration or option; adding a new pass, abstraction
   layer, config knob, or broader generality; any fail-closed guard or Section 4
   deferral that narrows a stated `Goal`; and any competing architectural
   trade-off or prior-decision conflict.
4. **`draft-from-commits` Override**: When `./agent-temp/<change>.commit`
   exists, apply the dual-baseline tier restrictions in
   [`draft-from-commits.md`](draft-from-commits.md) §3.3.
