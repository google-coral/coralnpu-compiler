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

### 4.2 Shared Main-Agent Fact-Checking Protocol (`audit` & `challenge`)

When an isolated `audit` or `challenge` subagent finishes:

1. **Do not modify any files automatically.**
2. **Fact-check every finding against the codebase**: Discard low-signal
   nitpicks, deduplicate overlapping points, and classify each claim as
   **[Verified]** or **[Refuted by `<file>` (`<symbol>`)]** (preserving
   `[Lens: ...]` tags when present).
3. **Present findings in a clear, scannable layout**:
   - **Blind Summary First (`audit` only)**: Present the 1-paragraph Blind
     Summary prominently at the top.
   - **Verified Findings (`#1`, `#2`, ...)**: Number only verified findings and
     format each with three explicit fields so the proposed edit and
     recommendation are immediately obvious:

     ```markdown
     ### #1: <Short Title> — [Verified] [Lens: <Tag>]
     - **Issue**: 1–2 sentences stating the flaw, ambiguity, drift, or missed
       simplification (with clickable file/symbol links).
     - **Proposed Change**: The exact edit to make, naming the target file(s)
       and section(s) (`<change>.md` §2, `<component>.md`, `<file>.h`, or
       `<file>.cpp`).
     - **Recommendation**: **<Apply | Defer to Section 4 | Skip | Decision Needed>** —
       1-sentence rationale (or numbered options `1. (Recommended) ...`,
       `2. ...` per [`interview.md`](interview.md) when `Decision Needed`).
     ```

   - **Refuted Findings (Separate Compact Footer)**: Group any refuted claims at
     the bottom under **Refuted Findings (No Action Needed)** as 1-line bullets
     (`- **[Refuted by <file> (<symbol>)]**: 1-line claim and why it already
     holds`).
   - **Next Step Prompt**: End by listing the recommended actions by number
     (e.g., *"Reply to apply the recommended changes (`#1`, `#2`), override any
     item by number, or choose an option for `#3`"*).
