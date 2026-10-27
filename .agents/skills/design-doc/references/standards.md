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
   - Target length: **80–150 lines** (hard ceiling **250 lines**). Follow the
     4-section layout in
     [`TEMPLATE.md`](../../../docs/design/changes/TEMPLATE.md) without
     duplicating `<component>.md` or copy-pasting `.h`/`.td` code blocks:
     - **Section 1 (Summary & Motivation)**: State the problem and target
       outcome without embedding the implementation mechanism in the problem
       statement (`go/writing-a-design-doc`), followed by verifiable **Goals**
       and plausible **Non-Goals** with 1-line reasons.
     - **Section 2 (Proposed Change & Component Doc Delta)**: Summarize the
       design delta, key invariants, and fallback behavior, linking to the
       updated `<component>.md` and `.h`/`.td` files in CL 2. Place
       **Alternatives Considered** *after* the proposal (`go/tott/773`), stating
       why the chosen design wins, what downside it accepts, and why realistic
       alternatives were rejected.
     - **Section 3 (Implementation & Verification Plan)**: One unified checklist
       (`egm-draft` Step 5.1) of implementation steps (with affected files
       inline), positive and negative/fallback tests, and the final sync step.
       Never split affected files and tasks into separate lists.
     - **Section 4 (Follow-Up Items)**: Deferred work for Buganizer
       (`564546727`).
2. **Living Component Doc & Boundary Interfaces (`docs/design/<component>.md` +
   `.h`/`.td` — Chained Initial `[WIP]` CL 2 / `CL 2a..2n`)**:
   - **`docs/design/<component>.md`** (`+ INDEX.md` if updated): Holds the
     timeless post-change architecture (summary, pipeline table/diagram, rules
     and invariants, IR examples, test files, and `TODO(...)` limits). Integrate
     changes in-place (`"One Contract, One Place"`) without temporal phrasing
     (*"This CL adds..."*, *"Previously..."*). For multi-CL chains (`CL 2a ..
     CL 2n`), keep each stage's `<component>.md` additions in separate
     paragraphs, bullets, or rows so per-stage commits rebase without conflicts
     (merged by `/implement-design archive` in `CL 2n`).
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
   - **Readability, Altitude & Length Violations**: Flag C++-in-prose
     transcription, cross-section repetition, sentences > 35 words or walls of
     backticks, duplication between `<change>.md` and `<component>.md`, length
     exceeding 80–150 lines (250 hard ceiling), or undefined terms that forced
     guessing.
2. **Zoom-Out, Technical Accuracy & Trade-Off Audit**:
   - Verify cited file paths, symbols, interfaces, and build targets exist;
     check whether the design solves the root problem without YAGNI or missed
     upstream/codebase utility reuse (or whether commits in `draft-from-commits`
     can be simplified); and verify realistic alternatives, invariants, and
     graceful fallbacks ([`../../../AGENTS.md`](../../../AGENTS.md)).
3. **Scope & Implementation Plan Check**:
   - Verify **Section 3** is a single unified checklist (with target files
     inline) including positive and negative/fallback tests, and recommend Scope
     Right-Sizing (§1.2) if too large for a single CL.

### 4.2 Shared Main-Agent Fact-Checking Protocol (`audit` & `challenge`)

When an isolated `audit` or `challenge` subagent finishes:

1. **Do not modify any files automatically.**
2. **Fact-check every finding against the codebase**: Discard low-signal
   nitpicks, deduplicate overlapping points, and label each finding (`#1`,
   `#2`, ...) as **[Verified]** or **[Refuted by `<file>` (`<symbol>`)]**
   (preserving `[Lens: ...]` tags when present).
3. **Present findings & wait for direction**: For `audit`, present the
   1-paragraph Blind Summary prominently first, followed by the verified
   findings list (using [`interview.md`](interview.md) if any finding opens a
   non-trivial design choice).
