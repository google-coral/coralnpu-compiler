# Design Document Authoring, Interaction & Audit Standards

This document defines the general engineering, prose, and review standards used
by both the authoring session (the main agent during `draft`,
`draft-from-commits`, `revise`, and `commit`) and the isolated narrow-view audit
subagent (with zero chat history during `audit`). Project-specific rules live in
[`../../../AGENTS.md`](../../../AGENTS.md).

---

## 1. Interaction, Alignment Check & Scope Right-Sizing (`draft` & `revise`)

1. **Default to Stress-Testing (Critique First, No Flattery)**:
   - Never agree by default or use validating praise (*"great idea"*,
     *"brilliant"*, *"spot-on"*, *"excellent"*) or filler apologies.
   - Lead with what is missing, risky, or could fail, and present the strongest
     counter-argument before affirming a design choice.
2. **Pre-Draft Alignment Check & Scope Right-Sizing**:
   - Proactively infer answers from the codebase and living component docs
     first. Unless asked to write the docs immediately in one shot, pause before
     generating the documents to present:
     1. **Baseline Understanding**: Plain-language summary of the current flow
        and invariants (and which `<component>.md` living doc(s) will be updated
        alongside `changes/<change>.md`).
     2. **Assumptions & Alternatives**: Risks, edge cases, reuse opportunities,
        and 2–3 viable (non-strawman) options with trade-offs.
     3. **Scope Right-Sizing**: If the change is large (>300–400 LOC or spanning
        multiple independent features), propose either:
        - **Option A — Tighten the Spec & Defer Follow-Ups (Preferred)**: Narrow
          the change to a minimal end-to-end vertical slice and move remaining
          slices to **Section 4: Follow-Up Items**.
        - **Option B — Multi-CL Implementation Plan (`CL 2a`, `CL 2b`, ...)**:
          Review the full architecture in CL 1 + initial CL 2 and group
          **Section 3** into independently testable implementation CLs.
     4. **Clarifying Question**: Ask at most **one** unresolved design question
        at a time.
3. **Contradiction Auditing Gate**:
   - If a new developer request or inline `A:` note contradicts an earlier
     design goal or an established system invariant, **do not modify any files
     in that turn**. Point out the contradiction in plain text and ask how to
     resolve it first.

---

## 2. Two-CL Division of Responsibility & Structural Standards

During the design phase, the agent updates **two complementary sets of files**
that are committed as a 2-CL chain by `/design-doc commit` (so `master` never
merges an unimplemented `<component>.md` or header when CL 1 lands, while Gerrit
reviewers of CL 1 can inspect the `<component>.md` and interface diffs in the
chained initial `[WIP]` CL 2):

1. **Change Design Brief (`docs/design/changes/<change>.md` — CL 1)**:
   - Target length: **80–150 lines** (hard ceiling **250 lines** even for
     multi-CL designs). Because Initial CL 2 carries `<component>.md` and any
     `.h`/`.td` interface diffs, `changes/<change>.md` must **not** duplicate
     them or copy-paste `.h`/`.td` code blocks.
   - Structure `changes/<change>.md` into four flat sections (per
     [`TEMPLATE.md`](../../../docs/design/changes/TEMPLATE.md)):
     - **Section 1: Summary & Motivation**: State the problem and target outcome
       first (without putting the implementation mechanism inside the problem
       statement, per `go/writing-a-design-doc`), followed by verifiable
       **Goals** and plausible **Non-Goals** paired with 1-line reasons.
     - **Section 2: Proposed Change & Component Doc Delta**: Summarize the
       design delta, key invariants, and fallback behavior, linking to the
       updated sections of `<component>.md` and any `.h`/`.td` interface files
       in CL 2 (or explaining the change inline if no `<component>.md` applies).
       Place **Alternatives Considered** *after* the proposed change (per
       `go/tott/773`), stating why the chosen option wins, what downside it
       accepts, and why realistic alternatives were rejected.
     - **Section 3: Implementation & Verification Plan**: A single unified
       checklist (`egm-draft` Step 5.1) listing implementation steps (with
       affected files inline), positive and negative/fallback test cases, and
       the final build/doc/archive sync step. Do **not** create separate
       affected-file and task-list sections that repeat the same files.
     - **Section 4: Follow-Up Items**: Deferred work and adjacent issues for
       reviewer approval and Buganizer (`564546727`) filing.
2. **Living Component Doc & Boundary Interfaces (`docs/design/<component>.md` +
   `.h`/`.td` — Chained Initial `[WIP]` CL 2 / `CL 2a..2n`)**:
   - **`docs/design/<component>.md`** (and `docs/design/INDEX.md` if updated):
     Holds the timeless architectural description at the target state (summary,
     pass pipeline table/diagram, design rules and invariants, concrete IR shape
     / before-after examples, test files, and current `TODO(...)` limits).
     Integrate new or changed behavior in-place (`"One Contract, One Place"`)
     without temporal phrasing (*"This CL adds..."*, *"Previously..."*). When
     the implementation is staged into `CL 2a .. CL 2n`, keep each stage's
     `<component>.md` additions in separate paragraphs, bullets, or table rows
     so the per-stage `[WIP]` commits rebase cleanly without conflicts (relying
     on `/implement-design archive` in `CL 2n` to merge them into cohesive
     text).
   - **Boundary Interface Files (`.h` Headers & `.td` TableGen Definitions)**:
     When the design adds, modifies, or deletes cross-file C/C++ declarations
     (`.h`) or MLIR passes/ops/attributes (`.td`), update those interface files
     directly (with complete doc comments per
     [`../../../AGENTS.md`](../../../AGENTS.md)) for Initial CL 2 (`CL 2a..2n`).
     Never touch `.cpp`/`.cc`/`.c` implementation files during `/design-doc`,
     and never move pass-internal `.cpp` helpers into `.h` headers just to
     expose them in CL 2.

---

## 3. Technical Altitude & Prose Craft (`go/speak-like-tl`)

1. **Altitude Control: Principles & Examples by Default, Compact Tables Opt-In**:
   - **Never Transcribe C++ Control Flow into Prose**: Do not serialize C++
     boolean expressions, variable assignments, or helper-by-helper call trees
     into English sentences (e.g., never write *"When `!outputChannel.empty() ||
     isChannelLast`, set `vectorParallelSizes[secondPar] = ...`"*). If a reader
     has to parse C++ syntax inside prose, the doc is at the wrong altitude
     (`go/egm:standard:zoom-out`, `go/good-design-docs`).
   - **Default (Prose Principles, Invariants & Examples)**: State *what* rule or
     invariant holds, *why* it is needed, how unsupported cases fall back, and
     illustrate with 1–2 concrete examples (such as an IR snippet or tile-size
     trace in `<component>.md`).
   - **Opt-In (Compact Tables or Short Pseudocode)**: When the developer
     requests it or when a design hinges on a multi-category heuristic matrix
     (such as operation-category tiling rules), use a compact Markdown table
     (`< 15` words per cell) or a <= 10-line pseudocode block instead of
     multi-bullet prose paragraphs.
2. **Short Sentences, Concrete Examples First, Minimal Backticks**:
   - **Sentence Length**: Keep sentences short and direct (aim for **15–25
     words**; split any sentence exceeding ~30–35 words).
   - **Concrete Before Abstract**: Lead with the concrete concept or example
     (*"Support multi-op reductions such as decomposed `linalg.softmax` (`max`
     -> `exp` + `sum` -> `div`)"*) before stating structural conditions.
   - **Natural Language First (No Symbol Dumping)**: Explain flows by logical
     component role (*"the DTCM pass"*, *"the promotion pass"*, *"a contraction
     with reuse"*) calibrated to a newly hired SWE outside the team. Include at
     most 1–2 parenthetical symbol/file pointers per paragraph to anchor key
     entrypoints, using symbol names rather than fragile line numbers.
3. **Say It Once (Anti-Repetition & 50% Compression Check)**:
   - State each requirement, invariant, or file change **once** across the
     document (`go/speak-like-tl`). Never repeat table contents in prose below
     the table, and never restate the Motivation section inside the Design or
     Implementation sections.
4. **Grounded Claims, Clear Actors & Natural Human Voice**:
   - Back performance, memory, or code-size claims with concrete numbers or
     bounds (cycle counts, KB budget, register counts) rather than vague
     adjectives (*"fast"*, *"low overhead"*, *"scalable"*).
   - Name the specific pass, component, or actor when ambiguity is possible, and
     use active verbs over nominalizations (*"evaluate"* instead of *"perform an
     evaluation of"*). Natural first-person plural (*"we"*) and plain everyday
     words (*"simple"*, *"small"*, *"thin tile"*) are encouraged when they make
     sentences shorter and clearer.
   - Avoid AI filler vocabulary (*leverage*, *utilize*, *streamline*,
     *orchestrate*, *facilitate*, *delve*, *holistic*, *seamless*, *robust*,
     *overarching*, *pivotal*, *paradigm*).

---

## 4. Design Doc Audit Rubric & Fact-Checking Protocol (`audit`)

### 4.1 Audit Rubric for the Isolated Narrow-View Subagent (`/design-doc audit`)

The isolated zero-history subagent reads `docs/design/changes/<change>.md`, any
modified `docs/design/<component>.md` (and `INDEX.md`), any modified `.h` or
`.td` interface files, and the repository files they reference, evaluating three
dimensions:

1. **Blind Comprehension & Readability Check (Human-First Gate)**:
   - **1-Paragraph Blind Summary**: In 1 short paragraph, explain in your own
     words how the current system works and what the proposed change does, based
     *strictly* on `<change>.md` and `<component>.md`.
   - **Readability, Altitude & Length Violations**: Flag any place where:
     - C++ conditionals, formulas, or ASTs are transcribed into prose (violating
       the prose-principles altitude rule);
     - The same point or invariant is repeated across multiple sections;
     - Sentences exceed ~35 words or dump walls of backticked symbols;
     - `changes/<change>.md` duplicates architectural prose/examples that belong
       in `<component>.md`, or exceeds the 80–150 line target (250-line hard
       ceiling);
     - Unstated context or undefined domain terms forced you to guess.
2. **Zoom-Out, Technical Accuracy & Trade-Off Audit**:
   - **Referenced Symbols & Files**: Do all cited file paths, symbols,
     interfaces, and build targets exist and match the codebase?
   - **Zoom-Out (Root Cause, YAGNI & Reuse)**: Does the design solve the root
     problem without speculative generality (YAGNI), and could any proposed
     custom logic reuse an existing codebase or upstream utility? (When auditing
     `draft-from-commits`, could the design or the underlying commits be
     simplified or generalized?)
   - **Trade-Offs, Invariants & Fallbacks**: Are realistic alternatives compared
     after the proposal, and are key invariants, edge cases, and graceful
     fallbacks (compliant with [`../../../AGENTS.md`](../../../AGENTS.md))
     clear?
3. **Scope & Implementation Plan Check**:
   - Does **Section 3** provide a single unified checklist of steps (with target
     files inline) and positive + negative/fallback verification tests?
   - If the change is too large for a single CL, recommend either tightening the
     spec (moving deferred slices to **Section 4**) or grouping **Section 3**
     into staged CLs (`CL 2a`, `CL 2b`, ...).

### 4.2 Main-Agent Fact-Checking Protocol (No Automatic Edits)

When the isolated audit subagent finishes:
1. **Do not modify any files automatically.**
2. **Fact-check every finding against the codebase**: Label each finding (`#1`,
   `#2`, ...) as **[Verified]** or **[Refuted by `<file>` (`<symbol>`)]** (when
   the subagent missed an existing guarantee; for refuted points, suggest
   whether to clarify the doc or record the rebuttal).
3. **Present the 1-paragraph Blind Summary prominently first** so the developer
   can verify that a zero-context reader's understanding matches their intent,
   followed by the verified findings list, and wait for direction.
