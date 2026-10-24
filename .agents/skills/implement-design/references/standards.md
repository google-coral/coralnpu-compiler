# Design Implementation, Drift Audit & Living-Doc Standards

This document defines the general engineering, Spec-Driven Development (SDD),
drift-audit, and documentation-archival standards used by both the
implementation session (the main agent during `code`, `archive`, and
`file-followups`) and the isolated narrow-view audit subagent (with zero chat
history during `audit`). Project-specific rules live in
[`../../../AGENTS.md`](../../../AGENTS.md), and prose/citation rules live in
[`../../design-doc/references/standards.md`](../../design-doc/references/standards.md).

---

## 1. Spec-Driven Implementation & Adaptation Standards (`code`)

1. **Incremental Task Execution & Contract-Derived Verification**:
   - Execute tasks in order from **Section 2.2 (Implementation Tasks)** of
     `<change>.md` (or the current stage `CL 2a`, `CL 2b`, ... if staged).
   - Derive **Positive Tests** from every **Offered Postcondition** and
     **Negative / Fallback Tests** from every **Imposed Precondition** and
     error/fallback path in **Section 1.4 / 1.5**.
   - Run each task's verification command and mark the task `- [x]` only after
     compilation, tests, and any spec updates succeed.
2. **Bounded Autonomous Self-Repair (3-Strike Rule)**:
   - When a build or test fails, attempt up to **3 autonomous fix-and-retest
     cycles**.
   - If the failure persists after 3 attempts, **stop immediately**, leave the
     task unchecked (`- [ ]`), and report the failure root-cause hypothesis to
     the developer rather than thrashing or weakening test assertions.
3. **Continuous Spec Synchronicity ("Fixpoint" & Zero-Context Resumability)**:
   - Keep `<change>.md` at a "fixpoint" (zero divergence between spec and code)
     at the end of every turn so a fresh zero-history session can resume
     seamlessly:
     - **Local / Tactical Adaptations** (adjusting helper signatures, touching
       an additional caller or build target, or handling a local boundary case):
       Implement the improvement, update **Part 1 / Part 2** and **Section 2.3
       (Implementation Notes & Adaptations)** of `<change>.md` in the same turn
       (following
       [`../../design-doc/references/standards.md`](../../design-doc/references/standards.md)),
       and highlight the adaptation to the developer.
     - **Architectural Pivots** (changing module or pipeline boundaries,
       replacing the core algorithm or data model, adding a third-party patch,
       or altering user-facing flags/APIs): **Spec modifications that alter
       architectural contracts require human judgment.** Pause code edits,
       explain the newly discovered trade-offs to the developer, and update
       `<change>.md` and the code only after developer sign-off.
4. **Strict Scope Discipline (No Unanchored Scope Creep)**:
   - Implement the minimum code required to satisfy the specification and tests
     (YAGNI).
   - Do not bundle unrelated refactorings or adjacent feature work into the CL;
     record any newly discovered out-of-scope issues under **Part 3: Follow-Up
     Items** in `<change>.md`.

---

## 2. Implementation & Spec-to-Code Drift Audit Rubric (`audit`)

### 2.1 Audit Rubric for the Isolated Narrow-View Subagent (`/implement-design audit`)

The isolated zero-history subagent evaluates the implementation diff against
`<change>.md` and [`../../../AGENTS.md`](../../../AGENTS.md) across three
dimensions:

1. **Blind Implementation Summary**:
   - In 1 short paragraph, explain in your own words what the implementation
     diff actually does and how it works, based *strictly* on the diff and
     `<change>.md`.
2. **4-Way Bidirectional Spec-to-Code Drift Analysis**:
   - **Specified but Missing / Shortcut**: Requirements, boundary handlers,
     error/fallback paths, or contracts in `<change>.md` that are absent,
     stubbed, or hardcoded in the diff, or implemented tasks in **Section 2.2**
     still left unchecked (`- [ ]`).
   - **Implemented but Diverges from Spec**: Code that contradicts `<change>.md`
     (altered signatures, data structures, state transitions, or violated
     preconditions/postconditions) or adaptations missing from **Section 2.3**.
   - **Implemented but Unspecified (Scope Creep vs. Refactor Invariance)**: New
     observable component behavior, public interfaces, configuration/CLI flags,
     or architectural dependencies introduced in the diff that `<change>.md`
     does not cover. Apply the **Refactor Invariance Litmus Test**: *"If a
     behavior-preserving internal refactor that no caller can observe would
     change this detail, it is an internal implementation detail and does not
     require spec transcription."*
   - **Symbol & Terminology Drift**: File paths, symbol/type/enum names, or
     domain abbreviations in `<change>.md` that do not match the actual
     identifiers in the codebase at `HEAD`.
3. **Contract-Derived Test Coverage, Correctness & Project Rules**:
   - **Contract Test Coverage**: Verify that every **Offered Postcondition** has
     a positive test and every **Imposed Precondition / Fallback** in **Section
     1.4 / 1.5** has a negative or fallback test in the diff.
   - **Correctness & Safety**: Flag logic flaws, unhandled boundary conditions,
     resource/hardware hazards, or missing fallback diagnostics.
   - **Project Rule Compliance**: Verify strict compliance with all
     anti-test-hacking, build, code-style, and comment-density rules in
     [`../../../AGENTS.md`](../../../AGENTS.md).

### 2.2 Main-Agent Fact-Checking Protocol (No Automatic Edits)

When the isolated audit subagent finishes:
1. **Do not modify any code or documentation files automatically.**
2. **Fact-check every finding against the codebase**: Label each finding (`#1`,
   `#2`, ...) as **[Verified]** or **[Refuted by `<file>` (`<symbol>`)]**.
3. **Present the 1-paragraph Blind Implementation Summary prominently first** so
   the developer can verify that the implemented behavior matches their intent,
   followed by the verified findings list, and wait for direction.

---

## 3. Living Component Doc Promotion Standards (`archive`)

When folding a completed `<change>.md` into `./docs/design/<component>.md`:

1. **Durable Architecture Only (Timeless `HEAD` Framing)**:
   - Living component docs record durable architecture, boundaries, contracts,
     and invariants at `HEAD`.
   - Never use temporal or change-relative phrasing (*"Recently added..."*,
     *"This CL introduces..."*, *"Previously, X did Y"*).
2. **Extend In-Place ("One Contract, One Place" / Anti-Append Rule)**:
   - Integrate new or updated capabilities directly into the existing sections,
     diagrams, and tables of `<component>.md` rather than appending a
     disconnected changelog-style section at the end.
3. **Preserve "Why Not X" Guardrails, Strip Transient Scaffolding**:
   - Fold non-obvious rejected-alternative rationales and design constraints
     from **Section 1.3 (Alternatives Considered)** and **Section 2.3
     (Adaptations)** into `<component>.md` so future engineers and agents know
     why simpler-looking alternatives were rejected (Chesterton's fence).
   - Strip all transient task checklists (`- [x]`), bug numbers, and CL-specific
     notes (which remain preserved in `docs/design/archive/<change>.md`).
4. **Prose & Citation Craft**:
   - Follow **Section 3 (Prose, Terminology & Citation Standards)** of
     [`../../design-doc/references/standards.md`](../../design-doc/references/standards.md)
     (natural-language-first explanations without code-symbol dumping, symbol
     references over fragile line numbers, and falsifiable language).
