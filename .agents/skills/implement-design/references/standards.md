# Design Implementation, Drift Audit & Living-Doc Standards

This document defines the Spec-Driven Development (SDD), drift-audit, and
archival standards used by both the main agent (`code`, `archive`,
`file-followups`) and the isolated narrow-view audit subagent (`audit`).
Project-specific rules live in [`../../../AGENTS.md`](../../../AGENTS.md), and
prose/citation rules live in
[`../../design-doc/references/standards.md`](../../design-doc/references/standards.md).

---

## 1. Spec-Driven Implementation & Adaptation Standards (`code`)

1. **Incremental Task Execution & Verification**:
   - Execute tasks in order from **Section 3 (Implementation & Verification
     Plan)** of `<change>.md`, using both `<change>.md` and the updated
     `<component>.md` (already in the initial CL 2 commit) as the specification.
   - Implement the **Positive** and **Negative / Fallback** test cases promised
     in **Section 3** (and `<component>.md`), and mark each task `- [x]` only
     after compilation, tests, and any doc updates succeed.
2. **Bounded Autonomous Self-Repair (3-Strike Rule)**:
   - When a build or test fails, attempt up to **3 autonomous fix-and-retest
     cycles**. If the failure persists after 3 attempts, **stop immediately**,
     leave the task `- [ ]`, and escalate to the developer using
     [`../../design-doc/references/interview.md`](../../design-doc/references/interview.md)
     (presenting a self-contained root-cause primer and numbered recovery
     options rather than thrashing or weakening test assertions).
3. **Continuous Doc Synchronicity (At the Right Altitude)**:
   - Keep `<change>.md` and `<component>.md` synchronized with the code at the
     end of every turn without dumping low-level C++ mechanics into prose:
     - **Internal Implementation Details (Refactor Invariance)**: Helper
       function names, local data structures, or branch ordering that do not
       alter the architecture, invariants, or observable behavior belong in C++
       code only; do **not** transcribe them into `<change>.md`.
     - **Tactical Adaptations** (touching an additional file/build target,
       adjusting a public/cross-pass signature in `<change>.md`, or refining an
       edge-case rule): Update `<change>.md` (including **Implementation Notes &
       Adaptations** in Section 3) and `<component>.md` in the same turn, and
       highlight the adaptation to the developer.
     - **Architectural Pivots** (changing module or pipeline boundaries,
       replacing the core algorithm, adding a third-party patch, or altering
       user-facing flags/APIs): Pause code edits, explore alternatives for the
       newly discovered constraint using the relevant Design Lenses in
       [`../../design-doc/references/lenses.md`](../../design-doc/references/lenses.md)
       (e.g., `Minimalist`, `Precedent`, `Shifter`, `Modularity`), and align
       with the developer via
       [`../../design-doc/references/interview.md`](../../design-doc/references/interview.md)
       before updating `<change>.md`, `<component>.md`, and the code.
4. **Strict Scope Discipline (No Unanchored Scope Creep)**:
   - Implement the minimum code required to satisfy the specification and tests
     (YAGNI). Do not bundle unrelated refactorings or adjacent feature work;
     record newly discovered out-of-scope issues under **Section 4: Follow-Up
     Items** in `<change>.md`.

---

## 2. Implementation & Semantic Drift Audit Rubric (`audit`)

### 2.1 Audit Rubric for the Isolated Narrow-View Subagent (`/implement-design audit`)

The isolated zero-history subagent evaluates the implementation diff against
`<change>.md`, `<component>.md`, and [`../../../AGENTS.md`](../../../AGENTS.md)
across four dimensions:

1. **Blind Implementation Summary**:
   - In 1 short paragraph, explain in your own words what the implementation
     diff actually does and how it works, based *strictly* on the diff,
     `<change>.md`, and `<component>.md`.
2. **Semantic, Invariant & Doc-Sync Drift Analysis (No C++-in-Prose
   Enforcement)**:
   - **Specified Goals/Invariants Missing or Broken**: Goals, invariants,
     pipeline steps, or fallback behaviors in `<change>.md` and `<component>.md`
     that are absent, stubbed, or contradicted by the diff, or completed tasks
     in **Section 3** still left unchecked (`- [ ]`).
   - **Observable Scope Creep**: New user-facing flags, public interfaces,
     passes, or architectural dependencies in the diff that `<change>.md` and
     `<component>.md` do not cover. *(Apply the **Refactor Invariance Rule**:
     never flag internal helper functions, local variables, or C++ branch
     expressions for not being transcribed into prose.)*
   - **Symbol & File Accuracy**: Verify that every file path, pass name, or
     symbol explicitly cited in `<change>.md` and `<component>.md` matches the
     actual identifiers in the codebase at `HEAD`.
3. **Code-Level Simplification & Reuse Lenses (`[Lens: ...]`)**:
   - Challenge the C++ implementation through four code-level lenses (tagging
     each simplification finding with `[Lens: ...]`):
     - **`[Lens: Minimalist]` (Code Compression & Deletion)**: Can newly added
       helpers, branches, or now-obsolete legacy code be deleted or collapsed to
       reduce LOC while keeping control flow flat?
     - **`[Lens: Precedent]` (Upstream MLIR/IREE & Codebase Utility Reuse)**:
       Could custom loops, shape/indexing math, or op checks be replaced by
       existing MLIR/IREE utilities (`StaticValueUtils.h`, `IndexingUtils.h`,
       `TilingInterface`, `linalg::inferContractionDims`, `CPUUtils.h`) or
       established patterns in sibling passes/targets?
     - **`[Lens: Modularity]` (Single-Responsibility Functions & Boundaries)**:
       Are functions, classes, and headers short, single-purpose, and free of
       mixed responsibilities per [`../../../AGENTS.md`](../../../AGENTS.md)?
     - **`[Lens: Generalizer]` (Structural Checks Over Special Cases)**: Can
       ad-hoc op, shape, or dtype special cases in the C++ code be replaced by a
       single cleaner structural invariant?
4. **Test Coverage, Correctness & Project Rules**:
   - **Test Coverage**: Verify positive tests for the new capability and
     negative/fallback tests for unsupported inputs or violated preconditions
     promised in **Section 3**.
   - **Correctness, Safety & Project Rules**: Flag logic flaws, boundary bugs,
     hardware hazards, missing fallback diagnostics, or violations of the
     anti-test-hacking, build, style, and comment rules in
     [`../../../AGENTS.md`](../../../AGENTS.md).

### 2.2 Main-Agent Fact-Checking Protocol (No Automatic Edits)

Follow **Section 4.2** of
[`../../design-doc/references/standards.md`](../../design-doc/references/standards.md):
do not modify code or docs automatically; present the 1-paragraph Blind
Implementation Summary first, then format each verified finding (`#1`, `#2`,
...) with explicit **`Issue`**, **`Proposed Change`**, and **`Recommendation`**
(`Apply`, `Defer to Section 4`, `Skip (Keep Current)`, or `Decision Needed` via
[`../../design-doc/references/interview.md`](../../design-doc/references/interview.md))
fields while grouping any `[Refuted by <file> (<symbol>)]` claims into a compact
footer.

---

## 3. Living Component Doc Verification & Archival Standards (`archive`)

During `/implement-design archive` (in `CL 2` or the final stage `CL 2n`):

- **Coherence Pass & Adaptation Sync**: Merge and smooth any separate per-stage
  paragraphs/bullets added across `CL 2a .. CL 2n` and any adaptations or
  non-obvious "Why Not X" guardrails discovered during `code` into unified,
  cohesive `HEAD` prose in `./docs/design/<component>.md` (and
  `./docs/design/INDEX.md`).
- **Timeless `HEAD` Prose**: Verify that `<component>.md` contains no temporal
  phrasing (*"Recently added..."*, *"This CL introduces..."*), transient task
  checklists, or C++-in-prose dumps per
  [`../../design-doc/references/standards.md`](../../design-doc/references/standards.md).
