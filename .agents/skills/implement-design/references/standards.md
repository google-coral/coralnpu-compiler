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

1. **Incremental Task Execution & Verification**:
   - Execute tasks in order from **Section 3 (Implementation & Verification
     Plan)** of `<change>.md` (or the current stage `CL 2a`, `CL 2b`, ... if
     staged), using both `<change>.md` and the updated `<component>.md` (already
     present in the initial CL 2 commit) as the architectural specification.
   - Implement the **Positive** and **Negative / Fallback** test cases promised
     in **Section 3** (and `<component>.md`), and mark each task `- [x]` only
     after compilation, tests, and any doc updates succeed.
2. **Bounded Autonomous Self-Repair (3-Strike Rule)**:
   - When a build or test fails, attempt up to **3 autonomous fix-and-retest
     cycles**.
   - If the failure persists after 3 attempts, **stop immediately**, leave the
     task unchecked (`- [ ]`), and report the failure root-cause hypothesis to
     the developer rather than thrashing or weakening test assertions.
3. **Continuous Doc Synchronicity (At the Right Altitude)**:
   - Keep `<change>.md` and `<component>.md` synchronized with the code at the
     end of every turn without dumping low-level C++ mechanics into prose:
     - **Internal Implementation Details (Refactor Invariance)**: Helper
       function names, local data structures, or branch ordering that do not
       alter the architecture, invariants, or observable behavior described in
       `<component>.md` and `<change>.md` belong in the C++ code only; do **not**
       transcribe them into `<change>.md`.
     - **Tactical Adaptations** (touching an additional file/build target,
       adjusting a public/cross-pass signature in `<change>.md`, or refining an
       edge-case rule): Update `<change>.md` (including **Implementation Notes &
       Adaptations** in Section 3) and `<component>.md` in the same turn, and
       highlight the adaptation to the developer.
     - **Architectural Pivots** (changing module or pipeline boundaries,
       replacing the core algorithm, adding a third-party patch, or altering
       user-facing flags/APIs): Pause code edits, explain the newly discovered
       trade-offs to the developer, and update `<change>.md`, `<component>.md`,
       and the code only after developer sign-off.
4. **Strict Scope Discipline (No Unanchored Scope Creep)**:
   - Implement the minimum code required to satisfy the specification and tests
     (YAGNI).
   - Do not bundle unrelated refactorings or adjacent feature work into the CL;
     record any newly discovered out-of-scope issues under **Section 4:
     Follow-Up Items** in `<change>.md`.

---

## 2. Implementation & Semantic Drift Audit Rubric (`audit`)

### 2.1 Audit Rubric for the Isolated Narrow-View Subagent (`/implement-design audit`)

The isolated zero-history subagent evaluates the implementation diff against
`<change>.md`, `<component>.md`, and [`../../../AGENTS.md`](../../../AGENTS.md)
across three dimensions:

1. **Blind Implementation Summary**:
   - In 1 short paragraph, explain in your own words what the implementation
     diff actually does and how it works, based *strictly* on the diff,
     `<change>.md`, and `<component>.md`.
2. **Semantic, Invariant & Doc-Sync Drift Analysis (No C++-in-Prose Enforcement)**:
   - **Specified Goals/Invariants Missing or Broken**: Goals, invariants,
     pipeline steps, or fallback behaviors in `<change>.md` and `<component>.md`
     that are absent, stubbed, or contradicted by the diff, or completed tasks
     in **Section 3** still left unchecked (`- [ ]`).
   - **Observable Scope Creep**: New user-facing flags, public interfaces,
     passes, or architectural dependencies introduced in the diff that
     `<change>.md` and `<component>.md` do not cover. *(Apply the **Refactor
     Invariance Rule**: never flag internal helper functions, local variables,
     or C++ branch expressions for not being transcribed into prose in
     `<change>.md` or `<component>.md`.)*
   - **Symbol & File Accuracy**: Verify that every file path, pass name, or
     symbol explicitly cited in `<change>.md` and `<component>.md` matches the
     actual identifiers in the codebase at `HEAD`.
3. **Test Coverage, Correctness & Project Rules**:
   - **Test Coverage**: Verify that the diff includes positive tests for the new
     capability and negative/fallback tests for unsupported inputs or violated
     preconditions promised in **Section 3**.
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

## 3. Living Component Doc Verification & Archival Standards (`archive`)

Because `<component>.md` was already updated in the initial CL 2 (`CL 2a..2n`)
commit(s) during the design phase, `/implement-design archive` (run in `CL 2` or
the final stage `CL 2n`) performs a coherence pass and finalizes the docs:

1. **Coherence Pass, Timeless `HEAD` Framing & Adaptation Sync in `<component>.md`**:
   - When a change was split across `CL 2a .. CL 2n` with separate per-stage
     paragraphs or bullets in `<component>.md`, merge and smooth those separate
     additions into unified, cohesive prose.
   - Ensure any adaptations made during `code` (and any non-obvious "Why Not X"
     guardrails that future engineers need to know) are reflected in-place in
     `./docs/design/<component>.md` (and `./docs/design/INDEX.md`).
   - Verify that `<component>.md` contains no temporal phrasing (*"Recently
     added..."*, *"This CL introduces..."*), transient task checklists, or
     C++-in-prose dumps, following
     [`../../design-doc/references/standards.md`](../../design-doc/references/standards.md).
2. **Archive `<change>.md` via `git mv`**:
   - Move `./docs/design/changes/<change>.md` to
     `./docs/design/archive/<change>.md` using `git mv` and check off the
     final sync item in **Section 3**.
