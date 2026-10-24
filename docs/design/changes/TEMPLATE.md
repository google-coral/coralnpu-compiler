# <Title of Change>

- **Author(s)**: <name>
- **Component Doc(s)**: [<component>.md](../<component>.md)
- **Bug(s)**: b/<id>

---

## Part 1: Design Specification

### 1.1 Objective & Scope
- **Problem Statement**: Describe the root problem, current limitation, and why
  this change is needed. State *only* the problem space and target outcome — do
  not put the implementation mechanism here.
- **Goals**: List the concrete, verifiable outcomes this change will achieve
  (scoped to a focused, reviewable increment).
- **Non-Goals**: List plausible scope assumptions, deferred phases, or open
  Buganizer issues (`b/<id>`) that this change **deliberately refuses to solve**
  in this iteration, each paired with a 1-line reason (and tracked in **Part 3:
  Follow-Up Items** if they should be filed as follow-up tickets).

### 1.2 Background & Invariants
Summarize how the relevant subsystem works today in plain natural language first
(referring to components by their logical role and linking to the living
component doc in `docs/design/<component>.md` instead of duplicating it). List
the system, hardware, or interface invariants that must be preserved.

### 1.3 Alternatives Considered
Compare at least two viable design options that a competent engineer could
reasonably advocate for (avoid strawmen), including their trade-offs and a rough
estimate of lines of code added or removed (`+X / -Y LOC`).

| Option                    | Pros                 | Cons                 | Est. LOC |
| :------------------------ | :------------------- | :------------------- | :------- |
| **Option 1 (Chosen)**: .. | - ...                | - ...                | +.. / -..|
| **Option 2**: ...         | - ...                | - ...                | +.. / -..|

- **Decision Rationale**: Justify the choice using the 3-part formula:
  *"Quality/Constraint X is a priority over Y, so we chose Option 1, accepting
  downside Z."*
- **Reconsideration Trigger**: For each rejected alternative, state why it
  failed against the top priorities and what future condition would trigger
  reconsidering it.

### 1.4 Detailed Technical Design
Walk through the default steady-state flow in cohesive prose first before
covering boundary conditions and fallbacks:
- **Architecture / Control Flow**: Include a Mermaid diagram or step-by-step
  data flow when helpful.
- **Contracts & Boundaries**:
  - *Imposed Preconditions (Caller / Input Obligations)*: Input invariants,
    state, or constraints assumed on entry.
  - *Offered Postconditions (Callee / Output Guarantees)*: Output invariants or
    state transitions guaranteed on exit.
- **Interfaces & Signatures**: Show C++/C/Python declarations, TableGen (`.td`)
  definitions, schemas, or CLI flags. Do not paste verbatim function bodies.
- **Representative Examples**: Show concrete before/after examples (such as
  input/output IR, API calls, or state transitions).
- **Edge Cases & Fallbacks**: Specify how violated preconditions, unsupported
  inputs, or resource limits are handled or fall back gracefully.

### 1.5 Verification Plan
- **Positive Tests**: Unit, LIT, or integration test files verifying the new
  capability and `WHEN`/`THEN` requirements.
- **Negative / Fallback Tests**: Tests verifying that violated preconditions or
  unsupported inputs are rejected or fall back gracefully without regressions.
- **End-to-End / Model / Hardware Checks**: Scoped model check targets or
  resource/performance assertions where applicable.

---

## Part 2: Implementation Plan

### 2.1 Expected Affected Files
Group files by component (`Compiler`, `Runtime`, `CRT`, `PJRT Plugin`, `Tools`,
`Build`, `Patches`, `Tests`), sorted lexicographically within each component,
with an empty line between components and a 6-space indented sub-bullet for the
description:

- **<Component 1>**:
  - `path/to/file_a.cpp`
      - Implement ...
  - `path/to/file_b.h` *(new)*
      - Declare ...

- **Tests**:
  - `tests/path/to/test`
      - Add positive and negative/fallback verification tests for ...

### 2.2 Implementation Tasks (`code` step)
Leave tasks unchecked (`- [ ]`) during the design phase (CL 1). Check off each
task (`- [x]`) during implementation only after its code is written and
verified. If the implementation is large enough to split across multiple code
CLs (`CL 2a`, `CL 2b`, ...), group the tasks by CL so each CL compiles and
passes tests independently.

- [ ] **Task 1**: <Description of atomic implementation step>
  - *Target*: `path/to/file.cpp` (`SymbolName`)
  - *Verify*: `bazel test --config=dev //tests/...`
- [ ] **Task 2**: <Description of atomic implementation step>
  - *Target*: `path/to/file.cpp` (`SymbolName`)
  - *Verify*: `bazel test --config=dev //tests/...`
- [ ] **Task N (Build & Doc Sync)**: Sync build definitions (`BUILD.bazel`,
  `CMakeLists.txt`, `refresh_compile_commands`) and user-facing documentation
  (`README.md`) as required by `.agents/AGENTS.md`.

### 2.3 Implementation Notes & Adaptations
Record any adaptations made during the `code` step where the implementation
diverged from the initial plan or adopted a better solution discovered in the
codebase.

- *(None yet)*

### 2.4 Documentation Archive Checklist (`archive` step — Pre-Submit)
Run this step in the final implementation CL before uploading for review:

- [ ] Fold lasting architectural changes (including non-obvious "Why Not X"
      rejected-alternative rationales) into `docs/design/<component>.md`, and
      update `docs/design/INDEX.md` if a component doc was added or its scope
      changed.
- [ ] Move this document from `docs/design/changes/<change>.md` to
      `docs/design/archive/<change>.md` using `git mv`.
- [ ] Format modified code with `./scripts/format-code.sh`.

---

## Part 3: Follow-Up Items (For Reviewer Approval -> Buganizer `564546727`)

Record work deferred during design (CL 1) or adjacent issues discovered during
implementation (CL 2). Reviewers inspect and approve these items in Gerrit
alongside the CL. Once the implementation CL receives `+2`, run
`/implement-design file-followups` to file them in Buganizer under component
`564546727`.

- [ ] **<Short Follow-Up Title>** (`P2` | `path/to/file.cpp`):
      1–2 sentence description of the follow-up problem and proposed direction.
