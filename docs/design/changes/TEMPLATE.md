# <Title of Change>

- **Author(s)**: <name>
- **Component Doc(s)**: [<component>.md](../<component>.md)
- **Bug(s)**: b/<id>

---

## Part 1: Design Specification

### 1.1 Objective & Scope
- **Problem Statement**: Describe the problem and why this change is needed.
- **Goals**: List the concrete outcomes this change will achieve (scoped to a
  focused, reviewable increment).
- **Non-Goals**: List related work, deferred phases, or open Buganizer issues
  (`b/<id>`) that are explicitly out of scope for this change (and tracked in
  **Part 3: Follow-Up Items** if they should be filed as follow-up tickets).

### 1.2 Background & Invariants
Summarize the relevant compiler or runtime context (link to the living
component doc in `docs/design/<component>.md` instead of duplicating it). List
the hardware and IR invariants that must hold (for example: DTCM/ITCM capacity,
32 vector registers, zero register spills, or upstream IREE/LLVM hook
contracts).

### 1.3 Alternatives Considered
Compare at least two viable design options, including their trade-offs and a
rough estimate of lines of code added or removed (`+X / -Y LOC`).

| Option                    | Pros                 | Cons                 | Est. LOC |
| :------------------------ | :------------------- | :------------------- | :------- |
| **Option 1 (Chosen)**: .. | - ...                | - ...                | +.. / -..|
| **Option 2**: ...         | - ...                | - ...                | +.. / -..|

**Decision Rationale**: Explain why Option 1 was selected over the
alternatives.

### 1.4 Detailed Technical Design
Describe the technical design with minimal ambiguity:
- **Pipeline / Control Flow**: Include a Mermaid diagram or step-by-step data
  flow when helpful.
- **Interfaces & Signatures**: Show C++ class/function declarations, TableGen
  (`.td`) definitions, or CLI flags. Do not paste long C++ function bodies.
- **IR Examples**: For compiler passes or lowerings, show representative input
  MLIR IR and expected output MLIR IR.
- **Edge Cases & Fallbacks**: Specify how unsupported shapes, types, or
  configurations fall back gracefully.

### 1.5 Verification Plan
- **Unit / LIT Tests**: Target `.mlir` or C++ test files to add or update.
- **Model Check Tests**: Scoped `//tests/models/linalg/...` or
  `//tests/models/stablehlo/...` targets.
- **Register & Spill Verification**: If modifying tiling or codegen, specify
  the register allocation report checks (`Vec Spills: 0, Vec Reloads: 0`).

---

## Part 2: Implementation Plan

### 2.1 Expected Affected Files
- `path/to/file.h` — declare ...
- `path/to/file.cpp` — implement ...
- `tests/path/to/test.mlir` — add LIT verification tests for ...

### 2.2 Implementation Tasks (`code` step)
Leave tasks unchecked (`- [ ]`) during the design phase (CL 1). Check off each
task (`- [x]`) during implementation only after its code is written and
verified. If the implementation is large enough to split across multiple code
CLs (`CL 2a`, `CL 2b`, ...), group the tasks by CL so each CL compiles and
passes tests independently.

- [ ] **Task 1**: <Description of atomic implementation step>
  - *Target*: `path/to/file.cpp`
  - *Verify*: `bazel test --config=dev //tests/...`
- [ ] **Task 2**: <Description of atomic implementation step>
  - *Target*: `path/to/file.cpp`
  - *Verify*: `bazel test --config=dev //tests/...`
- [ ] **Task N (Build & Doc Sync)**: Update `CMakeLists.txt` (if build targets
  or source files changed), `refresh_compile_commands` in `BUILD.bazel` (if new
  top-level C/C++ targets were added), and `README.md` (if user-facing
  `--coralnpu-*` flags or runtime options changed).

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
