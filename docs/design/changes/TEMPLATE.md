# <Title of Change>

- **Author(s)**: <name>
- **Component Doc(s)**: [<component>.md](../<component>.md) *(updated in chained CL 2)*
- **Bug(s)**: b/<id>

---

## 1. Summary & Motivation

State the problem and what this change does in 1–2 short paragraphs. Lead with
concrete examples or measured impact (cycle counts, memory size, failing models)
rather than implementation mechanics.

- **Goals**:
  - Concrete, verifiable outcomes this change achieves (scoped to a focused,
    reviewable increment).
- **Non-Goals**:
  - Plausible scope extensions or open Buganizer issues (`b/<id>`) that this
    change deliberately leaves out, each with a 1-line reason (and listed in
    **Section 4: Follow-Up Items** if they should be filed as follow-up
    tickets).

---

## 2. Proposed Change & Component Doc Delta

Summarize the design delta relative to [`<component>.md`](../<component>.md) in
plain, natural language. Put the full timeless component architecture, pipeline
table, and IR examples directly into [`<component>.md`](../<component>.md) and
any boundary interface changes (`.h` headers and `.td` TableGen definitions)
directly into the chained CL 2 commit rather than duplicating them here:

- **How It Works & Key Invariants**: State the core mechanism, the invariants it
  preserves, and how unsupported inputs or violated preconditions fall back
  gracefully (citing the updated sections of
  [`<component>.md`](../<component>.md) and any `.h` / `.td` interface files
  updated in CL 2, or explaining the change inline if no living component doc
  applies).
  - *Default (Prose Principles & Examples)*: State principles, invariants, and
    1–2 representative examples in short sentences; never transcribe C++
    conditionals (`if (a && !b)`) into English prose.
  - *Opt-In (Compact Tables / Pseudocode)*: When specifying multi-category
    heuristics or decision matrices that do not belong in `<component>.md`, use
    a compact table (`< 15` words per cell) or a short pseudocode block instead
    of dense prose paragraphs.

### Alternatives Considered

Compare realistic alternatives a teammate could genuinely advocate for (avoid
strawmen). State why the chosen design wins, what downside it accepts, and why
each rejected option was not selected:

- **<Alternative 1>**: <Main advantage, and why it was rejected>.
- **<Alternative 2>**: <Main advantage, and why it was rejected>.

---

## 3. Implementation & Verification Plan

List the implementation steps (with affected files inline) and verification
tests in one unified checklist. Leave tasks unchecked (`- [ ]`) during the
design phase (CL 1) and check them off (`- [x]`) during CL 2 coding. Group tasks
into `CL 2a`, `CL 2b`, ... if splitting the implementation across multiple code
CLs.

- [ ] **Step 1 (`path/to/file.cpp`)**: <Concise description of change>
- [ ] **Step 2 (`path/to/file.h`, `path/to/file.cpp`)**: <Concise description of
  change>
- [ ] **Tests (`tests/path/to/test`)**:
  - *Positive*: <Cases verifying the new behavior> (`bazel test --config=dev
    //tests/...`)
  - *Negative / Fallback*: <Cases verifying graceful fallback or rejection>
- [ ] **Build, Doc & Archive Sync**: Sync `BUILD.bazel`, `CMakeLists.txt`,
  `refresh_compile_commands`, and `README.md` (if flags changed), verify
  `<component>.md` and `INDEX.md`, move this doc to
  `docs/design/archive/<change>.md` via `git mv`, and run
  `./scripts/format-code.sh`.

### Implementation Notes & Adaptations

Record only non-trivial design adaptations made during coding (omit or keep
`*(None yet)*` if none).

- *(None yet)*

---

## 4. Follow-Up Items (For Reviewer Approval -> Buganizer `564546727`)

Record work deferred during design (CL 1) or adjacent issues discovered during
implementation (CL 2). Once the implementation CL receives `+2`, run
`/implement-design file-followups` to file them in Buganizer under component
`564546727`.

- [ ] **<Short Follow-Up Title>** (`P2` | `path/to/file.cpp`):
      1–2 sentence description of the follow-up problem and proposed direction.
