# <Title of Change>

- **Author(s)**: <name>
- **Component Doc(s)**: [<component>.md](../<component>.md) *(updated in chained
  CL 2)*
- **Bug(s)**: b/<id>

---

## 1. Summary & Motivation

- **Paragraph 1 (Today & Gap)**: Explain how the system works today on a
  concrete example or scenario and why it breaks or falls short (cycle counts,
  memory size, failing models).
- **Paragraph 2 (Core Idea)**: State what this change does at a conceptual level
  in 2–3 plain sentences without dumping pass, helper, or field inventories.

- **Goals**:
  - Concrete, observable outcomes, capabilities, or guarantees this change
    achieves (write as user/system outcomes rather than a file-by-file or
    symbol-by-symbol changelog).
- **Non-Goals**:
  - Plausible scope extensions or open Buganizer issues (`b/<id>`) that this
    change deliberately leaves out, each with a 1-line reason (and listed in
    **Section 4: Follow-Up Items** if they should be filed as follow-up
    tickets).

---

## 2. Proposed Change & Component Doc Delta

Tell the story of the design from top to bottom in plain, natural language
before diving into component details and corner cases. Put the full timeless
component architecture, pipeline table, and IR examples directly into
[`<component>.md`](../<component>.md) and any boundary interface changes (`.h`
headers and `.td` TableGen definitions) directly into the chained CL 2 commit
rather than duplicating them here:

1. **Opening End-to-End Walkthrough (1–2 short paragraphs)**: Trace one concrete
   input or example from entry point -> transformation -> output so the reader
   has a complete mental model before any bullets.
2. **Key Mechanisms & Component Boundaries (2–4 focused bullets)**: Explain how
   each stage works and why, citing the updated sections of
   [`<component>.md`](../<component>.md) and any `.h` / `.td` interface files
   updated in CL 2 (or explaining inline if no living component doc applies).
   Never transcribe C++ conditionals (`if (a && !b)`) into English prose.
3. **Invariants, Fallbacks & Edge Cases**: Group strict invariants, graceful
   fallback rules for unsupported inputs, and corner cases at the end (or in a
   compact `< 15`-word-per-cell table / short pseudocode block) so they do not
   interrupt the main walkthrough narrative.

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

## 4. Follow-Up Items (For Reviewer Approval -> Buganizer `2161972`)

Record work deferred during design (CL 1) or adjacent issues discovered during
implementation (CL 2). Once the implementation CL receives `+2`, run
`/implement-design file-followups` to file them in Buganizer under component
`2161972`.

- [ ] **<Short Follow-Up Title>** (`P2` | `path/to/file.cpp`):
      1–2 sentence description of the follow-up problem and proposed direction.
