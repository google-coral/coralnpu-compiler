# CoralNPU Compiler Design Documents & CL Workflow

This directory holds the architectural documentation and change design briefs
for the CoralNPU Compiler. For non-trivial changes (new compiler passes,
operation enablements, third-party patches, or changes larger than ~150 LOC),
the project uses a **Two-CL Spec-Driven Development (SDD)** workflow supported
by the `/design-doc` and `/implement-design` agent skills (detailed in
[`.agents/skills/design-doc/`](../../.agents/skills/design-doc/SKILL.md) and
[`.agents/skills/implement-design/`](../../.agents/skills/implement-design/SKILL.md)).

---

## 1. Directory Layout

```text
docs/design/
├── README.md               # This workflow guide
├── INDEX.md                # Maps each living <component>.md doc to its source paths & passes
├── <component>.md          # Living component architecture at HEAD (updated in CL 2)
├── changes/
│   ├── TEMPLATE.md         # 4-section template for new change briefs
│   └── <change>.md         # Active, unmerged change brief (created in CL 1)
└── archive/
    └── <change>.md         # Completed change briefs (moved from changes/ via git mv in CL 2)
```

---

## 2. Two-CL Division of Responsibility

To keep design documents short and easy to read while preventing
`docs/design/<component>.md` or unimplemented headers on `master` from drifting
ahead of `HEAD` code, the design phase updates two sets of files and commits
them as a **2-CL chain**:

1. **Change Design Brief (`docs/design/changes/<change>.md` — CL 1 Commit)**:
   - A concise (**80–150 line**) brief based on
     [changes/TEMPLATE.md](changes/TEMPLATE.md) covering four flat sections:
     **1. Summary & Motivation**, **2. Proposed Change & Component Doc Delta**
     (followed by **Alternatives Considered**), **3. Implementation &
     Verification Plan** (a single unified `- [ ]` checklist), and **4.
     Follow-Up Items** (for Buganizer `564546727`).
   - Uses natural-language principles and examples by default (never
     transcribing C++ conditionals into prose) and avoids duplicating
     `<component>.md` or pasting `.h`/`.td` code blocks.
2. **Living Component Doc & Boundary Interfaces (`docs/design/<component>.md` +
   `.h`/`.td` — Initial `[WIP]` CL 2 / `CL 2a..2n` Commits)**:
   - **`docs/design/<component>.md`** (+ [INDEX.md](INDEX.md) if updated): Holds
     the timeless architectural explanation at the post-change state (summary,
     pass pipeline, invariants, IR examples, tests, and `TODO(...)` limits).
   - **Boundary Interface Files (`.h` Headers & `.td` TableGen Definitions)**:
     Includes any added, modified, or deleted cross-file C/C++ header
     declarations (`.h`, with doc comments) or TableGen definitions (`.td`),
     without `.cpp` implementation bodies.
   - Committed on top of CL 1 as an **initial `[WIP]` CL 2** (or across stacked
     `[WIP]` `CL 2a .. CL 2n` commits when Section 3 is split into multiple CLs)
     and pushed to Gerrit with `-o wip`. Reviewers of CL 1 can inspect the exact
     `<component>.md` and interface diffs in the Gerrit relation chain, while
     merging CL 1 leaves `master` untouched until CL 2 lands.

---

## 3. Workflow & Skill Commands

```text
===============================================================================
               PHASE 1: DESIGN (`/design-doc` -> CL 1 + Initial CL 2)
===============================================================================
  1. `/design-doc draft`     -> Co-design & update in the working tree:
     (or `draft-from-commits`)  - `docs/design/changes/<change>.md` (brief)
           │                    - `docs/design/<component>.md` (+ `INDEX.md`)
           │                    - `.h` / `.td` boundary interfaces (if any)
           ▼
  2. `/design-doc challenge` -> Red-team via adversarial Design Lenses (wide
     + `/design-doc audit`      search) & narrow-view blind-reader audit
           │
           ▼
  3. `/design-doc commit`    -> Create (or amend) the commit chain:
           │                    - CL 1: `docs/design/changes/<change>.md`
           │                    - CL 2 (`[WIP]`): `<component>.md` + `.h`/`.td`
           ▼
  4. Push Chain to Gerrit    -> `git push origin HEAD~<n>:refs/for/master && \`
           │                    `  git push -o wip origin HEAD:refs/for/master`
           ▼
  5. `/design-doc revise`    -> Address `A:` or Gerrit comments -> Re-run
     + `/design-doc commit`     `commit` (amends CLs preserving Change-Ids)
                                -> Receive +2 & Submit CL 1

===============================================================================
                 PHASE 2: IMPLEMENTATION (`/implement-design` -> CL 2)
===============================================================================
  1. `/implement-design code`           -> On top of CL 2 (`CL 2a..2n`),
           │                               implement tasks & keep docs synced
           ▼
  2. `/implement-design audit`          -> Optional isolated zero-history subagent
           │                               (code, test, & semantic-drift audit)
           ▼
  3. `/implement-design archive`        -> Coherence pass on `<component>.md`,
           │                               `git mv` `<change>.md` to `archive/`,
           │                               and run `./scripts/format-code.sh`
           ▼
  4. Amend CL 2 & Push Ready (`-o ready`) -> Pushes Patchset 2+ of the same CL 2
           │                               -> Receive +2 & Submit CL 2
           ▼
  5. `/implement-design file-followups` -> File approved Section 4 follow-ups in
                                           Buganizer (`564546727`)
```

### Command Reference

| Command                                                     | What It Does                                                                                                                                      |
| :---------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------ |
| `/design-doc draft [<change>] <description>`                | Co-designs the change, updates `<component>.md` + `.h`/`.td` interfaces, writes `changes/<change>.md`, and stops for your review.                 |
| `/design-doc draft-from-commits [<commits>] [<change>]`     | Abstracts and critiques a pre-code design from existing commit(s) (`<commits>`, default `HEAD`), highlighting opportunities to simplify the code. |
| `/design-doc challenge [<change>]`                          | Spawns parallel adversarial Design Lens subagents (with wide codebase search) to red-team the proposed solution without editing files.            |
| `/design-doc audit [<change>]`                              | Spawns a zero-history narrow-view subagent returning a **Blind Comprehension Summary** and fact-checked findings without editing files.           |
| `/design-doc revise [<change>] [<feedback>]`                | Updates `changes/<change>.md`, `<component>.md`, and `.h`/`.td` interfaces from inline `A:` comments or Gerrit review feedback.                   |
| `/design-doc commit [<change>]`                             | Commits `changes/<change>.md` as CL 1 and `<component>.md` + `.h`/`.td` as `[WIP]` CL 2 (`CL 2a..2n`), or offers to fold `<component>.md` for `draft-from-commits`. |
| `/implement-design code [<change>]`                         | Implements Section 3 tasks on top of CL 2 (`CL 2a..2n`), keeping `<change>.md` and `<component>.md` synced as the code evolves.                   |
| `/implement-design audit [<change>] [<commits>]`            | Spawns a zero-history subagent to audit the implementation diff for semantic/invariant alignment, test coverage, and `AGENTS.md` compliance.      |
| `/implement-design archive [<change>]`                      | Runs a coherence pass on `<component>.md`, moves `changes/<change>.md` to `archive/<change>.md` via `git mv`, and runs `./scripts/format-code.sh`. |
| `/implement-design file-followups [<change>]`               | Deduplicates and files approved Section 4 follow-up items into Buganizer component `564546727` after CL 2 receives `+2`.                          |

### Key Usage Tips

- **Short `<change>` Names**: Pass just the short name (e.g., `zvt_conv_igemm`),
  or omit it when only one active doc exists in `docs/design/changes/`.
- **Inline `A:` Comments & Same-Session Edits**: Within a session, you can ask
  for edits in plain text or add `A:` comments directly inside `<change>.md` or
  `<component>.md` and run `/design-doc revise`. In a new session (or after
  `/compact`), start with the `/design-doc` or `/implement-design` command.
- **Verify the Blind Comprehension Summary (`audit`)**: Always read the
  1-paragraph summary at the top of `/design-doc audit` (and
  `/implement-design audit`) to confirm that a zero-context reader understands
  the design and code the way you intended.
- **Pushing to Gerrit (`-o wip` vs. `-o ready`)**:
  - After `/design-doc commit` (`n = 1` for a single CL 2, or the number of
    staged `CL 2a..2n` commits):
    ```bash
    git push origin HEAD~<n>:refs/for/master && git push -o wip origin HEAD:refs/for/master
    ```
  - After `/implement-design archive` and `git commit --amend` (removing
    `[WIP]`):
    ```bash
    git push -o ready origin HEAD:refs/for/master
    ```

---

## 4. Background & References

The workflow and authoring/audit standards in
[`.agents/skills/design-doc/references/standards.md`](../../.agents/skills/design-doc/references/standards.md)
and
[`.agents/skills/implement-design/references/standards.md`](../../.agents/skills/implement-design/references/standards.md)
build on three sources:
- **Elephants & Goldfish Model (`go/egm-plugin`)**: Dave Rensin's paper,
  *Elephants, Goldfish, and the New Golden Age of Software Engineering*
  (`go/design-is-the-new-code` |
  [Google Research](https://research.google/pubs/elephants-goldfish-and-the-new-golden-age-of-software-engineering/)
  |
  [Medium](https://drensin.medium.com/elephants-goldfish-and-the-new-golden-age-of-software-engineering-c33641a48874))
  — separating context-rich design sessions from isolated zero-history audit
  subagents, blind comprehension checks, the Zoom-Out Standard
  (`go/egm:standard:zoom-out`), and anti-sycophancy protocols.
- **Design Doc Structure (`go/writing-a-design-doc`, `go/tott/773`,
  `go/greendoc`, `go/bluedoc`, `go/contracts`)**: Strict objective isolation,
  proposal-first structure with alternatives after the proposal, and falsifiable
  metrics.
- **Natural-Language-First Technical Writing (`go/speak-like-tl`,
  `go/good-design-docs`, `go/how-to-write-a-design-document`)**: Explaining
  system flows in short, plain sentences without C++-in-prose transcription or
  symbol dumping.
