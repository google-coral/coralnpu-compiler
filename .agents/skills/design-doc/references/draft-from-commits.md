# Drafting, Auditing & Committing Design Docs from Existing Commits (`draft-from-commits`)

Apply these rules on top of [`../SKILL.md`](../SKILL.md) and
[`standards.md`](standards.md) when running
`/design-doc draft-from-commits [<commits>] [<change>]` or when
`./agent-temp/<change>.commit` exists during `challenge`, `audit`, `revise`, or
`commit`.

---

## 1. Resolving `<commits>` (`<base_sha>..<tip_sha>`) & Recording Baseline State

1. **Resolve `<commits>` and `<change>` (Flexible Argument Order)**:
   - `<commits>` and `<change>` may be passed in either order (or omitted);
     disambiguate tokens via `git rev-parse --verify`.
   - Resolve `<base_sha>` as the parent commit immediately **before** the first
     included commit (e.g., `<commit>~1` for a single commit defaulting to
     `HEAD`), and `<tip_sha>` as the last included commit.
2. **Persist Per-Doc Baseline in `./agent-temp/<change>.commit`**:
   - Create `./agent-temp/` if needed and write:
     ```text
     BASE_SHA=<base_sha>
     TIP_SHA=<tip_sha>
     ```
   - If splitting a commit range into multiple docs (`<change_1>.md`,
     `<change_2>.md`, ...), write a separate `./agent-temp/<change_i>.commit`
     for each sub-range (`BASE_SHA=<sub_base_i>`, `TIP_SHA=<sub_tip_i>`).

---

## 2. `draft-from-commits` Overrides for Step 1 (`draft`)

1. **Inspect `<base_sha>..<tip_sha>` and `<base_sha>`**:
   - Run `git log -p <base_sha>..<tip_sha>` and
     `git diff <base_sha>..<tip_sha>`, inspect pre-change files at `<base_sha>`
     (`git show <base_sha>:<path>`), and include any `b/<id>` from commit
     messages in the Step 1.2 Buganizer lookup.
2. **Treat Commits as a Prototype (Abstract, Generalize & Critique)**:
   - Never transcribe the C++ diff of `<base_sha>..<tip_sha>` 1:1 into prose
     (`go/writing-a-design-doc`: a post-code design doc must not be a
     *"how-I-implemented-it"* dump). Extract the underlying problem, invariants,
     and clean, slightly generalized architectural model, and check where the
     commits in `<base_sha>..<tip_sha>` are overly narrow, brittle, or complex.
3. **Baseline Triage, Optional Design-Lens Exploration & Co-Design Interview
   Anchored at `<base_sha>` ([`lenses.md`](lenses.md) &
   [`interview.md`](interview.md))**:
   - In Turn 1 ([`interview.md`](interview.md) §1), show `git log --oneline
     <base_sha>..<tip_sha>` and `<base_sha>`, present the `<base_sha>` Baseline
     Primer, the Sub-Problem Triage (`Straightforward` vs. `Open` items, plus
     any initial **Design-vs-Commit Improvement Opportunities** in
     `<base_sha>..<tip_sha>` and whether to keep a **Single Design Doc** or
     **Split into Multiple Design Docs**), and ask **Question 1** (`interview.md`
     §1.3).
   - When Design Lenses are run (targeted or full), anchor the Factual Context
     Pack in [`lenses.md`](lenses.md) at `<base_sha>` (plus a factual summary of
     what `<base_sha>..<tip_sha>` implements for the selected items) so the
     parallel Design Lens subagents independently propose how to solve or
     simplify the problem from `<base_sha>` and critique the prototype commits,
     then conduct the 1-question-at-a-time interview per
     [`interview.md`](interview.md) §2.
4. **"Written-Before-Code" (Anti-Hindsight) Authoring Rules**:
   - **Anchor Baseline at `<base_sha>`**: Every claim about "current/existing"
     behavior in `changes/<change>.md` must hold true at `<base_sha>` (never
     cite symbols/files added in `<base_sha>..<tip_sha>` as pre-existing), while
     `docs/design/<component>.md` describes the post-change `HEAD` state.
   - **Fold Intermediate Fixups & Omit Commit Metadata**: Synthesize the clean
     target architecture (never structure Section 3 around intermediate bugs
     fixed within `<base_sha>..<tip_sha>`), use prospective specification
     language (*"Add..."*, *"Introduce..."*), and never mention commit SHAs in
     `<change>.md` or `<component>.md`.

---

## 3. Pre-Commit (`<base_sha>`) Isolation for Step 2 (`challenge`) & Step 3 (`audit`)

When `./agent-temp/<change>.commit` exists during `/design-doc challenge` or
`/design-doc audit`, isolate the subagent(s) from the C++ implementation in
`<base_sha>..<tip_sha>`:

1. **Create a Temporary Snapshot at `<base_sha>`**:
   - Read `BASE_SHA` and `TIP_SHA` from `./agent-temp/<change>.commit` and
     populate a read-only `git archive` snapshot containing `<base_sha>` plus
     `changes/<change>.md`, `docs/design/<component>.md` (`+ INDEX.md`), and any
     `.h` or `.td` interface files modified in `<base_sha>..<tip_sha>` (no
     `.cpp`/`.cc`/`.c` files):

     ```bash
     mkdir -p ./agent-temp
     mktemp -d ./agent-temp/audit-snapshot-<change>-XXXXXX   # prints <snapshot_dir>
     git archive <base_sha> | tar -x -C <snapshot_dir>
     mkdir -p <snapshot_dir>/docs/design/changes <snapshot_dir>/.agents/skills/design-doc/references
     cp ./docs/design/changes/<change>.md <snapshot_dir>/docs/design/changes/<change>.md
     cp ./docs/design/*.md <snapshot_dir>/docs/design/
     cp ./.agents/skills/design-doc/references/*.md <snapshot_dir>/.agents/skills/design-doc/references/
     cp ./.agents/AGENTS.md <snapshot_dir>/.agents/AGENTS.md
     ```

   - Copy or remove any `.h` and `.td` interface files from
     `git diff --name-status <base_sha>..<tip_sha> -- '*.h' '*.td'` into
     `<snapshot_dir>`.
   - **Submodules (`third_party/`)**: Never symlink `./third_party` into
     `<snapshot_dir>`. Copy only individual `third_party/...` files referenced
     by `<change>.md` or `<component>.md` (at their `<base_sha>` version).
2. **Scope the Subagent(s) Strictly to `<snapshot_dir>/`**:
   - Spawn the zero-history subagent(s) per **Section 4** of
     [`lenses.md`](lenses.md) (`challenge`) or **Section 4.1** of
     [`standards.md`](standards.md) (`audit`), instructing them to treat
     `<abs_snapshot_dir>` as the repository root, read **only** files inside
     `<abs_snapshot_dir>/`, run **no** `git` commands, edit **no** files, and
     not be told that the change was already implemented.
3. **Clean Up Snapshot & Dual-Baseline Fact-Check**:
   - Remove only this run's snapshot (`rm -rf <snapshot_dir>`) and follow
     **Section 4.2** of [`standards.md`](standards.md), labeling findings
     against both `<base_sha>` and `<tip_sha>`:
     - **[Verified — Hindsight Leak or Ambiguity]**: `<change>.md` cited a
       symbol/file added in `<base_sha>..<tip_sha>` as pre-existing at
       `<base_sha>`, or omitted context needed by a zero-context reader.
     - **[Verified — Design/Code Improvement in `<base_sha>..<tip_sha>`]**:
       Exposed an edge case, fallback gap, `AGENTS.md` violation, or
       simplification/generalization opportunity in `<base_sha>..<tip_sha>`.
     - **[Refuted by `<file>` (`<symbol>`) at `<base_sha>`]**: Existing
       guarantee at `<base_sha>`.

---

## 4. `draft-from-commits` Overrides for Step 5 (`commit`)

When `./agent-temp/<change>.commit` exists during
`/design-doc commit [<change>]`:

1. **Create CL 1 on `<base_sha>` & Rebase Existing Implementation Commits**:
   - Do **not** create a separate CL 2 commit for `docs/design/<component>.md`
     (the implementation commits in `<base_sha>..<tip_sha>` already are CL 2).
   - Stash/save uncommitted edits to `docs/design/<component>.md` and
     `docs/design/INDEX.md`, commit `docs/design/changes/<change>.md` as **CL
     1** on `<base_sha>` (or amend the existing CL 1 commit preserving its
     `Change-Id`), rebase `<base_sha>..<tip_sha>` onto CL 1, restore the saved
     `<component>.md` / `INDEX.md` edits, and update `BASE_SHA` and `TIP_SHA` in
     `./agent-temp/<change>.commit`.
2. **Offer to Fold `<component>.md` into the Target Implementation Commit**:
   - Inspect `git log --oneline <cl1_sha>..HEAD` to suggest the specific commit
     that modified `docs/design/<component>.md` (and `INDEX.md`) should be
     folded into, and offer to fold it in (preserving its `Change-Id`).
   - Show the Gerrit push command (CL 1 Ready for Review, CL 2 WIP) and remind
     the developer to run `/implement-design audit <change>` in a fresh session
     once CL 1 receives `+2`:

     ```bash
     git push origin <cl1_sha>:refs/for/master && git push -o wip origin HEAD:refs/for/master
     ```
