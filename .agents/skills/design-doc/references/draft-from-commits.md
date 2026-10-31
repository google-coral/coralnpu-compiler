# Drafting, Auditing & Committing Design Docs from Existing Commits (`draft-from-commits`)

Apply these rules on top of [`../SKILL.md`](../SKILL.md) and
[`standards.md`](standards.md) when running
`/design-doc draft-from-commits [<commits>] [<change>]` or when
`./agent-temp/<change>.commit` exists during `audit`, `revise`, or `commit`.

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
   - Run `git log -p <base_sha>..<tip_sha>` and `git diff <base_sha>..<tip_sha>`,
     inspect pre-change files at `<base_sha>` (`git show <base_sha>:<path>`),
     and include any `b/<id>` from commit messages in the Step 1.2 Buganizer
     lookup.
2. **Treat Commits as a Prototype: Abstract, Generalize & Critique**:
   - Do **not** transcribe the C++ diff of `<base_sha>..<tip_sha>` 1:1 into
     prose (`go/writing-a-design-doc`: a design doc written after coding must
     not be a post-hoc *"how-I-implemented-it"* dump).
   - Step back from the code to extract the **underlying problem, invariants,
     and clean, slightly generalized architectural model** in
     `docs/design/<component>.md` and `docs/design/changes/<change>.md`.
   - Actively check whether stepping back to the abstract design reveals that
     parts of the existing commits in `<base_sha>..<tip_sha>` are overly
     narrow, brittle, or complicated compared to a cleaner general rule.
3. **Alignment Check Additions (Section 1.2 of [`standards.md`](standards.md))**:
   - **Confirm Range & `<base_sha>` Context**: Show
     `git log --oneline <base_sha>..<tip_sha>` and `<base_sha>`, summarize the
     abstracted design and reconstructed alternatives from the perspective of
     `<base_sha>`, and highlight any **Design-vs-Commit Improvement
     Opportunities** where the existing commits in `<base_sha>..<tip_sha>` could
     be simplified or generalized to match a cleaner design.
   - **Single vs. Multiple Design Docs**: When `<base_sha>..<tip_sha>` spans
     multiple independent features, propose whether to keep a **Single Cohesive
     Design Doc** (`changes/<change>.md`) or **Split into Multiple Design Docs**
     (`changes/<change_1>.md`, `changes/<change_2>.md`, ...).
4. **"Written-Before-Code" (Anti-Hindsight) Authoring Rules**:
   - **Anchor Baseline at `<base_sha>`**: Every claim about "current/existing"
     behavior in `changes/<change>.md` must hold true at `<base_sha>`. Never
     describe symbols or files added in `<base_sha>..<tip_sha>` as already
     existing before the change, and Keep `docs/design/<component>.md` in
     timeless `HEAD` prose describing the post-change state.
   - **Fold Intermediate Fixups**: Synthesize the clean target architecture;
     never structure the design doc or **Section 3** tasks around intermediate
     bugs introduced and fixed within `<base_sha>..<tip_sha>`.
   - **Prospective Voice & Zero Commit Metadata**: Use prospective specification
     language (*"Add..."*, *"Introduce..."*) in `changes/<change>.md` rather
     than retrospective commit language (*"This commit fixed..."*), and never
     mention commit SHAs inside `<change>.md` or `<component>.md`.

---

## 3. Pre-Commit (`<base_sha>`) Isolation for Step 2 (`audit`)

When `./agent-temp/<change>.commit` exists during `/design-doc audit [<change>]`,
prevent the isolated audit subagent from seeing the C++ implementation changes
in `<base_sha>..<tip_sha>`:

1. **Create a Unique Temporary Snapshot at `<base_sha>`**:
   - Read `BASE_SHA` and `TIP_SHA` from `./agent-temp/<change>.commit` and
     populate a read-only `git archive` snapshot containing `<base_sha>` plus
     the drafted `changes/<change>.md`, updated `docs/design/<component>.md`
     (and `docs/design/INDEX.md` if modified), and any `.h` or `.td` interface
     files modified in `<base_sha>..<tip_sha>` (without copying `.cpp`/`.cc`/`.c`
     implementation files):
     ```bash
     mkdir -p ./agent-temp
     mktemp -d ./agent-temp/audit-snapshot-<change>-XXXXXX   # prints <snapshot_dir>
     git archive <base_sha> | tar -x -C <snapshot_dir>
     mkdir -p <snapshot_dir>/docs/design/changes <snapshot_dir>/.agents/skills/design-doc/references
     cp ./docs/design/changes/<change>.md <snapshot_dir>/docs/design/changes/<change>.md
     cp ./docs/design/*.md <snapshot_dir>/docs/design/
     cp ./.agents/skills/design-doc/references/standards.md <snapshot_dir>/.agents/skills/design-doc/references/standards.md
     cp ./.agents/AGENTS.md <snapshot_dir>/.agents/AGENTS.md
     ```
   - Also copy or remove any `.h` and `.td` interface files modified or deleted
     in `git diff --name-status <base_sha>..<tip_sha> -- '*.h' '*.td'` into
     `<snapshot_dir>` so the audit subagent can verify interface declarations.
   - **Submodules (`third_party/`)**: Never symlink `./third_party` into
     `<snapshot_dir>`. If `<change>.md` or `<component>.md` references files
     under `third_party/...`, copy **only** those individual files into
     `<snapshot_dir>/third_party/...` (using their `<base_sha>` version if
     modified in `<base_sha>..<tip_sha>`).
2. **Scope the Isolated Subagent Strictly to `<snapshot_dir>/`**:
   - Spawn the zero-history subagent with **Section 4.1** of
     [`standards.md`](standards.md), instructing it to treat `<abs_snapshot_dir>`
     as the repository root, read **only** files inside `<abs_snapshot_dir>/`,
     run **no** `git` commands, edit **no** files, and not be told that the
     change was already implemented.
3. **Clean Up Snapshot & Dual-Baseline Fact-Check**:
   - Remove only this run's snapshot (`rm -rf <snapshot_dir>`).
   - Follow **Section 4.2** of [`standards.md`](standards.md), fact-checking
     against both `<base_sha>` and `<tip_sha>` (without penalizing intentional
     prose abstraction of low-level C++ branches):
     - **[Verified — Hindsight Leak or Ambiguity]**: `<change>.md` referred to a
       symbol/file added in `<base_sha>..<tip_sha>` as already existing at
       `<base_sha>`, or omitted architectural context needed by a zero-context
       reader.
     - **[Verified — Design/Code Improvement in `<base_sha>..<tip_sha>`]**:
       Stepping back to the design doc exposed an edge case, fallback gap,
       `AGENTS.md` violation, or an opportunity to simplify/generalize the code
       in `<base_sha>..<tip_sha>`.
     - **[Refuted by `<file>` (`<symbol>`) at `<base_sha>`]**: Existing
       guarantee at `<base_sha>`.

---

## 4. `draft-from-commits` Overrides for Step 4 (`commit`)

When `./agent-temp/<change>.commit` exists during `/design-doc commit [<change>]`:

1. **Create CL 1 on `<base_sha>` & Rebase Existing Implementation Commits**:
   - Do **not** create a separate CL 2 commit for `docs/design/<component>.md`
     (because the existing implementation commit(s) in `<base_sha>..<tip_sha>`
     already *are* CL 2).
   - Temporarily save or stash any uncommitted working-tree edits to
     `docs/design/<component>.md` and `docs/design/INDEX.md` before checking out
     `<base_sha>` so `git checkout` and `git rebase` do not fail on a dirty
     working tree.
   - Commit `docs/design/changes/<change>.md` as **CL 1** on `<base_sha>` (or
     amend the existing CL 1 commit on `<base_sha>` if re-running `commit`,
     preserving its `Change-Id`), rebase the implementation commit(s)
     `<base_sha>..<tip_sha>` on top of CL 1, restore the saved uncommitted
     `<component>.md` / `INDEX.md` edits into the working tree, and update
     `BASE_SHA` and `TIP_SHA` in `./agent-temp/<change>.commit` to the rebased
     commit SHAs.
2. **Offer to Fold `<component>.md` into the Target Implementation Commit**:
   - If `docs/design/<component>.md` (and/or `docs/design/INDEX.md`) was
     modified, inform the developer that those living-doc changes need to be
     folded into their existing implementation commit(s) so reviewers of CL 1
     can see the `<component>.md` diff in the chained CL 2 commit.
   - Inspect `git log --oneline <cl1_sha>..HEAD` to **identify and suggest the
     specific commit** that `docs/design/<component>.md` (and `INDEX.md`) should
     be folded into, and **offer to fold it into that commit for them**
     (preserving the commit's `Change-Id`).
   - Show the Gerrit push command so CL 1 is pushed as Ready for Review and the
     rebased implementation commit(s) are pushed as Work-In-Progress (`-o wip`)
     until CL 1 is reviewed:
     ```bash
     git push origin <cl1_sha>:refs/for/master && git push -o wip origin HEAD:refs/for/master
     ```
   - Once CL 1 receives `+2` in Gerrit, remind the developer to start a **fresh
     session** on the rebased implementation commit(s) and invoke
     `/implement-design audit <change>` (skipping `/implement-design code`
     unless they chose to refine the code based on the design review).
