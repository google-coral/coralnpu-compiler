# Drafting & Auditing Design Docs from Existing Commits (`draft-from-commits`)

Apply these rules on top of [`../SKILL.md`](../SKILL.md) and
[`standards.md`](standards.md) when running
`/design-doc draft-from-commits [<commits>] [<change>]` or when
`./agent-temp/<change>.commit` exists during `audit` or `revise`.

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
2. **Alignment Check Additions (Section 1.2 of [`standards.md`](standards.md))**:
   - **Confirm Range & `<base_sha>` Context**: Show
     `git log --oneline <base_sha>..<tip_sha>` and `<base_sha>`, and evaluate
     the baseline flow, reconstructed **Section 1.3** alternatives, and any edge
     cases or [`../../../AGENTS.md`](../../../AGENTS.md) issues in the commits
     from the perspective of `<base_sha>`.
   - **Single vs. Multiple Design Docs**: When `<base_sha>..<tip_sha>` spans
     multiple features or subsystems, propose whether to keep a **Single
     Cohesive Design Doc** (`changes/<change>.md`) or **Split into Multiple
     Design Docs** (`changes/<change_1>.md`, `changes/<change_2>.md`, ... with
     sub-ranges `<sub_base_i>..<sub_tip_i>`).
3. **"Written-Before-Code" (Anti-Hindsight) Authoring Rules**:
   - **Anchor Baseline at `<base_sha>`**: Every claim in **Section 1.2** and
     every reference to "existing" files or symbols must hold true at
     `<base_sha>`. Never describe symbols or files added in
     `<base_sha>..<tip_sha>` as already existing.
   - **Fold Intermediate Fixups**: Synthesize the net architecture at
     `<tip_sha>`; never structure the spec or **Part 2** tasks around
     intermediate bugs introduced and fixed within `<base_sha>..<tip_sha>`.
   - **Prospective Voice & Zero Commit Metadata**: Use prospective specification
     language (*"Add..."*, *"Introduce..."*) rather than retrospective commit
     language (*"This commit fixed..."*), and never mention commit SHAs inside
     `<change>.md`.

---

## 3. Pre-Commit (`<base_sha>`) Isolation for Step 2 (`audit`)

When `./agent-temp/<change>.commit` exists during `/design-doc audit [<change>]`,
prevent the isolated audit subagent from seeing `<base_sha>..<tip_sha>`:

1. **Create a Unique Temporary Snapshot at `<base_sha>`**:
   - Read `BASE_SHA` and `TIP_SHA` from `./agent-temp/<change>.commit` and
     populate a read-only `git archive` snapshot:
     ```bash
     mkdir -p ./agent-temp
     mktemp -d ./agent-temp/audit-snapshot-<change>-XXXXXX   # prints <snapshot_dir>
     git archive <base_sha> | tar -x -C <snapshot_dir>
     mkdir -p <snapshot_dir>/docs/design/changes <snapshot_dir>/.agents/skills/design-doc/references
     cp ./docs/design/changes/<change>.md <snapshot_dir>/docs/design/changes/<change>.md
     cp ./.agents/skills/design-doc/references/standards.md <snapshot_dir>/.agents/skills/design-doc/references/standards.md
     cp ./.agents/AGENTS.md <snapshot_dir>/.agents/AGENTS.md
     ```
   - **Submodules (`third_party/`)**: Never symlink `./third_party` into
     `<snapshot_dir>`. If `<change>.md` references files under `third_party/...`,
     copy **only** those individual files into `<snapshot_dir>/third_party/...`
     (using their `<base_sha>` version if modified in `<base_sha>..<tip_sha>`).
2. **Scope the Isolated Subagent Strictly to `<snapshot_dir>/`**:
   - Spawn the zero-history subagent with **Section 4.1** of
     [`standards.md`](standards.md), instructing it to treat `<abs_snapshot_dir>`
     as the repository root, read **only** files inside `<abs_snapshot_dir>/`,
     run **no** `git` commands, edit **no** files, and not be told that the
     change was already implemented.
3. **Clean Up Snapshot & Dual-Baseline Fact-Check**:
   - Remove only this run's snapshot (`rm -rf <snapshot_dir>`).
   - Follow **Section 4.2** of [`standards.md`](standards.md), fact-checking
     against both `<base_sha>` and `<tip_sha>`:
     - **[Verified — Spec Gap / Hindsight Leak]**: `<change>.md` omitted a
       detail from `<base_sha>..<tip_sha>` or referred to a new symbol/file as
       existing at `<base_sha>`.
     - **[Verified — Design/Code Issue in `<base_sha>..<tip_sha>`]**: Genuine
       edge case, fallback gap, or `AGENTS.md` violation in the commits.
     - **[Refuted by `<file>` (`<symbol>`) at `<base_sha>`]**: Existing
       guarantee at `<base_sha>`.

---

## 4. Rebase Before Gerrit Push & Transition to CL 2 (`revise` Override)

When `./agent-temp/<change>.commit` exists:
1. **Before Pushing CL 1 to Gerrit**: Remind the developer to commit
   `./docs/design/changes/<change>.md` as CL 1 on `<base_sha>`, **rebase the
   implementation commit(s) on top of the CL 1 commit** (re-rebasing if `revise`
   amends CL 1), and push CL 1 to Gerrit.
2. **After CL 1 Receives `+2` in Gerrit**: Skip `/implement-design code`; remind
   the developer to check out the rebased implementation commit(s) (CL 2), start
   a **fresh session**, and invoke `/implement-design audit <change>` directly.
