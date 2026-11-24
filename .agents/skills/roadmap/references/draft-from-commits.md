# Drafting & Challenging Roadmaps from Existing Commits

Apply these rules on top of [`../SKILL.md`](../SKILL.md) and
[`standards.md`](standards.md) when running
`/roadmap draft-from-commits [<commits>] [<topic>]` or when
`./agent-temp/<topic>_roadmap.commit` exists during `/roadmap draft` (inline
`A:` refinements) or `/roadmap challenge`.

---

## 1. Resolving `<commits>` (`<base_sha>..<tip_sha>`) & Recording Baseline

1. **Resolve `<commits>` and `<topic>` (Flexible Argument Order)**:
   - `<commits>` and `<topic>` may be passed in either order (or omitted);
     disambiguate tokens via `git rev-parse --verify`.
   - Resolve `<base_sha>` as the parent commit immediately **before** the first
     included commit (e.g., `<commit>~1` for a single commit defaulting to
     `HEAD`), and `<tip_sha>` as the last included commit.
2. **Persist Per-Roadmap Baseline in `./agent-temp/<topic>_roadmap.commit`**:
   - Create `./agent-temp/` if needed and write:

     ```text
     BASE_SHA=<base_sha>
     TIP_SHA=<tip_sha>
     ```

---

## 2. `draft-from-commits` Overrides for Step 1 (`draft`)

1. **Inspect `<base_sha>..<tip_sha>` and `<base_sha>`**:
   - Inspect `git log --oneline <base_sha>..<tip_sha>`,
     `git log --stat <base_sha>..<tip_sha>`, and diffs across
     `<base_sha>..<tip_sha>`, read pre-change files at `<base_sha>`
     (`git show <base_sha>:<path>`), and include any `b/<id>` from commit
     messages in the Buganizer lookup.
2. **Treat Commits as a Prototype (Abstract to Roadmap Altitude)**:
   - Never mirror raw commits 1:1 into `<change>` slices or transcribe C++ diffs
     into `<topic>_roadmap.md` (`standards.md` §1.3 strict roadmap altitude).
   - Squash intermediate fixups, split or regroup tangled commits into clean,
     independently landable `<change>` slices (`< 300–400 LOC`) across natural
     tracks with explicit conflict boundaries, and highlight gaps in the
     prototype (missing tests/CI gates, unaddressed fallbacks, sibling ops, or
     temporary workarounds to defer or reject in Section 3).
3. **Turn-1 `<base_sha>` Synthesis & 1-at-a-Time Interview**:
   - Anchor reconnaissance at `<base_sha>` using `git show <base_sha>:<path>`
     and `git grep <pattern> <base_sha> -- ...` (never grep the `<tip_sha>`
     working tree).
   - In Turn 1 (`standards.md` §1.4), show `git log --oneline
     <base_sha>..<tip_sha>` and `<base_sha>`, summarize the proposed clean
     track/slice decomposition versus the prototype commits (noting where
     commits should be split, merged, dropped, or extended), and ask **Question
     1** of [`interview.md`](../../design-doc/references/interview.md) §2
     (inviting additional sources before answering). When refining an existing
     roadmap via `/roadmap draft <topic>` while `.commit` exists, keep targeted
     `A:` reconnaissance anchored at `BASE_SHA`.
4. **"Written-Before-Code" (Anti-Hindsight) Authoring Rules**:
   - **Anchor Section 1 at `<base_sha>`**: Every claim in Section 1's `HEAD`
     Baseline must hold true at `<base_sha>` (never cite passes, flags, or files
     added in `<base_sha>..<tip_sha>` as already landed).
   - **Prospective Language & No Commit Metadata**: Use prospective roadmap
     phrasing in Section 2 (*"Add..."*, *"Introduce..."*), and omit commit SHAs
     and design-brief symbol/field inventories from `<topic>_roadmap.md`.

---

## 3. Pre-Commit (`<base_sha>`) Isolation for Step 2 (`challenge`)

When `./agent-temp/<topic>_roadmap.commit` exists during `/roadmap challenge`,
isolate the zero-history subagent from prototype code in
`<base_sha>..<tip_sha>`:

1. **Create a Temporary Snapshot at `<base_sha>`**:
   - Read `BASE_SHA` and `TIP_SHA` from `./agent-temp/<topic>_roadmap.commit`
     and populate a read-only `git archive` snapshot containing `<base_sha>`
     plus `<topic>_roadmap.md` and skill references (no code from
     `<base_sha>..<tip_sha>`):

     ```bash
     mkdir -p ./agent-temp
     mktemp -d ./agent-temp/roadmap-snapshot-<topic>-XXXXXX   # prints <snapshot_dir>
     git archive <base_sha> | tar -x -C <snapshot_dir>
     mkdir -p <snapshot_dir>/docs/design \
       <snapshot_dir>/.agents/skills/roadmap/references \
       <snapshot_dir>/.agents/skills/design-doc/references
     cp ./docs/design/<topic>_roadmap.md <snapshot_dir>/docs/design/<topic>_roadmap.md
     cp ./.agents/skills/roadmap/references/*.md <snapshot_dir>/.agents/skills/roadmap/references/
     cp ./.agents/skills/design-doc/references/*.md <snapshot_dir>/.agents/skills/design-doc/references/
     cp ./.agents/AGENTS.md <snapshot_dir>/.agents/AGENTS.md
     ```

   - **Submodules (`third_party/`)**: Never symlink `./third_party` into
     `<snapshot_dir>`. Copy only individual `third_party/...` files explicitly
     referenced by `<topic>_roadmap.md` (at their `<base_sha>` version).
2. **Scope the Subagent Strictly to `<snapshot_dir>/`**:
   - Spawn the zero-history wide-search subagent per **Section 4.1** of
     [`standards.md`](standards.md), instructing it to treat
     `<abs_snapshot_dir>` as the repository root, read **only** files inside
     `<abs_snapshot_dir>/`, run **no** `git` commands (use `rg
     --no-ignore-parent` and file reads so parent `.gitignore` rules for
     `./agent-temp/` do not hide snapshot files), edit **no** files, and not be
     told that prototype commits already exist.
3. **Clean Up Snapshot & Dual-Baseline Tier Mapping**:
   - Remove only this run's snapshot (`rm -rf <snapshot_dir>`) and apply
     **Section 4.2** of [`standards.md`](standards.md) against both `<base_sha>`
     and `<tip_sha>`:
     - **[Verified — Hindsight Leak or Baseline Drift]**: `<topic>_roadmap.md`
       assumed a capability from `<base_sha>..<tip_sha>` already existed at
       `<base_sha>`, or missed a real `<base_sha>` constraint. Eligible for
       **Tier 1** fixes in `<topic>_roadmap.md` to keep it truthful to
       `<base_sha>`.
     - **[Verified — Prototype Scope/Slicing Improvement]**: Uncovered a cleaner
       track/slice decomposition, missing cross-stack piece, or conflict hazard
       compared to `<base_sha>..<tip_sha>`. **Always route to Tier 2**
       (`interview.md` §2) so the developer decides whether to adopt the
       improved plan or record the trade-off in Section 3.
     - **[Refuted by `<file>` (`<symbol>`) at `<base_sha>`]**: Existing
       guarantee at `<base_sha>` (compact footer).
