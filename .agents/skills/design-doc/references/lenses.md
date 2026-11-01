# Design Lenses: Alternative Generation (`draft`) & Adversarial Red-Teaming (`challenge`)

Use this reference in two modes during [`../SKILL.md`](../SKILL.md):
- **Generative Mode (`draft` & `draft-from-commits` — Sections 2–3)**: Explore
  the design space across competing objectives before converging on a proposal.
- **Adversarial Mode (`challenge` — Section 4)**: Red-team a drafted proposal
  (`changes/<change>.md`, `<component>.md`, and `.h`/`.td` interfaces) with wide
  codebase search.

---

## 1. The 10 Design Lenses Catalog

| Lens Tag      | Core Objective & Signature Question                                                                                                                                                                                                     |
| :------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Minimalist`  | Minimize added LOC, new passes, and state, or **delete** redundant passes, analyses, and special cases (subtractive design). *"What is the smallest surgical change—or code deletion—that solves 80–100% of the problem?"*              |
| `Modularity`  | Enforce single-responsibility passes, clean component boundaries, and well-defined interfaces/invariants. *"Which component boundary owns this invariant so stages stay decoupled and composable?"*                                     |
| `Performance` | Maximize CoralNPU hardware utilization (Zvt matrix unit, RVV `LMUL`/registers, 32 KB DTCM, 3 KB stack, DMA transfers). *"What achieves the best cycle count and SRAM/register footprint without cliff-edge fallbacks?"*                 |
| `Precedent`   | Search how analogous problems are solved in other backends, targets, pipelines, or passes (`LLVMCPU`, `SPIRV`, `AMDGPU`, `VMVX`, MLIR, or sibling CoralNPU passes). *"How do other targets solve this, and how do we align with them?"* |
| `Rewriter`    | Assume willingness to deeply refactor or restructure surrounding passes and helpers to escape a local maximum or tech debt. *"What is the clean-slate design if we are not constrained by current structure?"*                          |
| `OOTB`        | Challenge the problem framing and data/IR representation. *"What non-obvious viewpoint, control-flow inversion, or IR attribute/type/metadata change makes the consumer algorithm trivial?"*                                            |
| `Skeptical`   | Challenge necessity and system boundaries (YAGNI). *"Why is this not worth solving here, or why should it be solved outside the compiler/runtime (in the model/frontend, simulator/HW config, or flags)?"*                              |
| `Generalizer` | Replace case-by-case heuristics or op-specific rules with one general structural or mathematical formulation (e.g., an indexing-map relation or interface property) that subsumes existing special cases.                               |
| `Shifter`     | Ask whether the problem is being solved at the wrong pipeline stage. *"Can we normalize/canonicalize or decompose the IR much earlier so downstream passes work untouched, or handle it much later in LLVM/CRT/HAL?"*                   |
| `Incremental` | Design a staged evolution or bridge path. *"How do we deliver immediate value in a small, low-risk Step 1 that migrates cleanly toward the long-term architecture without a flag-day rewrite?"*                                         |

---

## 2. Generative Mode: Adaptive Selection & Parallel Subagent Fan-Out

1. **Build a Neutral Factual Context Pack (Scoped to Selected Open Items)**:
   - After Turn-1 Sub-Problem Triage and Exploration Choice
     ([`interview.md`](interview.md) §1), assemble a solution-neutral **Factual
     Context Pack** covering **only the open items selected for Design Lens
     exploration**: problem statement, concrete motivating example, any
     straightforward items or developer outline constraints already locked in as
     fixed context, hardware/project constraints (`AGENTS.md`), and relevant
     pipeline stages/files/symbols (plus a factual summary of
     `<base_sha>..<tip_sha>` for `draft-from-commits`).
2. **Select Task-Relevant Lenses & Group into 4–5 Parallel Subagents**:
   - Select the 6–10 task-relevant lenses from Section 1 (recording a 1-line
     note of any skipped lenses) and group them into **4–5 parallel isolated
     read-only subagents** (`1–2` lenses per subagent) that inherit local
     file-reading and shell/git search capabilities (`git grep` / `rg`) on the
     worktree:
     - **Never pair opposing lenses** in the same subagent (e.g., never pair
       `Minimalist` with `Rewriter`, or `Skeptical` with `Generalizer`).
     - **Compatible pairings**: `Minimalist` + `Skeptical`; `Precedent` +
       `Shifter`; `Modularity` + `Generalizer`; `Rewriter` + `Incremental`;
       `Performance` + `OOTB`.
   - Pass each subagent the **Factual Context Pack**, `.agents/AGENTS.md`, and
     its assigned lens row(s) from Section 1.
   - **Subagent Mandate**: Operate read-only on the worktree (do not edit files
     or prompt the user directly), use local shell/git search (`git grep` /
     `rg`) and file reads to inspect `compiler/`, `runtime/`, and `third_party/`,
     and address every assigned lens without watering down its core objective:
     - Present concrete proposals tagged with `[Lens: <Tag>]` (mechanism,
       affected files/passes, pros, trade-offs, and any **strong evidence** such
       as an upstream IREE/MLIR precedent or a small fix in another pass that
       bypasses the problem), combining your assigned lenses into a joint
       `[Lens: <Tag1>, <Tag2>]` proposal whenever they naturally reinforce each
       other rather than inventing artificial filler.
     - Surface any critical **open questions, unstated assumptions, or
       decision forks** (*"if invariant X holds, Option A wins; if not, Option
       B"*) for the main agent to bring into the developer interview.

---

## 3. Critical Review, Curation & Presentation

Before starting the interview in [`interview.md`](interview.md):
1. **Evaluate, Deduplicate & Synthesize**: Fact-check proposals against the
   codebase, merge overlapping ideas (combining tags, e.g., `[Lens: Minimalist,
   Precedent]`), check whether a **Recommended Synthesis** combining mechanisms
   across lenses beats any single proposal, and collect the sharpest open
   decision questions surfaced by the subagents for the interview.
2. **Present Curated Options + 1-Line Filtered List**:
   - **Top 3–4 Curated Options (Plus Recommended Synthesis)**: State each
     option's `[Lens: ...]` tag(s), 2–4 sentence mechanism, pros/cons, and the
     trade-off fault lines separating the options. When a lens has **strong
     evidence** for an option (e.g., *"IREE solves the same problem this way in
     the GPU pipeline"* or *"a minor fix in a different pass bypasses the
     problem"*), state it shortly and clearly in plain prose without `file:line`
     clutter; omit evidence notes when no strong evidence exists.
   - **`Other Filtered Ideas (Ask to Elaborate)`**: 1-line summary + `[Lens:
     ...]` tag + 1-clause filter reason for every remaining lens idea, noting
     you can elaborate on request, followed by a 1-line **`Lenses Run /
     Skipped`** footer.
   - Keep `[Lens: ...]` tags strictly for interactive `draft` exploration; omit
     them in `### Alternatives Considered` inside committed
     `docs/design/changes/<change>.md` docs.

---

## 4. Adversarial Mode: Red-Teaming the Proposed Solution (`/design-doc challenge`)

Unlike `/design-doc audit` (a single narrow-view Blind Reader restricted to
cited files), `/design-doc challenge` spawns **4–5 parallel Adversarial Lens
Subagents** (`1–2` non-opposing lenses each, per Section 2.2) that inherit local
file-reading and shell/git search capabilities (`git grep` / `rg`) for **wide
codebase search** (`compiler/`, `runtime/`, `third_party/`) to stress-test
`changes/<change>.md`, `<component>.md`, and modified `.h`/`.td` interfaces:

1. **Adversarial Subagent Mandate**:
   - Read `### Alternatives Considered` and `Non-Goals` first. Do **not**
     merely re-propose a rejected alternative unless you uncover a **concrete
     flaw, unstated cost, broken invariant, unhandled edge case, or simpler
     hybrid** that `<change>.md` failed to account for.
   - Attack the proposed design strictly from each assigned lens (`[Lens:
     <Tag>]`):
     - **`Minimalist` / `Skeptical`**: What proposed `.h`/`.td` interfaces,
       pass logic, state, or scope can be deleted, merged, or solved outside
       the compiler/runtime?
     - **`Precedent` / `Shifter` / `OOTB`**: Does an uncited upstream IREE/MLIR
       utility, target hook, data/attribute representation, or earlier/later
       pipeline stage solve part of the design more cleanly?
     - **`Modularity` / `Generalizer` / `Performance` / `Rewriter` /
       `Incremental`**: Where does the proposal leak responsibilities across
       stages, rely on brittle special cases instead of a general invariant, or
       risk a hardware/performance cliff?
2. **Fact-Checking & Resolution**: Apply **Section 4.2** of
   [`standards.md`](standards.md) (`[Verified]` vs. `[Refuted]`, preserving
   `[Lens: ...]` tags, no automatic edits) and resolve resulting design choices
   via [`interview.md`](interview.md).
