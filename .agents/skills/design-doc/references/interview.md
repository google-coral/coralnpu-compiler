# Context Priming & Multi-Turn Co-Design Interview (`draft`, `challenge` & `revise`)

Use this reference during **Step 1 (`draft` and `draft-from-commits`)**, **Step
2 (`challenge`)**, and **Step 4 (`revise`)** of [`../SKILL.md`](../SKILL.md) to
give the developer self-contained codebase context and walk down the design
decision tree before generating or updating design documents.

---

## 1. Turn-1 Baseline Primer, Sub-Problem Triage & Exploration Choice

After fast baseline recon (`SKILL.md` Step 1.1) and **before** spawning Design
Lens subagents or inventing options for hard problems, present a self-contained
Turn-1 briefing in chat:

1. **Baseline Understanding ("How It Works Today")**:
   - **Pipeline & Component Context**: State which passes or components run
     immediately before and after the target area, what IR, attributes, or state
     they hand off, and which living `docs/design/<component>.md` doc(s) will be
     updated alongside `docs/design/changes/<change>.md`.
   - **Current Mechanism & Root Cause**: Explain in plain language (with
     clickable file and symbol links) how the current code works today and *why*
     it fails or falls short on a concrete example or trace.
2. **Sub-Problem Triage & Scope Right-Sizing** ([`standards.md`](standards.md)
   §1.2):
   - Break the request into numbered sub-problems (`P1..Pn`) in a compact table
     or list, classifying each as:
     - **Straightforward**: Has a clear, low-risk fix from baseline recon (state
       the proposed default mechanism inline).
     - **Open Architectural Item**: Has genuine multi-way trade-offs (state the
       core tension/problem **without** inventing pre-lens solution options).
3. **Question 1 — Exploration & Triage Choice**:
   - Ask how the developer wants to proceed on `P1..Pn`:
     1. **`1. (Recommended) Lock in the straightforward items ([P1, ...]) with
        the defaults above, and run targeted Design Lenses only on the open
        items ([P2, ...])`**
     2. **`2. Lock in [P1, ...], and let me write the design outline for [P2,
        ...]`**
     3. **`3. Run full parallel Design Lens exploration across all items
        ([P1–Pn]) now (note: this can take some time, depending on the model and
        the problem)`**
     4. **`4. Explain the background / current codebase flow before deciding`**
   - *(If there are no separate straightforward items, omit "Lock in [P1, ...]"
     from Options 1 and 2.)*
   - **Always remind the developer** right below the options that they can also
     ask for **any combination** of these (for example, overriding a default on
     a straightforward item, writing an outline for `P2` while running Design
     Lenses only on `P3`, or requesting specific lenses).
4. **Handling Each Turn-1 Choice**:
   - **Option 1 or 3 (or custom lens subset)**: Spawn the parallel Design Lens
     subagents on the selected items ([`lenses.md`](lenses.md) §2), then present
     the curated `[Lens: ...]` options and Scope Right-Sizing (in chat by
     default, or `./agent-temp/<change>-draft-prep.md` when complex) and start
     the 1-at-a-time interview (§2) for the open items.
   - **Option 2 (Developer Writes Outline)**:
     1. If the developer replied with `2` without including the outline yet,
        ask them to share their outline or notes for `[P2, ...]` (as brief or
        detailed as they like).
     2. Once the outline is provided, verify it against the codebase,
        `<component>.md`, and `AGENTS.md` invariants, surfacing any
        contradictions, unhandled edge cases, or simpler existing utilities
        ([`standards.md`](standards.md) §1.3).
     3. If the outline resolves all load-bearing decisions, proceed directly to
        authoring (`SKILL.md` Step 1.3). If it leaves genuine ambiguities,
        underspecified interfaces, or edge-case forks, walk through only those
        remaining gaps **one question at a time** (§2)—where each question can
        also be deferred to targeted Design Lenses or skipped via "Use
        recommended defaults and draft now."

---

## 2. Multi-Turn 1-Question-at-a-Time Co-Design Interview

Do **not** guess unspoken requirements, trade-off preferences, or edge-case
policies just to write `changes/<change>.md` and `<component>.md` in a single
turn. Walk down the decision tree **one question at a time** across turns until
all load-bearing decisions are resolved:

1. **When to Interview**:
   - **Post-Lens Architecture & Detail Alignment**: After Design Lenses run on
     the open items, start by selecting or synthesizing the core architectural
     direction for the first open item, then walk down its dependent detail
     questions one turn at a time before moving to the next open item.
   - **Post-Outline Clarification (After Turn-1 Option 2)**: Ask 1-at-a-time
     questions only for open forks, edge cases, or contradictions in the
     developer's outline (allowing the developer to defer any hard fork to
     targeted Design Lenses if needed).
   - **During `challenge` or `revise`**: Use the same question format when
     verified `challenge` findings, inline `A:` notes, or Gerrit comments open a
     multi-branch design choice.
2. **Never Ask Codebase-Answerable Questions**: Every question must resolve a
   **design trade-off, scope boundary, or unwritten requirement/priority**,
   never a fact readable from the repository.
3. **Question Format (Plain Markdown in Chat with Self-Contained Context)**:
   - Present each question as **plain Markdown text in chat** (never use
     `ask_question` UI modals, so the developer can reply with a number `1`, a
     hybrid tweak, a free-form background question, or `Continue` after editing
     `./agent-temp/<change>-draft-prep.md`).
   - **2–3 Sentences of Self-Contained Context First**: Open every question by
     briefly explaining the relevant codebase mechanism or hardware/pipeline
     constraint so the developer never lacks background context.
   - **Numbered Reply Options (`1..N`)**:
     1. **`1. (Recommended) <Option>`**: Recommended choice and its trade-off.
     2. **`2..k. <Alternative Option(s)>`**: Realistic alternatives and
        trade-offs.
        - **Strong Lens Evidence Only**: If a lens found strong evidence for an
          option (e.g., *"IREE solves the same problem this way in the GPU
          pipeline"* or *"a minor fix in a different pass bypasses the
          problem"*), state it shortly and clearly in plain prose without
          `file:line` noise; omit it otherwise so options stay easy to read.
     3. **`<k+1>. Explain the background / current codebase flow or elaborate on
        the options (or a filtered lens idea) before deciding`**: Always include
        this option. If selected (or if the developer asks any background
        question in chat), explain the surrounding code/pipeline flow and unpack
        the trade-offs thoroughly, then re-present the decision question without
        advancing prematurely.
     4. **`<k+2>. (Include once the core architecture is chosen) Use recommended
        defaults for remaining details and draft the docs now`**: Exits the
        interview and proceeds to authoring (`SKILL.md` Step 1.3).
