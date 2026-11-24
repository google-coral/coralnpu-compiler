# Context Priming & Multi-Turn Co-Design Interview

Use this reference during **Step 1 (`draft` and `draft-from-commits`)**, **Step
2 (`challenge`)**, **Step 3 (`audit`)**, and **Step 4 (`revise`)** of
[`../SKILL.md`](../SKILL.md)—as well as **Section 2** during `/roadmap`
(`draft`, `draft-from-commits`, `challenge`, `to-doc`, `from-doc`) and Tier 2
escalations in `/implement-design audit`—to give the developer self-contained
codebase context and walk down open design decisions before writing or revising
design or roadmap artifacts.

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
   - **During `challenge`, `audit`, `revise`, `/implement-design audit`, or
     `/roadmap` (`draft`, `draft-from-commits`, `challenge`, `to-doc`,
     `from-doc`)**: Use the same 1-at-a-time question format whenever open
     roadmap scope/track forks, Tier 2 findings, inline `A:` notes, Gerrit
     comments, re-export choices, or unresolved Google Doc comments/conflicts
     open a trade-off, scope/contract change, or prior-decision conflict. When
     `>= 2` Tier 2 items are queued, list the remaining topics in 1 line before
     asking Question 1, and apply each resolved decision as it settles
     (following the active skill's verification cadence).
2. **Never Ask Codebase-Answerable Questions**: Every question must resolve a
   **design trade-off, scope boundary, or unwritten requirement/priority**,
   never a fact readable from the repository.
3. **Question Format (Plain Markdown in Chat with Self-Contained Context)**:
   - Present each question as **plain Markdown text in chat** (never use
     `ask_question` UI modals, so the developer can reply with a number `1`, a
     hybrid tweak, a free-form background question, or—during `draft`—`Continue`
     after editing `./agent-temp/<change>-draft-prep.md`).
   - **Example-First Self-Contained Context**: Open every question with a
     scannable context block (never one dense paragraph cramming multiple
     formulas, constraints, or unrelated sub-topics):
     - **Anchor in today**: State what exists today (inputs, IR ops, parent
       key/siblings, or current output) before stating the problem.
     - **Show a worked example**: Illustrate schemas, IR, formulas, or failure
       modes with a short snippet, 3–4 row table, or plugged-in numbers.
     - **Plain English, one topic**: State the plain question each competing
       rule answers (no jargon shorthand), and split unrelated sub-topics
       across turns or separate bullets.
     - **Prior-Decision & Multi-Pass Oscillation Context**: Whenever a surfaced
       item reopens an earlier or persisted decision or reflects back-and-forth
       oscillation across review rounds (for example, zero-history subagents
       pulling pass placement, abstraction level, or detail back and forth),
       state inside this context block: (1) what was decided earlier and why,
       and (2) what new evidence challenges it—or what competing constraint is
       driving the repeated oscillation.
   - **Numbered Reply Options (`1..N`, with Explicit Pros, Cons & Trade-Offs)**:
     - **`1. (Recommended) <Option>`**: Recommended choice with concise pros,
       cons, and trade-offs.
     - **`2..k. <Alternative Option(s)>`**: Realistic alternatives with pros,
       cons, and trade-offs. **Strong Lens Evidence Only**: If a lens found
       strong evidence for an option (e.g., *"IREE solves the same problem this
       way in the GPU pipeline"* or *"a minor fix in a different pass bypasses
       the problem"*), state it shortly and clearly in plain prose without
       `file:line` noise; omit it otherwise so options stay easy to read.
     - **Simplicity & Tie-Breaker Default (Review / Audit Escalations)**:
       Whenever a Tier 2 question proposes expanding scope, adding machinery, or
       reopening an oscillating/settled choice, include an explicit option to
       **keep the current simpler design** and persist a **1–2 line** rationale
       (with no `[Lens: ...]` tags) in the active persisted decision record
       (`### Alternatives Considered` / `Non-Goals`, `Implementation Notes &
       Adaptations`, or roadmap Section 3) so subsequent fresh-session passes
       stop reopening it. Make that keep-current option **`1. (Recommended)`**
       for oscillation or speculative-addition escalations unless concrete
       failure evidence outweighs the added complexity.
     - **`<k+1>. Explain the background / current codebase flow or elaborate on
       the options before deciding`** *(during `draft`, include *"or a filtered
       lens idea"*)*: Always include this option. If selected (or if the
       developer asks any background question in chat), explain the surrounding
       code/pipeline flow and unpack the trade-offs thoroughly, then re-present
       the decision question without advancing prematurely.
     - **`<k+2>.` Fast-Exit Option (Conditional)**:
       - *During `draft` (once the core architecture is chosen)*: **`Use
         recommended defaults for remaining details and draft the docs now`** —
         exits the interview and proceeds to authoring (`SKILL.md` Step 1.3).
       - *During review/audit steps (only when `>= 2` Tier 2 items remain; omit
         when only 1 item remains)*: **`Apply recommended choices for this item
         and all remaining queued items (#<i>–#<N>) now`** — applies `1.
         (Recommended)` for the current question plus the rest of the queue and
         finishes the step.
