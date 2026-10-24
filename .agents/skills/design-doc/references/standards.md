# Design Document Authoring, Interaction & Audit Standards

This document defines the general engineering, prose, and review standards used
by both the authoring session (the main agent during `draft`,
`draft-from-commits`, and `revise`) and the isolated narrow-view audit subagent
(with zero chat history during `audit`). Project-specific rules live in
[`../../../AGENTS.md`](../../../AGENTS.md).

---

## 1. Interaction, Alignment Check & Scope Right-Sizing (`draft` & `revise`)

1. **Default to Stress-Testing (Critique First, No Flattery)**:
   - Never agree by default or use validating praise (*"great idea"*,
     *"brilliant"*, *"spot-on"*, *"excellent"*) or filler apologies.
   - Lead with what is missing, risky, or could fail, and present the strongest
     counter-argument before affirming a design choice.
2. **Pre-Draft Alignment Check & Scope Right-Sizing**:
   - Proactively infer answers from the codebase and living component docs
     first. Unless asked to write `<change>.md` immediately in one shot, pause
     before generating the full document to present:
     1. **Baseline Understanding**: Plain-language summary of the current flow
        and invariants.
     2. **Assumptions & Alternatives**: Risks, edge cases, reuse opportunities,
        and 2–3 viable (non-strawman) options with trade-offs.
     3. **Scope Right-Sizing**: If the change is large (>300–400 LOC or spanning
        multiple independent features), propose either:
        - **Option A — Tighten the Spec & Defer Follow-Ups (Preferred)**: Narrow
          `<change>.md` to a minimal end-to-end vertical slice and move remaining
          slices to **Part 3: Follow-Up Items**.
        - **Option B — Multi-CL Implementation Plan (`CL 2a`, `CL 2b`, ...)**:
          Review the full architecture in CL 1 and group **Section 2.2** into
          independently testable implementation CLs.
     4. **Clarifying Question**: Ask at most **one** unresolved design question
        at a time.
3. **Contradiction Auditing Gate**:
   - If a new developer request or inline `A:` note contradicts an earlier
     design goal or an established system invariant, **do not modify any files
     in that turn**. Point out the contradiction in plain text and ask how to
     resolve it first.

---

## 2. Structural & Architectural Standards (**Part 1: Design Specification**)

1. **Strict Objective Isolation & Falsifiable Scoping (Section 1.1)**:
   - **Never put the solution inside the Objective / Problem Statement**: State
     *only* the problem space, current system limitation, and measurable target
     outcomes. Keep the proposed implementation mechanism (specific classes,
     files, or algorithms) in **Section 1.4**.
   - **Goals vs. Non-Goals**: Every Goal must map to a verifiable outcome or
     test. Every Non-Goal must state a plausible scope assumption that the
     design **deliberately refuses to solve** in this iteration, paired with a
     1-line reason (or tracking reference).
2. **Decisive Proposals & the Rational Decision Formula (Section 1.3)**:
   - **Decisive Stance**: Advocate for a single recommended option and
     explicitly admit its downsides.
   - **The 3-Part Rational Decision Formula**: Justify the chosen design using
     the formula:
     > *"Quality/Constraint X is a priority over Y, so we chose Option 1,
     > accepting downside Z."*
   - **Anti-Strawman Rejection Checklist**: Only list alternatives a competent
     engineer could genuinely advocate for. For every rejected alternative,
     state:
     1. *Why it was considered* (its main advantage),
     2. *Where it failed* (which higher-priority requirement or constraint it
        violates), and
     3. *What condition would make us reconsider it* in the future.
3. **Design by Contract: Imposed vs. Offered Specifications (Section 1.4)**:
   - When specifying APIs, passes, interfaces, or module boundaries, explicitly
     state both sides of the contract:
     - **Imposed Specifications (Preconditions / Caller Obligations)**: What
       input invariants, state, or constraints the caller or upstream stage must
       guarantee on entry.
     - **Offered Specifications (Postconditions / Callee Guarantees)**: What
       output invariants or state transitions hold on success, and the exact
       error or fallback behavior when preconditions are not met.
   - **Make Invalid States Unrepresentable**: Prefer types, enums, or structures
     that prevent invalid state combinations over sentinel values or loosely
     coupled booleans.
   - **Steady-State First, Edge Cases Second**: Walk through the default
     steady-state flow in cohesive prose before detailing boundary conditions,
     error paths, and fallbacks.
4. **The "Zoom-Out" Standard: Root Problem, YAGNI, Altitude & Reuse**:
   - **Root Cause vs. Symptom**: Solve the underlying root problem rather than
     patching downstream symptoms with local workarounds.
   - **YAGNI (You Aren't Gonna Need It)**: Reject abstractions, configuration
     flags, interfaces, or code paths built for speculative future requirements.
   - **Anti-Weeds (Altitude Control)**: Keep **Part 1** focused on architecture,
     contracts, data flows, signatures, and representative examples; do not drop
     into low-level implementation weeds (private helper bodies or local loop
     variables).
   - **Codebase Reuse Verification**: Before proposing a new utility, class, or
     abstraction, search the codebase to verify whether an existing utility,
     interface, or upstream component can be reused.
5. **Positive & Negative/Fallback Verification (Section 1.5)**:
   - Specify both **Positive Tests** (verifying new behavior) and **Negative /
     Fallback Tests** (verifying that violated preconditions or unsupported
     inputs are rejected or fall back gracefully without side effects or
     regressions).

---

## 3. Prose, Terminology & Citation Standards

1. **Natural Language First, Key Code Pointers Sparingly (No Symbol Dumping)**:
   - Never open a section or explain a flow with a wall of `file.cpp:123`,
     `InternalHelper()`, and raw enum names *instead of* explaining the logic.
   - Explain how the system works in plain natural language first by logical
     component role (calibrated to a newly hired L5 SWE outside the team) so the
     paragraph is clear even if the reader skips every parenthetical code
     pointer.
   - Add at most 1–2 parenthetical symbol/file pointers per paragraph where they
     anchor a key entrypoint, and keep exhaustive file-by-file inventories in
     **Part 2 (Implementation Plan)**.
2. **Self-Containment & Terminology Provenance**:
   - Define every custom term, domain abbreviation, or internal component inline
     on first use or link to its living component doc. Never guess or invent
     expansions for unfamiliar acronyms.
3. **Symbol References Over Fragile Line Numbers**:
   - Reference code objects by file path and symbol/function name rather than
     raw line numbers so references remain valid across edits and rebases.
4. **Falsifiable Language & Clean Prose Craft**:
   - **Purge Subjective Adjectives**: Replace non-falsifiable words (*"fast"*,
     *"low overhead"*, *"scalable"*, *"simple"*, *"clean"*, *"easy to
     maintain"*, *"soon"*) with concrete, verifiable metrics, bounds, or
     structural reasons.
   - **Disambiguate Pronouns**: Replace ambiguous pronouns (*"we"*, *"they"*,
     *"it"*) with the explicit system component, module, or actor performing the
     action.
   - **Active Verbs & "Given -> New" Flow**: Make real components the
     grammatical subjects and their actions the main verbs; avoid nominalizations
     ("zombie nouns" like *"perform an evaluation of"* -> *"evaluate"*).
   - **Say It Once & Visual Restraint**: Do not repeat table contents in long
     prose paragraphs below the table; keep table cells concise (`< 15` words
     per cell) and avoid AI filler vocabulary (*leverage*, *utilize*,
     *streamline*, *orchestrate*, *facilitate*, *delve*, *holistic*, *seamless*,
     *robust*, *overarching*, *pivotal*, *paradigm*).

---

## 4. Design Doc Audit Rubric & Fact-Checking Protocol (`audit`)

### 4.1 Audit Rubric for the Isolated Narrow-View Subagent (`/design-doc audit`)

The isolated zero-history subagent evaluates `<change>.md` across three
dimensions:

1. **Blind Comprehension & Terminology Check**:
   - In 1 short paragraph, explain in your own words how the current system
     works and what the proposed change does, based *strictly* on `<change>.md`.
   - Flag any place where you had to guess unstated context, any undefined or
     unlinked domain terms/acronyms, or any code-symbol dumping that obscures
     the plain-English flow.
2. **Zoom-Out, Contracts & Referenced-File Audit**:
   - **Referenced Symbols & Files**: Do all file paths, symbols, interfaces, and
     build targets referenced in `<change>.md` exist and match the codebase?
   - **Objective & Trade-Off Rigor**: Is **Section 1.1** free of implementation
     details? Does **Section 1.3** evaluate realistic (non-strawman)
     alternatives using the Rational Decision Formula (including accepted
     downsides and reconsideration triggers)?
   - **Zoom-Out (Root Cause, YAGNI & Reuse)**: Does the design solve the root
     problem without speculative generality (YAGNI), and could any proposed
     custom logic reuse an existing codebase or upstream utility?
   - **Contracts, Edge Cases & Project Rules**: Are **Imposed Preconditions**
     and **Offered Postconditions** (including error/fallback behavior) explicit
     in **Section 1.4**, and does the design comply with
     [`../../../AGENTS.md`](../../../AGENTS.md)?
3. **Scope, 1-Pass Implementation Readiness & Style Craft**:
   - **Actionability (Part 2)**: Does **Section 2.1** group affected files by
     component (sorted lexicographically within each component, with an empty
     line between components and 6-space indented description sub-bullets)? Does
     **Part 2** specify every affected file, signature, task order, and positive
     + negative/fallback verification command needed to implement the change in
     one pass without making independent design decisions?
   - **Scope Sizing**: If the change is too large for a single implementation
     CL, recommend either tightening the spec (deferring slices to **Part 3:
     Follow-Up Items**) or splitting **Section 2.2** into staged CLs (`CL 2a`,
     `CL 2b`, ...).
   - **Prose & Falsifiability**: Flag subjective adjectives lacking concrete
     bounds, ambiguous pronouns, or Part 1 code dumps.

### 4.2 Main-Agent Fact-Checking Protocol (No Automatic Edits)

When the isolated audit subagent finishes:
1. **Do not modify any files automatically.**
2. **Fact-check every finding against the codebase**: Label each finding (`#1`,
   `#2`, ...) as **[Verified]** or **[Refuted by `<file>` (`<symbol>`)]** (when
   the subagent missed an existing guarantee; for refuted points, suggest
   whether to clarify `<change>.md` or record the rebuttal).
3. **Present the 1-paragraph Blind Summary prominently first** so the developer
   can verify that a zero-context reader's understanding matches their intent,
   followed by the verified findings list, and wait for direction.
