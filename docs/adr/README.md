# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for significant GenGPU technical decisions.

ADRs preserve the reasoning behind decisions that affect architecture, dependencies, public interfaces, compatibility semantics, or long-term implementation direction. They are intended to help contributors understand not only what was chosen, but why.

## When to write an ADR

Write an ADR when a decision has one or more of these characteristics:

- It changes or establishes a major component boundary.
- It introduces, replaces, or materially changes a dependency or technology.
- It defines a public API, ABI, ownership, or resource-lifetime contract.
- It commits GenGPU to a compatibility or execution semantic.
- It has meaningful long-term consequences or constrains future design choices.

Routine implementation details, local refactors, and reversible choices with no architectural impact generally do not need an ADR.

When uncertain, propose the ADR rather than allowing an important decision to remain implicit.

## Lifecycle

Use these statuses:

- **Proposed** — under discussion; not yet an accepted project decision.
- **Accepted** — approved and considered the current decision.
- **Rejected** — considered but not selected.
- **Superseded** — replaced by a later ADR. Link to the replacement.
- **Deprecated** — no longer recommended, with no direct replacement.

Typical lifecycle:

1. Copy `template.md` to the next numbered file, for example `0001-use-llvm.md`.
2. Assign the next sequential number. Do not reuse numbers, including for rejected or superseded records.
3. Set the initial status to **Proposed**.
4. Describe the context, decision drivers, options considered, proposed decision, and consequences.
5. Discuss the proposal before implementation when the decision is consequential.
6. Change the status to **Accepted** only after the decision is approved.
7. Once accepted, do not silently rewrite its historical rationale. If the decision changes, create a new ADR and mark the earlier record **Superseded**, linking to the new record.

A proposal may be rejected. Keep rejected ADRs so the project retains the decision history.

## Naming and cross-references

- Use four-digit, zero-padded sequence numbers: `0001`, `0002`, and so on.
- Use lowercase kebab-case summaries in filenames: `0001-rust-cpp-boundary.md`.
- Use the matching title format inside the document: `ADR-0001: Rust/C++ Boundary`.
- Link related, superseding, or superseded ADRs with relative Markdown links.
- Keep one primary decision per ADR. Split unrelated decisions into separate records.

## Writing guidance

An effective ADR is concise but sufficiently complete for a future contributor to understand the trade-offs. Include:

- **Context** — the problem, constraints, and relevant system state.
- **Decision drivers** — the requirements that matter most.
- **Options considered** — credible alternatives and their trade-offs.
- **Decision** — the chosen direction, stated unambiguously.
- **Consequences** — benefits, costs, risks, and follow-up work.

Distinguish facts from assumptions. Cite external specifications or documentation when they materially support a decision. Record unresolved questions rather than presenting them as settled facts.

## Current records

- [ADR-0001: Context Ownership and Threading](0001-context-ownership-and-threading.md) — Accepted. Defines context ownership, backend lifetime, scoped current-context behavior, and initial thread-safety constraints.
- [ADR-0002: Synchronous Host–Device Memory Transfers](0002-synchronous-host-device-transfers.md) — Accepted. Defines borrowed-buffer transfer APIs, exact context ownership checks, bounds validation, synchronous completion, and error semantics.
