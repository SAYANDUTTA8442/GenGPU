# Contributing to GenGPU

Thank you for your interest in GenGPU. GenGPU is an early-stage systems research project. Contributions should favor correctness, explicit contracts, reproducibility, and clear architectural boundaries over feature count or premature optimization.

This document describes the repository's current development workflow. Some conventions will be expanded when the C++ compiler, FFI, and hardware-dependent test infrastructure are introduced.

## 1. Before contributing

1. Read [README.md](README.md) for the product goals, architecture, scope, and roadmap.
2. Check the existing issues and discussions before starting substantial work.
3. For a significant architectural or behavioral change, discuss the proposal before implementation.
4. Keep changes aligned with the current milestone. Do not introduce future compiler, CUDA-compatibility, or GPU-runtime features as incidental scope.

The repository's documented architecture is the intended long-term design. The current source tree is smaller and reflects only the components that have actually been implemented.

## 2. Current repository layout

```text
GenGPU/
├── .github/
│   └── workflows/
│       └── ci.yml
├── crates/
│   ├── gengpu-core/
│   ├── gengpu-driver/
│   └── gengpu-runtime/
├── Cargo.toml
├── Cargo.lock
├── rust-toolchain.toml
└── README.md
```

Current crate responsibilities:

- **`gengpu-core`** — shared foundational types and contracts. Keep it independent of higher-level runtime behavior.
- **`gengpu-driver`** — low-level GPU driver integration. Driver-specific details belong here rather than in general runtime abstractions.
- **`gengpu-runtime`** — higher-level runtime abstractions built on core types and driver integration.

Do not add dependencies between crates casually. A dependency should follow a clear architectural direction and have a documented reason.

## 3. Development environment

Install Rust using [rustup](https://rustup.rs/). The repository pins the **stable** channel in `rust-toolchain.toml` and requests the minimal profile plus the `clippy` and `rustfmt` components.

From the repository root, verify the active toolchain:

```bash
rustup show active-toolchain
rustc --version
cargo --version
```

The Rust toolchain is managed by the repository. Do not silently change the toolchain channel or add toolchain components without explaining why the change is needed.

## 4. Required checks

Run the same quality gates locally before submitting Rust changes:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo check --workspace
cargo test --workspace
```

All four commands must pass. Do not suppress a warning merely to make CI green; fix the underlying issue or document a justified exception in the change.

GitHub Actions runs these checks on pushes to `main` and pull requests targeting `main`. A local pass is useful, but it does not replace checking the CI result for the submitted commit.

### GPU-dependent validation

The current CI job runs on a CPU-only GitHub-hosted Linux runner. It does not establish that NVIDIA hardware, the NVIDIA driver, PTX loading, or kernel execution works.

When a change requires a GPU, state the hardware, driver, and software versions used for validation. Separate CPU-only tests from tests that genuinely require NVIDIA hardware. Never report a hardware test as passing unless it was actually run.

## 5. Coding expectations

### Rust

- Follow the repository's `rustfmt` formatting.
- Keep Clippy warnings at zero; CI treats warnings as errors.
- Prefer explicit, understandable APIs and meaningful names.
- Handle errors explicitly. Avoid `unwrap()` or `expect()` in production paths unless the invariant and rationale are clear.
- Minimize `unsafe`. Every necessary unsafe block must have a concise safety comment explaining the conditions that make it sound.
- Document public APIs, ownership, resource lifetimes, and error behavior.
- Avoid unnecessary dependencies. Explain the need, alternatives considered, and maintenance implications when proposing one.
- Add unit or integration tests for behavior and regression tests for fixed bugs.

### Architecture

- Preserve the separation between core types, driver integration, runtime abstractions, compiler components, and the compatibility layer.
- Keep NVIDIA Driver API details behind the driver boundary.
- Do not expose Rust-specific types across the planned C ABI boundary.
- Do not claim CUDA compatibility for an operation until its semantics and tests support that claim.
- Prefer a small, complete, testable vertical slice over broad placeholder APIs.

C++ formatting, linting, build-system, and test conventions will be specified before C++ implementation is introduced. Do not assume that the Rust quality gates validate C++ code.

## 6. Tests and documentation

A contribution should include tests appropriate to its behavior:

- Unit tests for local logic and invariants.
- Integration tests for interactions between components.
- Regression tests for reproducible defects.
- Hardware tests for behavior that cannot be validated without a supported GPU and driver.

Tests should be deterministic where practical and should not depend on undocumented machine state. Document prerequisites for tests that need external hardware or software.

Update documentation when changing public interfaces, supported behavior, build instructions, architecture, or known limitations. Performance claims must include reproducible methodology and relevant environment details.

## 7. Architecture Decision Records

Significant decisions that affect module boundaries, dependencies, public interfaces, compatibility semantics, or long-term implementation direction should be recorded as Architecture Decision Records (ADRs) under `docs/adr/`.

An ADR should explain the context, the decision, alternatives considered, and consequences. Small implementation details do not need an ADR. The ADR process and template will be maintained alongside the ADR directory.

## 8. Commit messages

Use a concise Conventional Commits-style format:

```text
<type>(<optional-scope>): <imperative summary>
```

Common types:

- `feat` — add functionality.
- `fix` — correct a defect.
- `docs` — documentation-only change.
- `test` — add or modify tests.
- `refactor` — restructure code without changing intended behavior.
- `ci` — change continuous integration.
- `build` — change build configuration or dependencies.
- `chore` — maintenance work.

Examples:

```text
feat(runtime): add device enumeration interface
fix(driver): propagate initialization errors
test(core): cover device identifier validation
docs: clarify local development checks
ci: run workspace quality gates
```

Keep each commit focused. Do not combine unrelated refactors, generated files, and feature work in one commit.

## 9. Change and review workflow

For external contributions:

1. Create a focused branch from the latest `main`.
2. Make one logically cohesive change.
3. Run the required checks and relevant tests.
4. Open a pull request against `main`.
5. Describe the motivation, implementation, tests performed, limitations, and any hardware/software prerequisites.
6. Resolve review feedback and confirm the final CI result before merge.

For the current maintainer-led bootstrap phase, explicitly scoped work may be committed directly to `main` when the maintainer has approved that implementation step. This is a temporary project workflow, not a reason to bypass scope control or verification. Each implementation step should be reviewed before the next dependent step begins.

A review should examine correctness, API contracts, ownership and lifetime behavior, error handling, tests, documentation, dependency impact, and architectural consistency—not only whether the code compiles.

## 10. Scope and project principles

Do not expand the approved task without discussion. In particular:

- Do not add CUDA Runtime API compatibility claims without tests.
- Do not introduce compiler, LLVM, MLIR, PTX execution, or hardware features ahead of their roadmap milestones.
- Do not add dependencies or change toolchain policy as incidental cleanup.
- Do not weaken CI checks to accommodate an implementation defect.
- State limitations honestly; distinguish implemented, tested, and planned behavior.

GenGPU follows one central engineering rule: **correctness before performance, and evidence before claims.**
