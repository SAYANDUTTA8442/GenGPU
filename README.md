# GenGPU — Product Specification

**Document:** `PRODUCT.md`  
**Project:** GenGPU  
**Version:** 0.1.0  
**Status:** Initial Product Specification  
**Project Type:** Open-Source Systems Research  
**Primary Language:** Rust + C++  
**Initial Hardware Target:** NVIDIA GPUs  
**Compiler Infrastructure:** LLVM / Clang / MLIR  
**Initial GPU Target:** NVIDIA PTX  
**License:** Apache License 2.0 (proposed; subject to dependency and redistribution review)

---

# 1. Product Overview

## 1.1 What is GenGPU?

GenGPU is an open-source research project that aims to develop an independently implemented CUDA-compatible GPU programming and execution ecosystem.

The project will provide a software stack that enables developers to write GPU programs, compile supported CUDA C++ code, generate GPU executable artifacts, manage GPU resources, and execute kernels on NVIDIA hardware.

GenGPU will initially target NVIDIA GPUs through documented NVIDIA Driver API interfaces and the PTX intermediate representation.

The project will use established compiler infrastructure while independently implementing its runtime, compatibility layer, compilation orchestration, and selected GPU programming abstractions.

## 1.2 Product Vision

Build an open, extensible, and independently developed GPU computing ecosystem that reduces dependence on proprietary CUDA compilation and runtime components while preserving interoperability with NVIDIA hardware.

The long-term vision is to provide:

- An independently developed GPU compilation pipeline.
- An open-source GPU runtime.
- A CUDA-compatible programming interface.
- A modular compiler architecture.
- GPU memory and execution management.
- A growing collection of GPU computing libraries.
- Research infrastructure for heterogeneous computing.

## 1.3 Core Mission

GenGPU aims to make GPU computing infrastructure more transparent, extensible, and accessible to systems researchers and software developers.

The project will prioritize correctness, architectural clarity, reproducibility, and compatibility over premature performance optimization.

---

# 2. Problem Statement

NVIDIA CUDA provides a mature ecosystem for GPU programming.

However, developing and distributing CUDA applications commonly involves dependencies on proprietary NVIDIA software components, including the CUDA compiler and parts of the CUDA software stack.

This creates several challenges:

1. Limited transparency into important components of the GPU software ecosystem.
2. Dependence on vendor-specific compilation and runtime infrastructure.
3. Difficulty independently experimenting with alternative GPU execution models.
4. Challenges in developing alternative GPU compiler and runtime implementations.
5. Limited opportunities to investigate compatibility at different software-stack boundaries.

GenGPU addresses these challenges by independently implementing selected components of the GPU programming ecosystem.

The project will reuse open compiler infrastructure where appropriate and interact with NVIDIA hardware through documented interfaces.

---

# 3. Product Goals

## 3.1 Primary Goals

### G1 — Independent Compilation Pipeline

Develop an independently orchestrated compilation pipeline capable of translating supported CUDA C++ programs into NVIDIA PTX.

The pipeline must not require NVIDIA `nvcc` for its own compilation process.

### G2 — Open GPU Runtime

Implement a Rust-based runtime capable of:

- GPU discovery.
- Driver initialization.
- Context management.
- GPU memory allocation.
- Host-device memory transfers.
- PTX module loading.
- Kernel execution.
- Synchronization.
- Error handling.

### G3 — CUDA Compatibility

Implement a progressively expanding subset of CUDA Runtime API semantics.

Compatibility will be defined by explicit support matrices and automated tests.

### G4 — Compiler and Runtime Independence

Maintain clear boundaries between:

- Source-language frontend.
- Compiler infrastructure.
- GPU code generation.
- Runtime execution.
- Hardware driver interface.

### G5 — Research Infrastructure

Provide reproducible tests, benchmarks, technical documentation, and architecture specifications.

## 3.2 Secondary Goals

- Python integration.
- GPU profiling interfaces.
- MLIR-based optimization.
- GPU mathematical libraries.
- Support for additional GPU vendors.
- Alternative GPU programming abstractions.

---

# 4. Non-Goals

The initial versions of GenGPU will NOT attempt to:

1. Replace NVIDIA's proprietary GPU drivers.
2. Implement an independent NVIDIA hardware driver.
3. Support every CUDA C++ language feature.
4. Guarantee binary compatibility with arbitrary CUDA applications.
5. Reimplement the entire CUDA ecosystem immediately.
6. Reverse-engineer undocumented NVIDIA hardware interfaces.
7. Compete with NVIDIA in raw GPU performance from the beginning.
8. Implement a complete machine-learning framework.
9. Support every NVIDIA GPU architecture simultaneously.

These restrictions are intentional.

The project must establish a correct, testable execution pipeline before expanding its compatibility surface.

---

# 5. Target Users

## 5.1 Systems Researchers

Researchers investigating:

- GPU execution models.
- Compiler architecture.
- Runtime systems.
- Heterogeneous computing.
- Parallel programming.
- GPU memory management.

## 5.2 Compiler Developers

Developers interested in:

- LLVM.
- Clang.
- MLIR.
- PTX generation.
- GPU-specific compiler optimization.

## 5.3 GPU Software Developers

Developers who want to experiment with an alternative GPU compilation and runtime environment.

## 5.4 Academic Institutions

Universities and research groups studying GPU programming systems and compiler design.

---

# 6. Product Architecture

GenGPU will use a modular architecture.

## 6.1 High-Level Architecture

```text
                 GPU APPLICATIONS
                        |
                        v
              CUDA C++ SOURCE CODE
                        |
                        v
                GEN GPU COMPILER
                        |
             +----------+----------+
             |                     |
             v                     v
        Clang Frontend        Compiler Driver
             |                     |
             +----------+----------+
                        |
                        v
                 LLVM IR / MLIR
                        |
                        v
                LLVM OPTIMIZATION
                        |
                        v
                  NVPTX BACKEND
                        |
                        v
                    PTX CODE
                        |
                        v
                GEN GPU RUNTIME
                        |
             +----------+----------+
             |                     |
             v                     v
        Memory Manager       Execution Manager
             |                     |
             +----------+----------+
                        |
                        v
              NVIDIA DRIVER API
                        |
                        v
                   NVIDIA GPU
```

## 6.2 Architectural Components

### A. Compiler

Language: C++

Responsibilities:

- CUDA C++ frontend integration.
- Source parsing.
- Semantic validation.
- Intermediate representation generation.
- Compiler optimization.
- GPU-specific lowering.
- PTX generation.
- Compilation diagnostics.

Primary technologies:

- Clang.
- LLVM.
- LLVM NVPTX backend.
- MLIR (introduced incrementally).

### B. Runtime

Language: Rust.

Responsibilities:

- GPU device discovery.
- Driver initialization.
- Context lifecycle.
- Memory management.
- Module loading.
- Kernel launching.
- Execution synchronization.
- Resource management.
- Runtime error handling.

### C. Foreign Function Interface

The compiler and runtime will communicate through a stable C ABI.

Responsibilities:

- Compiler-runtime communication.
- Opaque resource handles.
- Error propagation.
- Explicit memory ownership.
- ABI versioning.

Rust-specific types must not cross the public C ABI boundary.

### D. CUDA Compatibility Layer

The compatibility layer will expose a supported subset of CUDA Runtime API functionality.

Its purpose is to translate supported CUDA operations into GenGPU runtime operations.

The compatibility layer must preserve documented semantics wherever an API is declared supported.

### E. Hardware Driver Interface

The initial backend will use NVIDIA's documented CUDA Driver API.

Initial responsibilities:

- Driver initialization.
- Device discovery.
- Context creation.
- Memory operations.
- PTX module loading.
- Kernel launching.
- Synchronization.

GenGPU will not initially implement its own NVIDIA hardware driver.

---

# 7. Technology Stack

| Component | Technology |
|---|---|
| Runtime | Rust |
| Compiler | C++ |
| Compiler frontend | Clang |
| Compiler infrastructure | LLVM |
| Intermediate representation | LLVM IR |
| Advanced compiler transformations | MLIR |
| GPU intermediate representation | PTX |
| Hardware interface | NVIDIA CUDA Driver API |
| FFI | C ABI |
| Build system | Cargo + CMake |
| Build orchestration | Ninja |
| Testing | Rust tests + GoogleTest |
| Documentation | Markdown |
| CI/CD | GitHub Actions |
| Version control | Git |

## 7.1 Technology Selection Principles

Every major dependency must have an explicit architectural justification.

GenGPU should not independently recreate mature infrastructure without a compelling research or engineering reason.

LLVM and MLIR will provide compiler infrastructure.

GenGPU will focus its independent implementation effort on GPU runtime behavior, compatibility semantics, compilation orchestration, and research contributions.

---

# 8. Functional Requirements

Functional requirements define what GenGPU must do.

## 8.1 Driver Management

**FR-001:** GenGPU must load the NVIDIA Driver API dynamically.

**FR-002:** GenGPU must initialize the driver.

**FR-003:** GenGPU must detect available NVIDIA GPUs.

**FR-004:** GenGPU must retrieve device properties.

**FR-005:** GenGPU must report driver initialization errors.

**FR-006:** GenGPU must handle environments where the NVIDIA driver is unavailable.

## 8.2 Device Management

**FR-007:** GenGPU must expose device enumeration.

**FR-008:** GenGPU must provide device metadata.

**FR-009:** GenGPU must support device selection.

**FR-010:** GenGPU must maintain device and context lifecycle information.

## 8.3 Memory Management

**FR-011:** GenGPU must support GPU device-memory allocation.

**FR-012:** GenGPU must support GPU device-memory deallocation.

**FR-013:** GenGPU must support host-to-device transfers.

**FR-014:** GenGPU must support device-to-host transfers.

**FR-015:** GenGPU must validate memory-operation parameters.

**FR-016:** GenGPU must prevent invalid resource ownership and use-after-free through its runtime abstractions.

## 8.4 Module Management

**FR-017:** GenGPU must load PTX modules.

**FR-018:** GenGPU must resolve kernel entry points.

**FR-019:** GenGPU must report module-loading errors.

**FR-020:** GenGPU must release module resources correctly.

## 8.5 Kernel Execution

**FR-021:** GenGPU must support kernel execution.

**FR-022:** GenGPU must support grid and block execution configuration.

**FR-023:** GenGPU must support kernel argument passing.

**FR-024:** GenGPU must support device synchronization.

**FR-025:** GenGPU must validate kernel execution parameters.

## 8.6 Compiler

**FR-026:** GenGPU must accept supported GPU kernel source programs.

**FR-027:** GenGPU must produce LLVM IR.

**FR-028:** GenGPU must support NVIDIA PTX generation.

**FR-029:** GenGPU must report compilation errors.

**FR-030:** GenGPU must not require `nvcc` for its own supported compilation pipeline.

---

# 9. CUDA Compatibility Strategy

Compatibility will be incremental.

GenGPU will maintain a documented compatibility matrix.

## 9.1 Initial Compatibility Scope

| Feature | Priority | Initial status |
|---|---|---|
| Device discovery | P0 | Planned |
| Driver initialization | P0 | Planned |
| Context management | P0 | Planned |
| Device memory allocation | P0 | Planned |
| Memory transfers | P0 | Planned |
| PTX module loading | P0 | Planned |
| Kernel launching | P0 | Planned |
| Basic arithmetic kernels | P0 | Planned |
| Thread indexing | P1 | Planned |
| Block indexing | P1 | Planned |
| Grid synchronization | P2 | Research |
| Shared memory | P2 | Research |
| Atomic operations | P2 | Research |
| Streams and events | P2 | Research |
| Unified memory | P3 | Future |
| CUDA Graphs | P3 | Future |
| cuBLAS compatibility | P3 | Future |

P0, P1, P2, and P3 represent implementation priority, not completion status.

## 9.2 Compatibility Levels

GenGPU will distinguish three types of compatibility.

### Source Compatibility

Supported CUDA source programs can be compiled using GenGPU's compiler.

### Runtime Compatibility

Supported CUDA Runtime API operations have equivalent behavior through GenGPU.

### Binary Compatibility

Existing CUDA binaries can execute without recompilation.

Binary compatibility is explicitly outside the initial release scope.

No feature will be described as compatible without corresponding tests.

---

# 10. Non-Functional Requirements

## 10.1 Correctness

Correctness takes priority over performance.

All supported operations must have deterministic and reproducible correctness tests wherever the underlying execution model permits.

## 10.2 Reliability

The runtime must:

- Handle driver errors.
- Manage GPU resources safely.
- Avoid invalid memory access.
- Handle resource exhaustion.
- Avoid leaking contexts and modules.
- Provide meaningful error diagnostics.

## 10.3 Maintainability

The architecture must remain modular.

Compiler, runtime, FFI, and compatibility code must have clearly defined boundaries.

## 10.4 Portability

The initial supported development environment will be:

- Ubuntu Linux.
- Ubuntu through WSL2.
- NVIDIA GPUs with a compatible driver.

Native Linux support is the primary deployment target.

WSL2 is an important development and testing environment.

## 10.5 Performance

Performance optimization will begin after correctness and API semantics are established.

The project will benchmark:

- Compilation latency.
- Runtime initialization overhead.
- Kernel-launch overhead.
- Memory-transfer throughput.
- Kernel execution time.
- Resource-management overhead.

## 10.6 Observability

GenGPU should provide:

- Structured error messages.
- Diagnostic logging.
- Compilation diagnostics.
- Runtime execution diagnostics.
- Benchmark metadata.

---

# 11. Repository Structure

The intended repository structure is:

```text
gengpu/
│
├── README.md
├── PRODUCT.md
├── LICENSE
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
│
├── Cargo.toml
├── CMakeLists.txt
├── rust-toolchain.toml
│
├── compiler/
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── gengpu/
│   ├── src/
│   │   ├── frontend/
│   │   ├── ir/
│   │   ├── optimization/
│   │   ├── lowering/
│   │   └── codegen/
│   └── tests/
│
├── runtime/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── device.rs
│       ├── context.rs
│       ├── memory.rs
│       ├── transfer.rs
│       ├── module.rs
│       ├── kernel.rs
│       ├── launch.rs
│       ├── stream.rs
│       ├── driver/
│       └── error.rs
│
├── ffi/
│   ├── include/
│   └── src/
│
├── cuda-compat/
│   ├── include/
│   └── src/
│
├── tools/
│   └── gengpu/
│
├── examples/
│   ├── vector_add/
│   └── matrix_multiplication/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── compatibility/
│   └── benchmarks/
│
├── docs/
│   ├── architecture/
│   ├── specifications/
│   ├── adr/
│   └── research/
│
└── scripts/
    ├── build.sh
    └── test.sh
```

This structure represents the intended architecture. Directories will be introduced when their corresponding implementation milestones begin.

---

# 12. Development Roadmap

The project will follow milestone-driven development.

## M0 — Project Foundation

Estimated duration: 1–2 weeks.

Deliverables:

- Repository initialization.
- Cargo workspace.
- CMake integration.
- Rust and C++ build configuration.
- Initial architecture specification.
- Initial architecture decision records.
- CI pipeline.

Acceptance criteria:

- Clean checkout builds.
- Rust tests execute.
- C++ compiler configuration works.
- Build instructions are documented.

## M1 — NVIDIA Driver Integration

Estimated duration: 2–4 weeks.

Deliverables:

- Dynamic driver loading.
- Driver initialization.
- GPU enumeration.
- Device-property retrieval.
- Driver error handling.

Acceptance criteria:

- NVIDIA RTX 3050 detected.
- Device metadata retrieved.
- Missing-driver conditions handled.
- Automated tests pass.

## M2 — GPU Runtime Foundation

Estimated duration: 4–6 weeks.

Deliverables:

- Context management.
- Device-memory allocation.
- Memory transfers.
- Resource ownership.
- Runtime error handling.

Acceptance criteria:

- Device memory can be allocated and released.
- Data can be transferred correctly.
- Resource cleanup is verified.

## M3 — PTX Execution

Estimated duration: 4–8 weeks.

Deliverables:

- PTX module loading.
- Kernel-function resolution.
- Kernel argument management.
- Kernel launching.
- Synchronization.

Acceptance criteria:

- A valid PTX kernel executes on NVIDIA hardware.
- Numerical output matches a CPU reference.
- Invalid PTX and launch configurations produce meaningful errors.

## M4 — Independent Compiler Pipeline

Estimated duration: 6–10 weeks.

Deliverables:

- LLVM integration.
- NVPTX backend configuration.
- LLVM IR generation.
- PTX generation.
- Compiler diagnostics.
- Compiler-runtime integration.

Acceptance criteria:

- GenGPU generates PTX without invoking `nvcc`.
- Generated PTX executes through the GenGPU runtime.

## M5 — CUDA C++ Compatibility Subset

Estimated duration: 2–3 months.

Deliverables:

- Clang CUDA frontend integration.
- Kernel declaration support.
- Thread and block indexing.
- Basic arithmetic.
- Pointer-based memory operations.
- Selected CUDA Runtime API functions.

Acceptance criteria:

- A documented CUDA C++ subset compiles and executes.
- Compatibility tests pass.
- Unsupported language constructs fail with useful diagnostics.

## M6 — Advanced Runtime and Language Support

Estimated duration: 3–6 months.

Deliverables:

- Streams.
- Events.
- Shared memory.
- Atomic operations.
- Device functions.
- Expanded synchronization support.

Acceptance criteria:

- Advanced supported operations pass correctness and compatibility tests.

## M7 — Research and Optimization

Ongoing.

Deliverables:

- MLIR integration.
- Compiler optimization passes.
- Runtime optimization.
- Benchmark suite.
- Compatibility analysis.
- Research publications.

Acceptance criteria:

- Reproducible experiments.
- Documented performance comparisons.
- Clearly identified research contributions.

These are provisional estimates. Milestone completion will depend on test results and actual hardware behavior.

---

# 13. First Release Strategy

## 13.1 GenGPU 0.1 — Driver Probe

The first development milestone is GEN-001.

Scope:

- NVIDIA driver loading.
- Driver initialization.
- GPU discovery.
- Device metadata retrieval.
- Error handling.

Target hardware:

NVIDIA GeForce RTX 3050.

Target development environment:

Windows with WSL2 Ubuntu.

Primary implementation language:

Rust.

Acceptance criteria:

1. NVIDIA Driver API loads successfully.
2. Driver initialization succeeds.
3. RTX 3050 is detected.
4. Device name is retrieved.
5. Compute capability is retrieved.
6. Missing driver errors are handled.
7. Rust formatting and lint checks pass.
8. The implementation is committed to Git.

## 13.2 Subsequent Releases

| Version | Objective |
|---|---|
| 0.1 | Driver discovery |
| 0.2 | Context management |
| 0.3 | Memory management |
| 0.4 | PTX execution |
| 0.5 | LLVM compiler integration |
| 0.6 | Basic CUDA source compilation |
| 0.7 | Expanded runtime compatibility |
| 0.8 | Advanced execution features |
| 0.9 | Research evaluation |
| 1.0 | Stable documented compatibility subset |

Release versions are targets, not promises of full CUDA compatibility.

---

# 14. Testing Strategy

Testing will be performed at multiple levels.

## 14.1 Unit Tests

Test individual runtime and compiler components.

Examples:

- Error conversion.
- Device metadata handling.
- Memory parameter validation.
- Runtime state transitions.
- Compiler IR transformations.

## 14.2 Integration Tests

Test complete operations across components.

Examples:

- Driver initialization.
- Context creation.
- GPU memory allocation.
- PTX loading.
- Kernel execution.

## 14.3 Compatibility Tests

Test supported CUDA source programs and API operations against their documented expected semantics.

## 14.4 Numerical Correctness Tests

Use CPU reference implementations to validate GPU results.

Initial workloads:

- Vector addition.
- Vector multiplication.
- Element-wise operations.
- Reductions.
- Matrix multiplication.

## 14.5 Regression Tests

Every fixed bug that can be reproduced should receive a regression test.

## 14.6 Hardware Testing

CI should separate:

- CPU-only tests.
- Mock-driver tests.
- NVIDIA GPU integration tests.

GPU-dependent tests must run on appropriately configured NVIDIA hardware.

---

# 15. Benchmarking and Evaluation

GenGPU will evaluate performance using reproducible experiments.

## 15.1 Initial Benchmarks

| Benchmark | Purpose |
|---|---|
| Vector addition | Basic kernel correctness |
| Vector multiplication | Element-wise throughput |
| Memory copy | Transfer performance |
| Reduction | Parallel execution |
| Matrix multiplication | Compute performance |
| Kernel launch | Runtime overhead |

## 15.2 Reference Implementations

Comparisons may include:

- CPU reference implementations.
- CUDA Runtime API implementations.
- NVIDIA `nvcc` compiled kernels.
- Alternative compiler-generated PTX.

The reference implementation and hardware configuration must be recorded.

## 15.3 Required Benchmark Metadata

Every benchmark report must identify:

- GPU model.
- Compute capability.
- Driver version.
- CUDA Toolkit version where applicable.
- LLVM version.
- GenGPU commit.
- Compiler flags.
- Input size.
- Number of iterations.
- Warm-up policy.
- Measurement methodology.

No performance claim should be published without reproducible evidence.

---

# 16. Security and Reliability

GenGPU interacts with GPU hardware and native driver interfaces.

The runtime must treat external binaries, PTX artifacts, kernel arguments, and driver errors as potentially unsafe inputs.

Security requirements:

1. Validate memory sizes and execution parameters.
2. Avoid unchecked integer arithmetic in allocation calculations.
3. Minimize unsafe Rust.
4. Document all unsafe FFI boundaries.
5. Validate resource ownership and lifetime.
6. Avoid arbitrary dynamic library loading from untrusted paths.
7. Handle malformed compiler inputs safely.
8. Avoid assuming that successful compilation guarantees safe kernel execution.

GenGPU does not promise to sandbox arbitrary GPU programs in its initial releases.

---

# 17. Open-Source Governance

## 17.1 License

Proposed license: Apache License 2.0.

Before release, all dependencies and bundled components must undergo license review.

LLVM and Clang components retain their applicable upstream licensing terms.

NVIDIA SDK headers, libraries, documentation, and redistributable components must be used according to their respective licenses and redistribution terms.

## 17.2 Contribution Rules

Contributions should:

- Follow the project's coding conventions.
- Include appropriate tests.
- Include documentation for new public APIs.
- Preserve established architectural boundaries.
- Explain performance-related changes.
- Include benchmarks for material optimization claims.

## 17.3 Architecture Decision Records

Significant technical decisions must be documented in ADRs.

Initial ADRs:

- ADR-001: Rust and C++ language boundary.
- ADR-002: LLVM as compiler infrastructure.
- ADR-003: NVIDIA Driver API as initial hardware interface.
- ADR-004: PTX as initial GPU intermediate representation.
- ADR-005: C ABI between compiler and runtime.

---

# 18. Project Success Metrics

Success will be measured through engineering evidence rather than marketing claims.

## Short-Term Success

- Reliable NVIDIA driver initialization.
- GPU enumeration.
- Context management.
- GPU memory operations.
- Successful PTX execution.

## Medium-Term Success

- Independent PTX generation.
- CUDA C++ subset compilation.
- Runtime API compatibility.
- Automated hardware integration testing.

## Long-Term Success

- Broad documented compatibility.
- Reproducible performance results.
- Extensible compiler architecture.
- Research contributions.
- External open-source contributors.
- Support for additional hardware backends where technically feasible.

---

# 19. Known Technical Risks

| Risk | Impact | Mitigation |
|---|---|---|
| NVIDIA driver dependency | High | Isolate driver interface |
| LLVM integration complexity | High | Incremental integration |
| CUDA language complexity | Very high | Explicit compatibility subset |
| PTX compatibility changes | Medium | Versioned backend testing |
| Runtime semantic differences | High | Reference tests |
| GPU resource management | High | Explicit ownership model |
| WSL2 limitations | Medium | Native Linux testing |
| Insufficient hardware testing | High | Dedicated GPU CI |
| Scope expansion | Very high | Milestone gates |

---

# 20. Product Principles

GenGPU development will follow these principles:

1. Correctness before performance.
2. Explicit compatibility over vague claims.
3. Independent implementation over proprietary dependency.
4. Reuse mature open-source infrastructure where appropriate.
5. Clear architectural boundaries.
6. Reproducible research.
7. Incremental engineering.
8. Transparent limitations.
9. Test-driven runtime development.
10. Long-term maintainability.

---

# 21. Immediate Execution Plan

The immediate development sequence is:

| Task | Description | Dependency |
|---|---|---|
| GEN-001 | NVIDIA driver initialization and discovery | WSL2 GPU access |
| GEN-002 | Context management | GEN-001 |
| GEN-003 | GPU memory allocation | GEN-002 |
| GEN-004 | Host-device transfers | GEN-003 |
| GEN-005 | PTX module loading | GEN-002 |
| GEN-006 | Kernel execution | GEN-004, GEN-005 |
| GEN-007 | First end-to-end GPU kernel | GEN-006 |
| GEN-008 | LLVM NVPTX integration | Compiler foundation |

The first end-to-end target is a vector-add kernel.

```cpp
__global__ void vector_add(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        C[i] = A[i] + B[i];
    }
}
```

The goal is to compile a supported implementation of this kernel through GenGPU's own compiler pipeline and execute it through GenGPU's own runtime.

NVIDIA's driver will provide the hardware interface.

NVIDIA's `nvcc` must not be part of the GenGPU compilation path.

---

# 22. Final Product Definition

GenGPU is an open-source GPU computing research platform focused on independently implementing a CUDA-compatible compiler and runtime ecosystem.

It will initially target NVIDIA GPUs using LLVM-based PTX generation and the documented NVIDIA Driver API.

The project will progressively implement CUDA language and runtime compatibility, beginning with basic GPU execution and expanding toward advanced parallel programming capabilities.

Its success will be determined by:

- Correctness.
- Compatibility.
- Reproducibility.
- Architectural quality.
- Research value.
- Community adoption.

**The first objective is simple: independently initialize, discover, and execute work on an NVIDIA GPU.**

Everything else builds on that foundation.

---

**End of Product Specification**

*GenGPU — Open GPU Computing Infrastructure.*
