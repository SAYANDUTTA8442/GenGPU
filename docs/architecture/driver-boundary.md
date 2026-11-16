# GenGPU Driver Abstraction Boundary

## Status

Approved design; interface contract introduced in PH1-002.

## Purpose

The driver crate isolates backend-specific GPU APIs from the GenGPU runtime.
The runtime consumes the backend-neutral gengpu_driver::Driver interface
and shared types from gengpu-core. It must not depend on CUDA handles,
CUDA error codes, or NVIDIA loader details.

## Dependency direction

The intended dependency direction is:

- gengpu-runtime depends on the driver contract.
- gengpu-driver depends on gengpu-core.
- The NVIDIA backend calls the NVIDIA CUDA Driver API.
- The runtime does not call the CUDA Driver API directly.

## Alternatives considered

### Direct CUDA calls from the runtime

This has the lowest initial abstraction cost, but couples runtime APIs to
NVIDIA-specific handles, errors, and lifecycle behavior. It also makes
backend substitution and isolated testing harder. This option is rejected.

### Backend-neutral Rust trait

A Rust trait provides an explicit boundary while keeping the runtime and
driver implementation in Rust. It permits test doubles and alternative
backends without exposing native handles. The abstraction overhead is
acceptable for this control-plane interface. This option is selected.

### C ABI wrapper

A C ABI could provide a language-neutral boundary, but it would introduce
another FFI layer and additional ownership and error-marshalling work before
there is a second language implementation that needs it. It is not selected
for the initial runtime-driver boundary.

## Public contract

The driver crate exposes:

- Driver::initialize(&self) -> Result<()>
- Driver::enumerate_devices(&self) -> Result<Vec<DeviceInfo>>

The contract uses only gengpu-core types. CUDA-specific types are not
permitted in its signatures.

## Lifecycle and ownership

- A caller owns an explicit driver instance; GenGPU does not expose a global
  mutable driver singleton.
- The concrete backend owns its loader state and resolved function pointers.
- Initialization is idempotent for the same driver instance.
- Device enumeration is valid only after successful initialization.
- Enumeration before successful initialization returns
  ErrorKind::InvalidState.
- The trait deliberately does not promise Send or Sync. Thread-safety
  requirements will be decided when context and concurrency semantics are
  designed.
- Native resource and context lifetimes are outside this interface for now;
  they will be specified before context management is implemented.

The trait uses shared references to permit implementations to manage
initialization state internally. Concrete implementations are responsible for
the synchronization needed to uphold the idempotence contract.

## Device identity

Enumeration returns gengpu_core::DeviceInfo. Each DeviceId is a
process-local logical identifier, not a persistent hardware identifier.
The backend assigns identifiers for the current enumeration; callers must
not persist them as hardware identity across process runs or device
enumeration changes.

## Error translation

The NVIDIA backend translates native failures into gengpu_core::Error
at the backend boundary. Driver failures use ErrorKind::Driver; invalid
lifecycle calls use ErrorKind::InvalidState.

Until the shared error type has a demonstrated need for structured native
error fields, the human-readable error message must preserve the diagnostic
details available from the driver:

- the failed CUDA operation;
- the CUDA error name, when available;
- the native numeric error code;
- the driver-provided description, when available.

Native error types must not leak into the public driver trait.

## NVIDIA loading policy

The initial NVIDIA backend is planned to load the CUDA Driver API dynamically
from the installed driver library (on Linux, typically libcuda.so.1).
Building GenGPU should not require the CUDA toolkit or link against the CUDA
driver library at build time. Missing libraries and missing required symbols
must become controlled GenGPU errors.

The loader implementation, symbol resolution, and actual CUDA calls are
explicitly deferred to a later implementation subphase.

## Scope exclusions

This boundary does not implement:

- CUDA Driver API calls or bindings;
- dynamic library loading or symbol resolution;
- GPU detection or device enumeration against hardware;
- CUDA contexts or context ownership;
- device memory, transfers, streams, or events;
- PTX module loading or kernel launch;
- compiler integration.
