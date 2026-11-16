# GenGPU Driver Abstraction Boundary

## Status

Approved design; driver and context contracts introduced in PH1-002 and
PH1-004B. NVIDIA context creation and destruction implemented in PH1-004C.

## Purpose

The driver crate isolates backend-specific GPU APIs from the GenGPU runtime.
The runtime consumes the backend-neutral `gengpu_driver::Driver` and
`gengpu_driver::Context` interfaces and shared types from gengpu-core. It
must not depend on CUDA handles, CUDA error codes, or NVIDIA loader details.

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

- `Driver::initialize(&self) -> Result<()>`
- `Driver::enumerate_devices(&self) -> Result<Vec<DeviceInfo>>`
- `Driver::create_context(&self, DeviceId) -> Result<Box<dyn Context>>`
- `Context::device_id(&self) -> DeviceId`
- `Context::make_current(&mut self) -> Result<CurrentContextGuard<'_>>`
- `Context::synchronize(&mut self) -> Result<()>`
- `Context::close(&mut self) -> Result<()>`

The contract uses only gengpu-core types and GenGPU-owned abstractions.
CUDA-specific types are not permitted in its signatures.

## Context ownership and lifecycle

- Context creation is explicit and is not part of device enumeration.
- A context is an owned resource associated with one logical DeviceId.
- A context retains the backend resources required for its lifetime.
- Context is Send but not Sync: it may move between threads but cannot be
  shared concurrently through the public abstraction.
- A CurrentContextGuard is neither Send nor Sync and is bound to the thread
  on which it was created.
- NVIDIA context creation validates the logical device against the current
  device count, creates the CUDA context, and immediately pops the newly
  created context from the calling thread's current-context stack.
- The NVIDIA context owns the native handle and a shared reference to the
  loaded CUDA API, so it can outlive the NvidiaDriver value.
- Explicit close reports CUDA destruction errors and retains the handle on
  failure so the caller can retry. Drop performs best-effort destruction.
- A closed context rejects operations that require a live native context
  with ErrorKind::InvalidState.
- Current-context activation and synchronization are not implemented yet;
  they are the next context subphase, PH1-004D.

## Driver lifecycle and ownership

- A caller owns an explicit driver instance; GenGPU does not expose a global
  mutable driver singleton.
- The concrete backend owns its loader state and resolved function pointers.
- Initialization is idempotent for the same driver instance.
- Device enumeration and context creation are valid only after successful
  initialization.
- Enumeration or context creation before successful initialization returns
  ErrorKind::InvalidState.
- The Driver trait deliberately does not promise Send or Sync. Thread-safety
  requirements for concrete drivers are implementation-specific.
- A context must retain the backend state it needs independently of the
  lifetime of the Driver value that created it.

The trait uses shared references to permit implementations to manage
initialization state internally. Concrete implementations are responsible
for the synchronization needed to uphold the idempotence contract.

## Device identity

Enumeration returns gengpu_core::DeviceInfo. Each DeviceId is a
process-local logical identifier, not a persistent hardware identifier.
The backend assigns identifiers for the current enumeration; callers must
not persist them as hardware identity across process runs or device
enumeration changes.

## Error translation

The NVIDIA backend translates native failures into gengpu-core::Error
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

## Scope exclusions

This boundary does not implement:

- current-context push/pop operations for ordinary caller use;
- synchronization;
- device memory, transfers, streams, or events;
- PTX module loading or kernel launch;
- compiler integration.
