# GenGPU Driver Abstraction Boundary

## Status

Approved design; driver and context contracts introduced in PH1-002 and
PH1-004B. NVIDIA context creation and destruction implemented in PH1-004C.
NVIDIA scoped context activation and synchronization implemented in PH1-004D.
Backend-neutral device allocation ownership abstractions introduced in PH1-005B.
NVIDIA device allocation and deallocation implemented in PH1-005C. The
backend-neutral transfer contract was introduced in PH1-006B and
synchronous NVIDIA transfers are implemented in PH1-006C. Backend-neutral
PTX module contracts were introduced in PH1-007C1. NVIDIA PTX loading and
function resolution are implemented in PH1-007C2. Context-owned orphan cleanup
and live-module close restrictions are implemented in PH1-007C3; physical
hardware validation remains pending.

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
- `Context::allocate(&self, usize) -> Result<DeviceAllocation>`
- `Context::copy_to_device(&self, &mut DeviceAllocation, usize, &[u8]) -> Result<()>`
- `Context::copy_from_device(&self, &mut DeviceAllocation, usize, &mut [u8]) -> Result<()>`
- `Context::synchronize(&mut self) -> Result<()>`
- `Context::close(&mut self) -> Result<()>`

The contract uses only gengpu-core types and GenGPU-owned abstractions.
CUDA-specific types are not permitted in its signatures.

- `Context::load_ptx(&str) -> Result<Box<dyn Module>>`
- `Module::get_function(&str) -> Result<KernelFunction<'_>>`
- `Module::unload() -> Result<()>`

The backend-neutral module contract is introduced in PH1-007C1. Empty PTX
source and embedded NUL bytes are rejected before backend dispatch. Backends
without PTX loading support return ErrorKind::Unsupported for otherwise valid
PTX. A KernelFunction carries a borrow lifetime tied to its parent Module and
is neither Send nor Sync. Its backend-specific handle is type-erased; only
backend code that knows the concrete handle type can retrieve it. Function
lookup rejects empty names and embedded NUL bytes, and reports NotFound when
the requested function does not exist. The NVIDIA backend loads PTX through `cuModuleLoadData`, resolves functions
through `cuModuleGetFunction`, and explicitly unloads modules through
`cuModuleUnload`. Each operation activates the owning context and restores
the previous current-context stack. A failed function lookup maps CUDA's
`CUDA_ERROR_NOT_FOUND` result to ErrorKind::NotFound. Robust orphan tracking
for failed module cleanup and context-close restrictions are completed in
PH1-007C3.

## Context ownership and lifecycle

- Context creation is explicit and is not part of device enumeration.
- A context is an owned resource associated with one logical DeviceId.
- A context retains the backend resources required for its lifetime.
- Context is Send but not Sync: it may move between threads but cannot be
  shared concurrently through the public abstraction.
- A CurrentContextGuard is neither Send nor Sync and is bound to the thread
  on which it was created.
- CurrentContextGuard<'a> holds an exclusive mutable borrow of its context
  for the guard's lifetime. Rust therefore prevents closing or mutably
  operating on that context until the guard is finished or dropped.
- NVIDIA context creation validates the logical device against the current
  device count, creates the CUDA context, and immediately pops the newly
  created context from the calling thread's current-context stack.
- The NVIDIA context owns the native handle and a shared reference to the
  loaded CUDA API, so it can outlive the NvidiaDriver value.
- Explicit close reports CUDA destruction errors and retains the handle on
  failure so the caller can retry. It is rejected while caller-owned device
  allocations or PTX modules remain live. Drop performs best-effort cleanup
  of context-owned orphans before best-effort destruction.
- A closed context rejects operations that require a live native context
  with ErrorKind::InvalidState.
- NVIDIA make_current pushes the context onto the calling thread's CUDA
  context stack and returns a scoped guard. Guard cleanup pops the stack and
  verifies that the popped native handle is the expected context.
- If guard cleanup encounters an unexpected context, the backend attempts
  to push that context back before reporting an InvalidState error. It does
  not issue another pop during Drop after a stack mismatch, because doing so
  could remove an unrelated context or repeat a known mismatch.
- Explicit guard cleanup is fallible. On ordinary pop failure, Drop makes one
  additional best-effort cleanup attempt; cleanup failures are never allowed
  to panic.
- NVIDIA synchronize temporarily pushes the target context, calls
  cuCtxSynchronize, and attempts to pop the target context on both success
  and failure paths. If synchronization and restoration both fail, the
  synchronization error remains primary and restoration failure is included
  in its diagnostic message.
- Current-context state is managed through CUDA's thread-local stack. GenGPU
  does not maintain a process-global current-context variable.

## PTX module contract

- `Context::load_ptx` validates non-empty PTX text without embedded NUL
  bytes, activates the owning context, and calls `cuModuleLoadData`.
- `NvidiaModule` retains shared `ContextInner` ownership, keeping the native
  context and loaded CUDA driver API alive while the module exists.
- Function lookup validates non-empty names without embedded NUL bytes,
  activates the owning context, and calls `cuModuleGetFunction`.
- CUDA's `CUDA_ERROR_NOT_FOUND` result is translated to
  ErrorKind::NotFound. Other native lookup and load failures are translated
  to ErrorKind::Driver with CUDA operation diagnostics.
- Module loading and function lookup restore the caller's previous
  current-context stack. If loading succeeds but restoration fails, the
  module is not returned; its handle is immediately registered as a
  context-owned orphan for later cleanup.
- Each successfully returned module increments the context's live-module
  count. Explicit context close is rejected while that count is nonzero.
- Explicit unload decrements the count only after native unload succeeds.
  If native unload succeeds but context restoration fails, the module is
  marked released and is never unloaded twice.
- If module Drop cannot unload the native module, it transfers the handle to
  the context-owned orphan registry before decrementing the live-module count.
- Explicit context close retries orphaned module unloads before orphaned
  allocation frees. Failed unloads remain registered for a later close
  attempt. If unload succeeds but context restoration fails, the orphan is
  removed (to prevent a second unload) and the restoration error is returned;
  a later close can destroy the context.
- Final ContextInner drop also makes a best-effort retry of orphaned module
  unloads before destroying the native context.
- A KernelFunction stores an opaque backend handle and carries a Rust borrow
  tied to its Module. The function handle therefore cannot outlive the module,
  and the module cannot be mutably unloaded while a function borrow is live.
- Explicit module unload is fallible and retryable when the native unload
  fails. If unload succeeds but context restoration fails, the module is
  marked unloaded and is never unloaded twice.
- Module Drop performs best-effort unload. PH1-007C3 adds context-owned
  orphan tracking and prevents explicit context closure while caller-owned
  modules remain live; until that subphase, context destruction is the final
  reclamation boundary for a module whose Drop cleanup fails.

## Device allocation contract

The backend-neutral API exposes an opaque `DeviceAddress` and an owned
`DeviceAllocation`:

- DeviceAddress is a numeric device-space address, not a host pointer. Host
  code must never dereference it.
- DeviceAllocation records its DeviceId, opaque address, and byte size.
- DeviceAllocation retains shared NVIDIA context state so the native context
  and driver library outlive the public Context wrapper when necessary.
- Explicit deallocation activates the owning context, calls `cuMemFree_v2`,
  and restores the previous current-context stack.
- A failed explicit free retains cleanup state on DeviceAllocation so the
  caller can retry. If the free succeeds but restoring the prior context fails,
  the allocation is marked released and the restoration error is still
  reported; the free is never repeated.
- If DeviceAllocation is dropped while its best-effort free still fails,
  cleanup responsibility transfers to the owning context's orphan registry.
  The live-allocation count is decremented only after the orphan address has
  been recorded.
- Explicit context close is rejected while caller-owned allocations remain
  live. Once that count reaches zero, close retries every orphaned free before
  destroying the native context. Failed orphan frees remain registered for a
  later close attempt.
- If the public Context wrapper is dropped while allocations remain live,
  shared context state keeps the native context alive until allocation cleanup
  completes. Final shared-state drop retries orphaned frees and then destroys
  the CUDA context; CUDA context destruction is the final reclamation boundary
  for resources whose individual best-effort free still fails.
- If allocation succeeds but context restoration and cleanup both fail before
  the allocation can be returned, the address is registered as an orphan so
  context close or final context destruction still owns its reclamation.
- Drop performs best-effort cleanup and never panics.
- Zero-byte allocations are rejected with ErrorKind::InvalidArgument.
- Context::allocate defaults to ErrorKind::Unsupported for backends that do
  not implement device memory allocation.

## Host-device transfer contract

PH1-006B introduced synchronous transfer method signatures on `Context`;
backends without transfer support return ErrorKind::Unsupported. PH1-006C
implements these methods in the NVIDIA backend using `cuMemcpyHtoD_v2` and
`cuMemcpyDtoH_v2`.

- `copy_to_device` borrows the source host bytes and mutably borrows the
  destination allocation for the full call.
- `copy_from_device` mutably borrows the source allocation and destination
  host bytes for the full call.
- Offsets and lengths are expressed in bytes.
- The allocation-owner identity helper compares the opaque shared owner
  token, not just DeviceId, so separate contexts on the same device remain
  isolated.
- The NVIDIA backend rejects deallocated allocations and validates the
  transfer range and device-address arithmetic before making a native call.
  Zero-length in-range transfers are no-ops after ownership and lifecycle
  validation.
- Non-empty transfers activate the owning context, issue the native copy,
  synchronize the context after a successful copy, and restore the previous
  current context. Copy or synchronization errors remain primary if
  restoration also fails.
- Mock-based behavior is covered by unit tests; physical GPU validation is
  reserved for PH1-006D.

Device-to-device copies, streams, events, pinned host memory, unified memory,
and asynchronous transfers remain outside PH1-006.

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

- native host-device or device-device memory transfers (the backend-neutral
  host-device method signatures are present);
- streams or events;
- kernel launch;
- compiler integration.
