# ADR-0002: Synchronous Host–Device Memory Transfers

- **Status:** Proposed
- **Date:** 2026-11-16
- **Decision owners:** GenGPU maintainers
- **Related ADRs:** [ADR-0001: Context Ownership and Threading](0001-context-ownership-and-threading.md)

## Context

GenGPU can initialize the NVIDIA CUDA Driver API, enumerate devices, create
contexts, allocate device memory, release allocations, and synchronize a
context. The runtime cannot yet move application data between host memory and
a GenGPU device allocation.

The product specification requires host-to-device and device-to-host
transfers (FR-013 and FR-014). The next runtime milestone should add these
operations without introducing streams, asynchronous execution, pinned host
memory, or device-to-device copies.

A transfer must use the context that owns the allocation. Device identity
alone is insufficient: two contexts may refer to the same physical device,
but an allocation must not be used through an unrelated context.

The current DeviceAllocation already retains an opaque shared owner token,
its DeviceId, device address, and allocation size. This provides the basis
for validating allocation ownership without exposing native CUDA handles.

## Decision drivers

- Preserve backend-neutral public interfaces.
- Prevent out-of-bounds device-memory access.
- Reject allocations owned by another context, including another context on
  the same physical device.
- Keep host buffers valid for the complete duration of a transfer.
- Prevent deallocation while a transfer is in progress.
- Preserve the calling thread's previous CUDA current-context state.
- Make transfer and context-restoration failures observable.
- Establish completion before returning from the synchronous API.
- Keep asynchronous execution and additional memory classes out of scope.

## Options considered

### Context methods with borrowed buffers

Add host-to-device and device-to-host methods to the backend-neutral Context
trait. Borrow host slices and the DeviceAllocation for the duration of the
call. This makes the lifetime and exclusive-use requirements visible in
Rust's type system and keeps the operation associated with its owning
context. This option is selected.

### Free-standing transfer functions

Free-standing functions would need an explicit context argument and would
provide no stronger ownership guarantee. They make it easier to accidentally
pair an allocation with an unrelated context. This option is rejected.

### Asynchronous transfer API

An asynchronous API would require stream semantics, host-buffer lifetime
management beyond the call, and completion tracking. Those concerns are
outside this milestone and are deferred.

## Decision

### Public API

Add the following methods to the backend-neutral Context trait:

```rust
fn copy_to_device(
    &self,
    destination: &mut DeviceAllocation,
    offset: usize,
    source: &[u8],
) -> Result<()>;

fn copy_from_device(
    &self,
    source: &mut DeviceAllocation,
    offset: usize,
    destination: &mut [u8],
) -> Result<()>;
```

Both methods have default implementations returning
ErrorKind::Unsupported, consistent with Context::allocate. The API transfers
raw bytes; typed serialization, element layouts, and alignment policy are
the responsibility of higher-level APIs.

### Ownership and borrowing

- The Context used for the operation must be the exact context that created
  the DeviceAllocation. Matching DeviceId values are not sufficient.
- The backend validates this using the existing opaque allocation owner
  token and context identity. No CUDA handle or owner token is exposed
  publicly.
- The DeviceAllocation is mutably borrowed for the entire operation. This
  prevents deallocation or another operation from borrowing that allocation
  through safe Rust while the transfer is active.
- Host-to-device source data is borrowed as &[u8].
- Device-to-host destination data is borrowed as &mut [u8].
- Host pointers are passed to the native call only while the corresponding
  Rust slice borrow remains active.
- No host memory is pinned or registered by GenGPU in this milestone. The
  caller must provide a host memory region accepted by the CUDA Driver API.

### Bounds and arithmetic

For a transfer of length N at allocation offset O:

- Compute O + N using checked arithmetic. Overflow returns
  ErrorKind::InvalidArgument.
- Require O + N <= allocation.size_bytes().
- Require O to be representable in the device-address arithmetic and require
  base_device_address + O to use checked addition.
- An out-of-range transfer returns ErrorKind::InvalidArgument before any
  CUDA copy call.
- A zero-length transfer is a no-op after context ownership, context-open,
  and offset/range validation. It does not invoke CUDA copy or synchronization
  functions. An offset equal to allocation size is valid for a zero-length
  transfer; an offset greater than allocation size is invalid.

### NVIDIA implementation and completion semantics

The NVIDIA backend loads and calls:

- cuMemcpyHtoD_v2 for host-to-device copies.
- cuMemcpyDtoH_v2 for device-to-host copies.

The public operations are synchronous: they must not return success until the
transfer is complete. After a successful native copy, the backend calls
cuCtxSynchronize before restoring the previous current-context state. This
establishes completion even where the native copy's host-blocking behavior
depends on the host-memory type.

This initial completion policy synchronizes the entire CUDA context. It may
wait for other work in that context and may surface errors from previously
submitted asynchronous work. This cost is accepted while GenGPU has no
stream-aware transfer API. A later stream/event design may replace this
policy through a separately reviewed decision.

The NVIDIA backend pushes the owning context before the operation and pops
it afterward, verifying that the popped handle is the expected context. It
must preserve the existing context-stack mismatch policy from ADR-0001.
Ordinary pop failures may use the established safe retry behavior; a detected
stack mismatch must not trigger an unsafe additional pop.

### Error semantics

- Context/ownership mismatch returns ErrorKind::InvalidArgument.
- A closed context returns ErrorKind::InvalidState.
- Bounds or arithmetic failures return ErrorKind::InvalidArgument.
- CUDA copy and synchronization failures are translated to ErrorKind::Driver.
- If the copy fails, the copy error remains primary; context-restoration
  failure is appended to its diagnostic.
- If the copy succeeds but synchronization fails, the synchronization error
  is returned, with any restoration failure appended.
- If copy and synchronization succeed but context restoration fails, the
  restoration error is returned.
- A failed transfer does not guarantee that the destination range is
  unchanged or that no bytes were transferred. Callers must not blindly retry
  a failed operation when duplicate or partial effects matter.
- GenGPU does not automatically retry a copy operation.
- The operation does not promise transactional or atomic transfer semantics.

### NVIDIA Driver API boundary

The CUDA function pointers and native device addresses remain private to the
NVIDIA backend. The public API uses only GenGPU-owned Rust types.

## Consequences

### Positive

- Host/device data movement becomes available through the existing Context
  abstraction.
- Rust borrowing protects the host buffers and prevents safe deallocation of
  the allocation during a transfer.
- Exact context identity is checked rather than relying only on device ID.
- Bounds checks happen before native memory operations.
- The synchronous API has an explicit completion point.
- CUDA-specific handles and function pointers remain encapsulated.

### Negative and trade-offs

- Context-wide synchronization can serialize unrelated work and increase
  latency. This is accepted for the initial synchronous API.
- Callers must use byte slices; typed views and serialization are not
  provided.
- Pinned host memory and mixed registered/pageable host regions are not
  managed by GenGPU.
- A transfer error does not imply that the target range is unchanged.
- The initial API does not provide asynchronous overlap or stream selection.

### Follow-up work

- PH1-006B: add the backend-neutral Context methods and allocation-owner
  identity helper without implementing native copies.
- PH1-006C: load and implement NVIDIA copy functions and context activation.
- PH1-006D: add mock tests, error-path coverage, and physical GPU round-trip
  validation.
- Keep device-to-device copies, streams, events, pinned memory, unified
  memory, and asynchronous transfers outside PH1-006.

## References

- NVIDIA, CUDA Driver API, Memory Management:
  https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MEM.html
- [ADR-0001: Context Ownership and Threading](0001-context-ownership-and-threading.md)
