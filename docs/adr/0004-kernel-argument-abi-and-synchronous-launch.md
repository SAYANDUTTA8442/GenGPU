# ADR-0004: Kernel Argument ABI and Synchronous Launch Semantics

- **Status:** Accepted
- **Date:** 2026-11-16
- **Decision owners:** GenGPU maintainers
- **Related ADRs:** [ADR-0001: Context Ownership and Threading](0001-context-ownership-and-threading.md), [ADR-0002: Synchronous Host–Device Memory Transfers](0002-synchronous-host-device-transfers.md), [ADR-0003: PTX Module Ownership and Lifecycle](0003-ptx-module-ownership-and-lifecycle.md)

## Context

GenGPU has established the design for PTX module ownership and kernel-function lookup. The runtime still needs a defined kernel argument representation, launch configuration, and execution completion contract.

The CUDA Driver API launch interface accepts either an array of pointers to argument storage (`kernelParams`) or a packed extra-parameter buffer (`extra`). The initial GenGPU implementation should choose one mechanism and make its storage, ownership, and ABI assumptions explicit.

GenGPU does not initially have compiler-generated kernel signature metadata. Therefore, the runtime cannot independently prove that caller-supplied argument types and order match an arbitrary PTX kernel's parameter declarations.

## Decision drivers

- Keep the public launch API backend-neutral.
- Preserve argument storage for the entire native launch and synchronous completion operation.
- Prevent safe deallocation of device allocations while a launch is using them.
- Enforce exact context and module ownership.
- Make scalar representation and device-pointer semantics explicit.
- Validate launch dimensions before making a native call.
- Avoid implementing a packed parameter ABI before it is required.
- Keep asynchronous execution, streams, and events outside the initial launch milestone.

## Options considered

### CUDA kernelParams argument array

Use one pointer per kernel parameter, with each pointer referencing storage containing that argument's value. The CUDA driver obtains parameter sizes and offsets from the loaded kernel image. GenGPU retains all argument storage while constructing and using the pointer array. This option is selected.

### CUDA extra packed parameter buffer

Construct one packed argument buffer and provide explicit offsets and sizes through the extra launch parameters. This provides direct control over packing but requires GenGPU to implement and validate the packing layout. It is deferred.

## Decision

### Native launch mechanism

The NVIDIA backend uses `cuLaunchKernel` with:

- The resolved CUDA function handle.
- Explicit grid and block dimensions.
- Dynamic shared-memory size in bytes.
- The default stream.
- A `kernelParams` array.
- No `extra` packed argument buffer.

### Initial argument categories

The initial safe argument API supports:

1. Explicitly supported scalar values.
2. Device pointers derived from live GenGPU `DeviceAllocation` values.

The initial scalar set is:

- `u8`, `i8`
- `u16`, `i16`
- `u32`, `i32`
- `u64`, `i64`
- `f32`, `f64`

The scalar argument trait is sealed so external implementations cannot silently introduce unsupported representations.

A device-pointer argument is derived from a `DeviceAllocation`; callers do not pass arbitrary host pointers or arbitrary numeric device addresses through the initial safe API.

Arbitrary Rust structs, references, slices, implicit serialization, and host pointers are not supported as initial kernel arguments.

### Argument storage and lifetime

- The runtime constructs the native `kernelParams` pointer array for the launch.
- Every argument's backing storage remains valid and at a stable address for the complete launch operation, including the synchronous completion point.
- Scalar storage contains the scalar value passed by value.
- Device-pointer storage contains the device address value passed by value.
- Device allocations used as arguments are mutably borrowed or otherwise exclusively retained for the duration required to prevent safe deallocation while the launch is in progress.
- Argument order and representation must match the PTX kernel's parameter declarations.

The absence of compiler-generated signature metadata means GenGPU cannot independently verify that the supplied argument count, order, and types match an arbitrary PTX kernel. This correspondence remains an explicit caller responsibility. The runtime must not claim automatic signature validation.

### Launch configuration

The backend-neutral API defines:

- `Dim3`: three unsigned dimensions `x`, `y`, and `z`.
- `LaunchConfig`: grid dimensions, block dimensions, and dynamic shared-memory byte count.

Validation before a native launch includes:

- Every grid dimension must be non-zero.
- Every block dimension must be non-zero.
- Block-thread multiplication must use checked arithmetic.
- Dynamic shared-memory size is expressed in bytes.

Architecture- and device-specific launch limits remain subject to validation by the CUDA driver. GenGPU may add device-property-based validation later, but must not claim limits it has not established.

### Ownership and context activation

- A KernelFunction is tied to its parent Module as specified by ADR-0003.
- The Module and KernelFunction must belong to the exact Context used for launch.
- DeviceAllocation arguments must belong to that same exact Context; matching DeviceId values alone are insufficient.
- The launch operation activates the owning CUDA context and restores the caller's previous current-context state according to ADR-0001.
- A context-stack mismatch follows the safe recovery policy established in ADR-0001.

### Completion semantics

The initial launch API is synchronous.

After a successful `cuLaunchKernel` submission, the backend calls `cuCtxSynchronize` before returning success. The launch uses the default stream.

This provides a clear completion point and keeps argument and allocation lifetimes bounded by the call. It may wait for unrelated work in the same context and may surface earlier asynchronous errors. This trade-off is accepted for the initial implementation. Streams, events, and asynchronous launch semantics are deferred.

A launch or synchronization failure is not automatically retried. A failed operation does not guarantee that no GPU work was submitted or executed.

If launch or synchronization fails and context restoration also fails, the primary operation error remains primary and restoration failure is included in its diagnostic. If launch and synchronization succeed but restoration fails, the restoration error is returned.

### Error mapping

- Zero grid or block dimension, or arithmetic overflow: ErrorKind::InvalidArgument.
- Context, module, or allocation ownership mismatch: ErrorKind::InvalidArgument.
- Closed context or unloaded module: ErrorKind::InvalidState.
- CUDA launch or synchronization failure: ErrorKind::Driver.
- Context restoration failure: ErrorKind::Driver, preserving any primary launch or synchronization error.

## Consequences

### Positive

- The initial implementation uses CUDA's parameter-pointer mechanism without implementing a packed parameter buffer.
- Scalar and device-pointer argument categories are explicit.
- Argument storage and device allocation lifetimes are bounded by the synchronous call.
- Launch dimensions have a backend-neutral representation and pre-launch validation.
- Context and module ownership are checked explicitly.
- The caller's responsibility for matching PTX parameter declarations is documented honestly.

### Negative and trade-offs

- The runtime cannot validate argument signatures without compiler-generated metadata.
- Arbitrary structs, host pointers, and implicit serialization are unsupported.
- Context-wide synchronization can reduce concurrency and surface earlier asynchronous errors.
- The initial API does not support streams, events, or asynchronous execution.

## Explicit exclusions

This ADR does not define or implement:

- Compiler-generated kernel signature metadata or automatic signature reflection.
- Packed `extra` argument buffers.
- Arbitrary struct serialization.
- Host pointers as kernel arguments.
- User-selected streams, events, or asynchronous launch.
- Cooperative launch or CUDA Graphs.
- Compiler integration or PTX generation.
- Performance optimization.

## Follow-up work

- PH1-007C: implement NVIDIA PTX module loading, function resolution, ownership, and cleanup.
- PH1-007D: implement kernel launch, argument handling, error handling, automated tests, and hardware validation.

## References

- NVIDIA, CUDA Driver API, Execution Control:
  https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html
- [ADR-0001: Context Ownership and Threading](0001-context-ownership-and-threading.md)
- [ADR-0002: Synchronous Host–Device Memory Transfers](0002-synchronous-host-device-transfers.md)
- [ADR-0003: PTX Module Ownership and Lifecycle](0003-ptx-module-ownership-and-lifecycle.md)
