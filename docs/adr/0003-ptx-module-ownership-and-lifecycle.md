# ADR-0003: PTX Module Ownership and Lifecycle

- **Status:** Accepted
- **Date:** 2026-11-16
- **Decision owners:** GenGPU maintainers
- **Related ADRs:** [ADR-0001: Context Ownership and Threading](0001-context-ownership-and-threading.md), [ADR-0002: Synchronous Host–Device Memory Transfers](0002-synchronous-host-device-transfers.md)

## Context

GenGPU has implemented NVIDIA driver initialization, device enumeration, CUDA context management, device-memory allocation, and synchronous host-device transfers. The runtime does not yet load PTX modules or resolve kernel entry points.

The product specification requires PTX module loading, kernel entry-point resolution, module resource release, kernel execution, and argument passing (FR-017 through FR-025). This decision establishes the module ownership and lifecycle boundary before implementation. Kernel launch and argument ABI are deliberately handled by a separate design milestone.

A CUDA function handle is associated with its loaded module. The module must remain loaded for the function handle to remain valid. Module and function lifetimes therefore need an explicit relationship in the GenGPU API.

## Decision drivers

- Preserve backend-neutral runtime interfaces.
- Keep CUDA module and function handles private to the NVIDIA backend.
- Ensure a module cannot outlive its owning context.
- Ensure a resolved function cannot outlive its module.
- Make unload failures observable and safely retryable.
- Preserve the calling thread's previous CUDA current-context state.
- Keep PTX loading independent of compiler integration.
- Avoid committing to kernel argument or launch semantics before their dedicated design review.

## Options considered

### PTX text through cuModuleLoadData

Load an in-memory PTX text buffer through the CUDA Driver API. This provides the minimal initial loading path and avoids introducing file-loading policy or JIT option configuration into the first module milestone. This option is selected.

### PTX text through cuModuleLoadDataEx

This API supports JIT options and diagnostic buffers. It provides more control over JIT compilation diagnostics but introduces additional configuration and implementation complexity. It is deferred. A later implementation may add richer diagnostics without changing the backend-neutral module ownership contract.

### File-based module loading

A file-based API would introduce path handling, filesystem policy, and file-lifetime questions. It is deferred; the initial runtime accepts PTX text supplied by the caller.

## Decision

### Initial loading interface

The initial NVIDIA backend uses:

- `cuModuleLoadData` to load a null-terminated PTX text buffer.
- `cuModuleUnload` to unload the module.
- `cuModuleGetFunction` to resolve a named kernel entry point.

The initial public runtime accepts PTX text only. It rejects empty input and embedded NUL bytes before making a native call. The backend creates a null-terminated buffer for the native loading operation.

Cubin, fatbin, file-based loading, JIT option configuration, and configurable JIT diagnostic buffers are outside this initial scope.

### Backend-neutral API contract

The Context abstraction gains a PTX loading operation conceptually equivalent to:

```rust
fn load_ptx(&self, ptx: &str) -> Result<Box<dyn Module>>;
```

The backend-neutral Module abstraction provides:

```rust
pub trait Module: Send {
    fn get_function<'module>(
        &'module self,
        name: &str,
    ) -> Result<KernelFunction<'module>>;

    fn unload(&mut self) -> Result<()>;
}
```

These signatures describe the ownership contract. Concrete type details may be refined during the backend-neutral API implementation, without weakening the lifetime and ownership guarantees in this ADR.

KernelFunction is an opaque GenGPU-owned handle. Its lifetime is tied to a borrow of its parent Module. It does not expose a CUDA-native function handle, and it cannot outlive the module that produced it.

Function lookup rejects an empty name or an embedded NUL byte. A name that cannot be resolved returns ErrorKind::NotFound. CUDA module-loading and function-resolution failures are translated at the NVIDIA backend boundary.

### Ownership and lifetime

- A Module belongs to exactly one Context.
- The Module retains shared ownership of the context/backend state needed for its lifetime.
- A Module cannot be used through a different Context, including another Context for the same physical device.
- A KernelFunction borrows its parent Module. Safe Rust borrowing prevents explicit unload or mutable module operations while a borrowed function handle remains live.
- The native CUmodule and CUfunction handles remain private to the NVIDIA backend.
- Context closure is rejected while caller-owned modules remain live.
- The public Context wrapper may be dropped while a Module remains live; shared context state keeps the native context and driver library alive.

### Context activation

Module loading, function resolution, and unloading activate the owning CUDA context using the existing push/pop context-stack policy established by ADR-0001. Each operation restores the calling thread's previous current-context state before returning.

A context-stack mismatch must follow the safe recovery policy already established in ADR-0001. The backend must not issue an unsafe additional pop after detecting a mismatch.

### Unload and cleanup semantics

- Explicit Module::unload is fallible.
- On successful unload, the module becomes unloaded and subsequent operations return ErrorKind::InvalidState.
- If the native module remains live after an unload failure, cleanup state is retained so the caller can retry.
- If the native module is successfully unloaded but restoring the previous context fails, the module is marked unloaded and the restoration error is reported. Unload must not be repeated.
- Module Drop performs best-effort cleanup and never panics.
- If best-effort unload fails while the native module remains live, cleanup responsibility transfers to the owning context's orphan registry.
- The live-module count is decremented only after the native module is released or its cleanup responsibility has been safely transferred to the context.
- Explicit context close is rejected while caller-owned modules remain live. Once caller-owned modules are gone, context close retries orphaned module unloads before destroying the native context.
- Final shared-context destruction is the final reclamation boundary for modules whose individual best-effort unload still fails.

### Error mapping

- Empty PTX or embedded NUL in PTX: ErrorKind::InvalidArgument.
- Empty function name or embedded NUL in function name: ErrorKind::InvalidArgument.
- Operation on a closed context or unloaded module: ErrorKind::InvalidState.
- Function name not found: ErrorKind::NotFound.
- CUDA module loading or function resolution failure: ErrorKind::Driver.
- Context restoration failure: ErrorKind::Driver, preserving the primary operation error when both fail.

### Explicit exclusions

This ADR does not define or implement:

- Kernel launch.
- Kernel argument packing, size, alignment, or pointer ABI.
- Grid or block dimensions.
- Streams or events.
- Compiler integration or PTX generation.
- Cubin or fatbin loading.
- File-based module loading.
- JIT option configuration or diagnostic buffers.
- Performance optimization.

Kernel launch and argument ABI require a separate design review in PH1-007B.

## Consequences

### Positive

- Module and function lifetimes are explicit in the Rust API.
- Context ownership and module ownership are enforced rather than left to caller convention.
- CUDA-native handles remain encapsulated.
- The initial PTX loading path stays small and testable.
- Module cleanup follows the resource-management principles already established for contexts and allocations.

### Negative and trade-offs

- The initial API accepts only PTX text.
- The initial loading path does not expose JIT option configuration or detailed JIT diagnostic buffers.
- Context-wide synchronization is not introduced as a module-loading requirement; module operations follow the documented CUDA Driver API semantics and restore context state.
- Kernel launch remains unavailable until a later milestone.

## Follow-up work

- PH1-007B: design kernel-function use, kernel argument ABI, launch configuration, and launch semantics.
- PH1-007C: implement NVIDIA PTX module loading, function resolution, ownership, and cleanup.
- PH1-007D: implement kernel launch, error handling, automated tests, and hardware validation.

## References

- NVIDIA, CUDA Driver API, Module Management:
  https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html
- NVIDIA, CUDA Driver API, Context Management:
  https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html
- [ADR-0001: Context Ownership and Threading](0001-context-ownership-and-threading.md)
- [ADR-0002: Synchronous Host–Device Memory Transfers](0002-synchronous-host-device-transfers.md)
