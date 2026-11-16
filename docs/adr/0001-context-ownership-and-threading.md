# ADR-0001: Context Ownership and Threading

## Status

Accepted

## Context

GenGPU's driver abstraction initially covered driver initialization and
device enumeration. The runtime now needs an explicit execution-context
contract before backend-specific context operations, memory management, or
kernel execution can be implemented.

CUDA contexts have thread-current behavior. A context handle's ownership,
the thread's current-context stack, and the lifetime of the dynamically
loaded driver library must therefore be treated as separate concerns.

## Decision drivers

- Keep native CUDA handles and error codes out of the backend-neutral API.
- Make context ownership and destruction explicit.
- Preserve the driver library and function table while dependent contexts
  remain alive.
- Model current-context state as thread-local, not process-global.
- Prevent concurrent access through one context object in the initial API.
- Make cleanup failures observable where explicit cleanup is requested.
- Keep the initial abstraction small enough to support future backends.

## Options considered

### Implicit or global context

This would simplify initial call sites but obscure ownership and make
thread-current behavior difficult to reason about. It is rejected.

### Shared, concurrently accessible context object

This would permit concurrent use through shared references, but would require
additional synchronization and more complex semantics before GenGPU has
streams or asynchronous execution. It is deferred.

### Explicit owned context with scoped current-context guard

This makes context lifetime explicit, models thread-current state through a
scoped guard, and permits the backend to retain its own native resources.
This option is selected.

## Decision

- Context creation is explicit and separate from device enumeration.
- Each context belongs to one logical GenGPU DeviceId.
- A context owns its native resource and retains the backend state required
  for its lifetime. The backend implementation will use shared ownership
  where needed so dropping the Driver value cannot invalidate a live context.
- The backend-neutral Context trait is Send but not Sync.
- CurrentContextGuard<'a> is neither Send nor Sync, is bound to its creating
  thread, and holds an exclusive mutable borrow of the associated Context for
  its lifetime. This makes closing or mutably reusing that context while the
  guard is alive a compile-time error.
- Making a context current is scoped. Finishing or dropping the guard restores
  the previously current context. The NVIDIA backend uses CUDA push/pop calls
  and verifies the popped native handle before considering cleanup successful.
- If ordinary explicit guard cleanup fails, its cleanup state remains
  available for a best-effort retry during Drop. A detected context-stack
  mismatch is terminal for that guard: after attempting to restore the
  unexpected stack entry, it must not pop again because that could remove an
  unrelated context or repeat a known mismatch.
- Synchronization temporarily activates the context as needed and restores
  the previous current-context state before returning. If synchronization and
  restoration both fail, the synchronization failure remains the primary
  diagnostic and the restoration failure is appended.
- Explicit close reports destruction errors and retains enough state for a
  retry. Dropping an open context performs best-effort cleanup and must not
  panic.
- Native backend handles and error codes never appear in the public contract.

## Consequences

Positive:

- Resource ownership is visible in the type/API model.
- Current-context behavior is scoped and does not rely on global mutable
  state.
- The driver library can outlive the Driver value when contexts still depend
  on it.
- The API can be tested with backend-independent test doubles.

Costs and limitations:

- The initial abstraction intentionally disallows concurrent shared access
  through one context object.
- Drop cannot report cleanup failures; callers requiring error visibility
  must use explicit close or guard finish.
- Backend implementations must carefully preserve thread-local context
  state on every success and failure path. The lifetime-bound guard prevents
  conflicting operations through the same Context value while it is active,
  but cannot prevent unrelated external CUDA calls from mutating that host
  thread's context stack. A detected stack mismatch is reported rather than
  attempting further pops that could corrupt unrelated context state.

Follow-up:

- Implement the backend-neutral Context and CurrentContextGuard contracts
  (PH1-004B).
- Implement NVIDIA context creation, activation, synchronization, and
  destruction (PH1-004C and PH1-004D).
