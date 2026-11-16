//! NVIDIA-owned execution context and device-memory resources.

use std::any::Any;
use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use gengpu_core::{DeviceId, Error, ErrorKind, Result};

use crate::allocation::CleanupOutcome;
use crate::{Context, CurrentContextGuard, DeviceAddress, DeviceAllocation};

use super::api::CudaApi;
use super::driver::cuda_error;

/// Shared native context state retained by every live allocation.
///
/// The native context is destroyed when the public context and all allocation
/// owners release this state. Explicit close is permitted only when no device
/// allocations remain live.
pub(super) struct ContextInner {
    api: Arc<dyn CudaApi>,
    handle: Mutex<Option<usize>>,
    live_allocations: AtomicUsize,
}

impl ContextInner {
    pub(super) fn new(api: Arc<dyn CudaApi>, handle: usize) -> Self {
        Self {
            api,
            handle: Mutex::new(Some(handle)),
            live_allocations: AtomicUsize::new(0),
        }
    }

    fn handle(&self) -> Result<usize> {
        self.handle
            .lock()
            .map_err(|_| {
                Error::new(
                    ErrorKind::Internal,
                    "NVIDIA context handle mutex is poisoned",
                )
            })?
            .ok_or_else(|| Error::new(ErrorKind::InvalidState, "NVIDIA context is already closed"))
    }

    fn close(&self) -> Result<()> {
        if self.live_allocations.load(Ordering::Acquire) != 0 {
            return Err(Error::new(
                ErrorKind::InvalidState,
                "cannot close an NVIDIA context while device allocations remain live",
            ));
        }

        let mut handle = self.handle.lock().map_err(|_| {
            Error::new(
                ErrorKind::Internal,
                "NVIDIA context handle mutex is poisoned",
            )
        })?;
        let Some(native_handle) = *handle else {
            return Ok(());
        };

        self.api
            .destroy_context(native_handle)
            .map_err(|error| cuda_error("cuCtxDestroy_v2", error))?;
        *handle = None;
        Ok(())
    }
}

impl Drop for ContextInner {
    fn drop(&mut self) {
        if let Ok(handle) = self.handle.get_mut() {
            if let Some(native_handle) = *handle {
                // Every live allocation owns an Arc<ContextInner>, so this
                // final drop cannot occur while a tracked allocation is live.
                let _ = self.api.destroy_context(native_handle);
                *handle = None;
            }
        }
    }
}

/// Public NVIDIA context wrapper.
///
/// Cell makes this type !Sync while preserving Send, matching the public
/// Context contract. Device allocations retain ContextInner independently.
pub(super) struct NvidiaContext {
    inner: Arc<ContextInner>,
    device_id: DeviceId,
    _not_sync: PhantomData<Cell<()>>,
}

impl NvidiaContext {
    pub(super) fn new(api: Arc<dyn CudaApi>, handle: usize, device_id: DeviceId) -> Self {
        Self {
            inner: Arc::new(ContextInner::new(api, handle)),
            device_id,
            _not_sync: PhantomData,
        }
    }

    fn ensure_open(&self) -> Result<usize> {
        self.inner.handle()
    }
}

impl Context for NvidiaContext {
    fn device_id(&self) -> DeviceId {
        self.device_id
    }

    fn make_current(&mut self) -> Result<CurrentContextGuard<'_>> {
        let handle = self.ensure_open()?;
        self.inner
            .api
            .push_current(handle)
            .map_err(|error| cuda_error("cuCtxPushCurrent_v2", error))?;

        let api = Arc::clone(&self.inner.api);
        let mut stack_mismatch_seen = false;
        Ok(CurrentContextGuard::new(self, move || {
            if stack_mismatch_seen {
                return Ok(());
            }

            let popped = api
                .pop_current()
                .map_err(|error| cuda_error("cuCtxPopCurrent_v2", error))?;

            if popped == handle {
                return Ok(());
            }

            stack_mismatch_seen = true;
            let mismatch = format!(
                "cuCtxPopCurrent_v2 returned context handle {popped:#x}, expected {handle:#x}"
            );

            if popped != 0 {
                if let Err(error) = api.push_current(popped) {
                    let restore_error = cuda_error("cuCtxPushCurrent_v2", error);
                    return Err(Error::new(
                        restore_error.kind(),
                        format!(
                            "{mismatch}; restoring the unexpected context also failed: {}",
                            restore_error.message()
                        ),
                    ));
                }
            }

            Err(Error::new(ErrorKind::InvalidState, mismatch))
        }))
    }

    fn allocate(&self, size_bytes: usize) -> Result<DeviceAllocation> {
        if size_bytes == 0 {
            return Err(Error::new(
                ErrorKind::InvalidArgument,
                "device allocation size must be greater than zero",
            ));
        }
        let handle = self.ensure_open()?;
        self.inner
            .api
            .push_current(handle)
            .map_err(|error| cuda_error("cuCtxPushCurrent_v2", error))?;

        let allocation_result = self
            .inner
            .api
            .allocate_memory(size_bytes)
            .map_err(|error| cuda_error("cuMemAlloc_v2", error));

        let restoration = pop_expected_context(&self.inner.api, handle);
        let address = match (allocation_result, restoration) {
            (Ok(address), Ok(())) => address,
            (Err(error), Ok(())) => return Err(error),
            (Err(error), Err(restore_error)) => {
                return Err(combine_errors(
                    error,
                    restore_error,
                    "restoring context after allocation",
                ));
            }
            (Ok(address), Err(restore_error)) => {
                // If the first pop failed without changing the stack, retry
                // it before pushing the context again for allocation cleanup.
                let recovery = if restore_error.kind() == ErrorKind::Driver {
                    pop_expected_context(&self.inner.api, handle)
                } else {
                    Err(restore_error.clone())
                };
                let cleanup = cleanup_unreturned_allocation(&self.inner, handle, address);
                return match (recovery, cleanup) {
                    (Ok(()), Ok(())) => Err(restore_error),
                    (Err(recovery_error), Ok(())) => Err(combine_errors(
                        restore_error,
                        recovery_error,
                        "retrying context restoration",
                    )),
                    (Ok(()), Err(cleanup_error)) => Err(combine_errors(
                        restore_error,
                        cleanup_error,
                        "cleaning up an allocation after context restoration failed",
                    )),
                    (Err(recovery_error), Err(cleanup_error)) => Err(combine_errors(
                        combine_errors(
                            restore_error,
                            recovery_error,
                            "retrying context restoration",
                        ),
                        cleanup_error,
                        "cleaning up an allocation after context restoration failed",
                    )),
                };
            }
        };

        self.inner.live_allocations.fetch_add(1, Ordering::AcqRel);
        let inner = Arc::clone(&self.inner);
        let owner: Arc<dyn Any + Send + Sync> = inner.clone();
        DeviceAllocation::new(
            self.device_id,
            DeviceAddress::from_raw(address),
            size_bytes,
            owner,
            move || release_allocation(&inner, address),
        )
    }

    fn synchronize(&mut self) -> Result<()> {
        let api = Arc::clone(&self.inner.api);
        let guard = self.make_current()?;
        let synchronization = api
            .synchronize()
            .map_err(|error| cuda_error("cuCtxSynchronize", error));
        let restoration = guard.finish();

        match (synchronization, restoration) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), Ok(())) => Err(error),
            (Ok(()), Err(error)) => Err(error),
            (Err(sync_error), Err(restore_error)) => Err(combine_errors(
                sync_error,
                restore_error,
                "restoring the previous current context",
            )),
        }
    }

    fn close(&mut self) -> Result<()> {
        self.inner.close()
    }
}

fn pop_expected_context(api: &Arc<dyn CudaApi>, expected: usize) -> Result<()> {
    let popped = api
        .pop_current()
        .map_err(|error| cuda_error("cuCtxPopCurrent_v2", error))?;
    if popped == expected {
        return Ok(());
    }

    let mismatch = Error::new(
        ErrorKind::InvalidState,
        format!("cuCtxPopCurrent_v2 returned context handle {popped:#x}, expected {expected:#x}"),
    );
    if popped != 0 {
        api.push_current(popped).map_err(|error| {
            combine_errors(
                mismatch.clone(),
                cuda_error("cuCtxPushCurrent_v2", error),
                "restoring the unexpected context",
            )
        })?;
    }
    Err(mismatch)
}

fn pop_expected_context_retry(api: &Arc<dyn CudaApi>, expected: usize) -> Result<()> {
    match pop_expected_context(api, expected) {
        Ok(()) => Ok(()),
        Err(first_error) if first_error.kind() == ErrorKind::Driver => {
            match pop_expected_context(api, expected) {
                Ok(()) => Err(first_error),
                Err(second_error) => Err(combine_errors(
                    first_error,
                    second_error,
                    "retrying context restoration",
                )),
            }
        }
        Err(error) => Err(error),
    }
}

fn cleanup_unreturned_allocation(
    inner: &Arc<ContextInner>,
    handle: usize,
    address: u64,
) -> Result<()> {
    inner
        .api
        .push_current(handle)
        .map_err(|error| cuda_error("cuCtxPushCurrent_v2", error))?;
    let free_result = inner
        .api
        .free_memory(address)
        .map_err(|error| cuda_error("cuMemFree_v2", error));
    let restore_result = pop_expected_context_retry(&inner.api, handle);

    match (free_result, restore_result) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) => Err(error),
        (Ok(()), Err(error)) => Err(error),
        (Err(free_error), Err(restore_error)) => Err(combine_errors(
            free_error,
            restore_error,
            "restoring context after cleanup of an unreturned allocation",
        )),
    }
}

fn release_allocation(inner: &Arc<ContextInner>, address: u64) -> CleanupOutcome {
    let handle = match inner.handle() {
        Ok(handle) => handle,
        Err(error) => return CleanupOutcome::Retain(error),
    };

    if let Err(error) = inner.api.push_current(handle) {
        return CleanupOutcome::Retain(cuda_error("cuCtxPushCurrent_v2", error));
    }

    let free_result = inner
        .api
        .free_memory(address)
        .map_err(|error| cuda_error("cuMemFree_v2", error));
    let restore_result = pop_expected_context_retry(&inner.api, handle);

    match (free_result, restore_result) {
        (Ok(()), Ok(())) => {
            inner.live_allocations.fetch_sub(1, Ordering::AcqRel);
            CleanupOutcome::Released(Ok(()))
        }
        (Ok(()), Err(restore_error)) => {
            // The device allocation is gone even though the caller's context
            // stack could not be restored. Disarm cleanup to prevent double-free.
            inner.live_allocations.fetch_sub(1, Ordering::AcqRel);
            CleanupOutcome::Released(Err(restore_error))
        }
        (Err(free_error), Ok(())) => CleanupOutcome::Retain(free_error),
        (Err(free_error), Err(restore_error)) => CleanupOutcome::Retain(combine_errors(
            free_error,
            restore_error,
            "restoring the previous current context after a failed free",
        )),
    }
}

fn combine_errors(primary: Error, secondary: Error, operation: &str) -> Error {
    Error::new(
        primary.kind(),
        format!(
            "{}; {operation} also failed: {}",
            primary.message(),
            secondary.message()
        ),
    )
}
