//! NVIDIA-owned execution context and device-memory resources.

use std::any::Any;
use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

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
    orphaned_allocations: Mutex<Vec<u64>>,
}

impl ContextInner {
    pub(super) fn new(api: Arc<dyn CudaApi>, handle: usize) -> Self {
        Self {
            api,
            handle: Mutex::new(Some(handle)),
            live_allocations: AtomicUsize::new(0),
            orphaned_allocations: Mutex::new(Vec::new()),
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

    fn register_orphaned_allocation(&self, address: u64, decrement_live_count: bool) {
        let mut orphaned = self
            .orphaned_allocations
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        orphaned.push(address);

        // Publish the orphan before making the live-allocation count reach
        // zero. Context::close checks the count before draining this registry.
        if decrement_live_count {
            self.live_allocations.fetch_sub(1, Ordering::AcqRel);
        }
    }

    fn cleanup_orphaned_allocations(&self, handle: usize) -> Result<()> {
        let pending = {
            let mut orphaned = self
                .orphaned_allocations
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            std::mem::take(&mut *orphaned)
        };

        let mut failed = Vec::new();
        let mut first_error: Option<Error> = None;

        for address in pending {
            match release_orphaned_allocation(&self.api, handle, address) {
                CleanupOutcome::Released(Ok(())) => {}
                CleanupOutcome::Released(Err(error)) => {
                    first_error = Some(match first_error.take() {
                        Some(previous) => combine_errors(
                            previous,
                            error,
                            "cleaning up another orphaned allocation",
                        ),
                        None => error,
                    });
                }
                CleanupOutcome::Retain(error) => {
                    failed.push(address);
                    first_error = Some(match first_error.take() {
                        Some(previous) => combine_errors(
                            previous,
                            error,
                            "cleaning up another orphaned allocation",
                        ),
                        None => error,
                    });
                }
            }
        }

        if !failed.is_empty() {
            self.orphaned_allocations
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .extend(failed);
        }

        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn close(&self) -> Result<()> {
        if self.live_allocations.load(Ordering::Acquire) != 0 {
            return Err(Error::new(
                ErrorKind::InvalidState,
                "cannot close an NVIDIA context while device allocations remain live",
            ));
        }

        let native_handle = {
            let handle = self.handle.lock().map_err(|_| {
                Error::new(
                    ErrorKind::Internal,
                    "NVIDIA context handle mutex is poisoned",
                )
            })?;
            match *handle {
                Some(native_handle) => native_handle,
                None => return Ok(()),
            }
        };
        self.cleanup_orphaned_allocations(native_handle)?;

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
        let native_handle = self.handle.get_mut().ok().and_then(|handle| *handle);

        if let Some(native_handle) = native_handle {
            // Every returned allocation retains this ContextInner. Any
            // allocation whose Drop cleanup failed is first transferred
            // to the orphan registry and no longer owns a separate Arc.
            // CUDA context destruction is the final reclamation boundary
            // if best-effort explicit frees still fail here.
            let _ = self.cleanup_orphaned_allocations(native_handle);
            let _ = self.api.destroy_context(native_handle);

            if let Ok(handle) = self.handle.get_mut() {
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
                let cleanup_error = match cleanup {
                    CleanupOutcome::Released(result) => result.err(),
                    CleanupOutcome::Retain(error) => {
                        // The allocation was never returned to the caller.
                        // Transfer ownership to the context for a later retry.
                        self.inner.register_orphaned_allocation(address, false);
                        Some(error)
                    }
                };

                return match (recovery, cleanup_error) {
                    (Ok(()), None) => Err(restore_error),
                    (Err(recovery_error), None) => Err(combine_errors(
                        restore_error,
                        recovery_error,
                        "retrying context restoration",
                    )),
                    (Ok(()), Some(cleanup_error)) => Err(combine_errors(
                        restore_error,
                        cleanup_error,
                        "cleaning up an allocation after context restoration failed",
                    )),
                    (Err(recovery_error), Some(cleanup_error)) => Err(combine_errors(
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
        let weak_inner: Weak<ContextInner> = Arc::downgrade(&inner);
        DeviceAllocation::new(
            self.device_id,
            DeviceAddress::from_raw(address),
            size_bytes,
            owner,
            move || release_allocation(&inner, address),
            move || {
                if let Some(inner) = weak_inner.upgrade() {
                    inner.register_orphaned_allocation(address, true);
                }
            },
        )
    }

    fn copy_to_device(
        &self,
        destination: &mut DeviceAllocation,
        offset: usize,
        source: &[u8],
    ) -> Result<()> {
        let handle = self.ensure_open()?;
        let address = validate_transfer(self, destination, offset, source.len())?;
        let Some(address) = address else {
            return Ok(());
        };

        let api = Arc::clone(&self.inner.api);
        run_synchronous_transfer(&api, handle, || {
            api.copy_to_device(address, source)
                .map_err(|error| cuda_error("cuMemcpyHtoD_v2", error))
        })
    }

    fn copy_from_device(
        &self,
        source: &mut DeviceAllocation,
        offset: usize,
        destination: &mut [u8],
    ) -> Result<()> {
        let handle = self.ensure_open()?;
        let address = validate_transfer(self, source, offset, destination.len())?;
        let Some(address) = address else {
            return Ok(());
        };

        let api = Arc::clone(&self.inner.api);
        run_synchronous_transfer(&api, handle, || {
            api.copy_from_device(address, destination)
                .map_err(|error| cuda_error("cuMemcpyDtoH_v2", error))
        })
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

fn validate_transfer(
    context: &NvidiaContext,
    allocation: &DeviceAllocation,
    offset: usize,
    length: usize,
) -> Result<Option<u64>> {
    if !allocation.is_owned_by(&context.inner) {
        return Err(Error::new(
            ErrorKind::InvalidArgument,
            "device allocation belongs to a different NVIDIA context",
        ));
    }
    if allocation.is_deallocated() {
        return Err(Error::new(
            ErrorKind::InvalidState,
            "cannot transfer data using a deallocated device allocation",
        ));
    }

    let end = offset.checked_add(length).ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidArgument,
            "device transfer offset and length overflow usize",
        )
    })?;
    if end > allocation.size_bytes() {
        return Err(Error::new(
            ErrorKind::InvalidArgument,
            format!(
                "device transfer range {offset}..{end} exceeds allocation size {}",
                allocation.size_bytes()
            ),
        ));
    }

    let offset = u64::try_from(offset).map_err(|_| {
        Error::new(
            ErrorKind::InvalidArgument,
            "device transfer offset cannot be represented as a device address",
        )
    })?;
    let address = allocation
        .device_address()
        .as_u64()
        .checked_add(offset)
        .ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidArgument,
                "device transfer address arithmetic overflow",
            )
        })?;

    if length == 0 {
        Ok(None)
    } else {
        Ok(Some(address))
    }
}

fn run_synchronous_transfer(
    api: &Arc<dyn CudaApi>,
    handle: usize,
    copy: impl FnOnce() -> Result<()>,
) -> Result<()> {
    api.push_current(handle)
        .map_err(|error| cuda_error("cuCtxPushCurrent_v2", error))?;

    let copy_result = copy();
    let synchronization_result = if copy_result.is_ok() {
        api.synchronize()
            .map_err(|error| cuda_error("cuCtxSynchronize", error))
    } else {
        Ok(())
    };
    let restoration_result = pop_expected_context_retry(api, handle);

    match (copy_result, synchronization_result, restoration_result) {
        (Ok(()), Ok(()), Ok(())) => Ok(()),
        (Err(copy_error), _, Ok(())) => Err(copy_error),
        (Err(copy_error), _, Err(restore_error)) => Err(combine_errors(
            copy_error,
            restore_error,
            "restoring the previous current context after a failed transfer",
        )),
        (Ok(()), Err(sync_error), Ok(())) => Err(sync_error),
        (Ok(()), Err(sync_error), Err(restore_error)) => Err(combine_errors(
            sync_error,
            restore_error,
            "restoring the previous current context after transfer synchronization failed",
        )),
        (Ok(()), Ok(()), Err(restore_error)) => Err(restore_error),
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
) -> CleanupOutcome {
    if let Err(error) = inner.api.push_current(handle) {
        return CleanupOutcome::Retain(cuda_error("cuCtxPushCurrent_v2", error));
    }

    let free_result = inner
        .api
        .free_memory(address)
        .map_err(|error| cuda_error("cuMemFree_v2", error));
    let restore_result = pop_expected_context_retry(&inner.api, handle);

    match (free_result, restore_result) {
        (Ok(()), Ok(())) => CleanupOutcome::Released(Ok(())),
        (Ok(()), Err(error)) => CleanupOutcome::Released(Err(error)),
        (Err(error), Ok(())) => CleanupOutcome::Retain(error),
        (Err(free_error), Err(restore_error)) => CleanupOutcome::Retain(combine_errors(
            free_error,
            restore_error,
            "restoring context after cleanup of an unreturned allocation",
        )),
    }
}

fn release_orphaned_allocation(
    api: &Arc<dyn CudaApi>,
    handle: usize,
    address: u64,
) -> CleanupOutcome {
    if let Err(error) = api.push_current(handle) {
        return CleanupOutcome::Retain(cuda_error("cuCtxPushCurrent_v2", error));
    }

    let free_result = api
        .free_memory(address)
        .map_err(|error| cuda_error("cuMemFree_v2", error));
    let restore_result = pop_expected_context_retry(api, handle);

    match (free_result, restore_result) {
        (Ok(()), Ok(())) => CleanupOutcome::Released(Ok(())),
        (Ok(()), Err(error)) => CleanupOutcome::Released(Err(error)),
        (Err(error), Ok(())) => CleanupOutcome::Retain(error),
        (Err(free_error), Err(restore_error)) => CleanupOutcome::Retain(combine_errors(
            free_error,
            restore_error,
            "restoring context after orphaned allocation cleanup",
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
