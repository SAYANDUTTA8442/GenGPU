//! NVIDIA-owned execution context resources.

use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::Arc;

use gengpu_core::{DeviceId, Error, ErrorKind, Result};

use crate::{Context, CurrentContextGuard};

use super::api::CudaApi;
use super::driver::cuda_error;

/// NVIDIA context retaining the CUDA API/library for its full lifetime.
///
/// Cell makes this type !Sync while preserving Send, matching the public
/// Context contract. The native CUDA handle never crosses the driver crate.
pub(super) struct NvidiaContext {
    api: Arc<dyn CudaApi>,
    handle: Option<usize>,
    device_id: DeviceId,
    _not_sync: PhantomData<Cell<()>>,
}

impl NvidiaContext {
    pub(super) fn new(api: Arc<dyn CudaApi>, handle: usize, device_id: DeviceId) -> Self {
        Self {
            api,
            handle: Some(handle),
            device_id,
            _not_sync: PhantomData,
        }
    }

    fn ensure_open(&self) -> Result<usize> {
        self.handle
            .ok_or_else(|| Error::new(ErrorKind::InvalidState, "NVIDIA context is already closed"))
    }
}

impl Context for NvidiaContext {
    fn device_id(&self) -> DeviceId {
        self.device_id
    }

    fn make_current(&mut self) -> Result<CurrentContextGuard<'_>> {
        let handle = self.ensure_open()?;
        self.api
            .push_current(handle)
            .map_err(|error| cuda_error("cuCtxPushCurrent_v2", error))?;

        let api = Arc::clone(&self.api);
        Ok(CurrentContextGuard::new(self, move || {
            let popped = api
                .pop_current()
                .map_err(|error| cuda_error("cuCtxPopCurrent_v2", error))?;

            if popped == handle {
                return Ok(());
            }

            let mismatch = format!(
                "cuCtxPopCurrent_v2 returned context handle {popped:#x}, expected {handle:#x}"
            );

            // A different context was on top of the thread's stack. Restore
            // it before reporting the invariant violation.
            if popped != 0 {
                if let Err(error) = api.push_current(popped) {
                    let restore_error = cuda_error("cuCtxPushCurrent_v2", error);
                    return Err(Error::new(
                        restore_error.kind(),
                        format!("{mismatch}; restoring the unexpected context also failed: {}", restore_error.message()),
                    ));
                }
            }

            Err(Error::new(ErrorKind::InvalidState, mismatch))
        }))
    }

    fn synchronize(&mut self) -> Result<()> {
        let guard = self.make_current()?;
        let synchronization = self
            .api
            .synchronize()
            .map_err(|error| cuda_error("cuCtxSynchronize", error));
        let restoration = guard.finish();

        match (synchronization, restoration) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), Ok(())) => Err(error),
            (Ok(()), Err(error)) => Err(error),
            (Err(sync_error), Err(restore_error)) => Err(Error::new(
                sync_error.kind(),
                format!(
                    "{}; restoring the previous current context also failed: {}",
                    sync_error.message(),
                    restore_error.message()
                ),
            )),
        }
    }

    fn close(&mut self) -> Result<()> {
        let Some(handle) = self.handle else {
            return Ok(());
        };

        self.api
            .destroy_context(handle)
            .map_err(|error| cuda_error("cuCtxDestroy_v2", error))?;
        self.handle = None;
        Ok(())
    }
}

impl Drop for NvidiaContext {
    fn drop(&mut self) {
        if let Some(handle) = self.handle {
            // Drop must not panic. Explicit close remains available when the
            // caller needs to observe and retry a destruction failure.
            let _ = self.api.destroy_context(handle);
            self.handle = None;
        }
    }
}
