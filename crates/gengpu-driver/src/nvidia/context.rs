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
        self.handle.ok_or_else(|| {
            Error::new(ErrorKind::InvalidState, "NVIDIA context is already closed")
        })
    }
}

impl Context for NvidiaContext {
    fn device_id(&self) -> DeviceId {
        self.device_id
    }

    fn make_current(&mut self) -> Result<CurrentContextGuard<'_>> {
        self.ensure_open()?;
        Err(Error::new(
            ErrorKind::Unsupported,
            "NVIDIA current-context activation is not implemented until PH1-004D",
        ))
    }

    fn synchronize(&mut self) -> Result<()> {
        self.ensure_open()?;
        Err(Error::new(
            ErrorKind::Unsupported,
            "NVIDIA context synchronization is not implemented until PH1-004D",
        ))
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
