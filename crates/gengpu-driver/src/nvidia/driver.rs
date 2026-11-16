//! Public NVIDIA driver implementation and lifecycle management.

use std::sync::{Arc, Mutex};

use gengpu_core::{DeviceId, DeviceInfo, Error, ErrorKind, Result};

use super::api::{CudaApi, CudaError, CudaLoader};
use super::context::NvidiaContext;
use super::loader::DynamicCudaLoader;

/// NVIDIA GPU driver backed by the CUDA Driver API.
///
/// The CUDA driver library is loaded only when initialize is first called.
/// Constructing this value does not require an installed NVIDIA driver. Calls
/// are serialized through an internal mutex.
pub struct NvidiaDriver {
    pub(super) loader: Box<dyn CudaLoader>,
    pub(super) state: Mutex<DriverState>,
}

#[derive(Default)]
pub(super) struct DriverState {
    api: Option<Arc<dyn CudaApi>>,
    driver_version: Option<i32>,
}

impl NvidiaDriver {
    /// Creates an NVIDIA driver backend using the platform's dynamic loader.
    pub fn new() -> Self {
        Self {
            loader: Box::new(DynamicCudaLoader),
            state: Mutex::new(DriverState::default()),
        }
    }

    /// Returns the CUDA Driver API version reported after initialization.
    ///
    /// CUDA encodes this as an integer (for example, 12080 represents
    /// version 12.8). The raw value is returned to avoid imposing a version
    /// parsing policy on callers.
    ///
    /// # Errors
    ///
    /// Returns InvalidState before successful initialization.
    pub fn driver_version(&self) -> Result<i32> {
        let state = self.lock_state()?;
        state.driver_version.ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidState,
                "NVIDIA driver has not been initialized",
            )
        })
    }

    fn lock_state(&self) -> Result<std::sync::MutexGuard<'_, DriverState>> {
        self.state
            .lock()
            .map_err(|_| Error::new(ErrorKind::Internal, "NVIDIA driver state mutex is poisoned"))
    }
}

impl Default for NvidiaDriver {
    fn default() -> Self {
        Self::new()
    }
}

impl super::super::Driver for NvidiaDriver {
    fn initialize(&self) -> Result<()> {
        let mut state = self.lock_state()?;

        if state.api.is_some() {
            return Ok(());
        }

        let api = self.loader.load()?;
        api.initialize()
            .map_err(|error| cuda_error("cuInit", error))?;
        let driver_version = api
            .driver_version()
            .map_err(|error| cuda_error("cuDriverGetVersion", error))?;

        state.driver_version = Some(driver_version);
        state.api = Some(Arc::from(api));
        Ok(())
    }

    fn enumerate_devices(&self) -> Result<Vec<DeviceInfo>> {
        let api = {
            let state = self.lock_state()?;
            state.api.clone().ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidState,
                    "NVIDIA driver must be initialized before device enumeration",
                )
            })?
        };

        let count = api
            .device_count()
            .map_err(|error| cuda_error("cuDeviceGetCount", error))?;

        if count < 0 {
            return Err(Error::new(
                ErrorKind::Driver,
                format!("cuDeviceGetCount returned an invalid negative device count: {count}"),
            ));
        }

        let mut devices = Vec::with_capacity(count as usize);
        for ordinal in 0..count {
            let raw = api
                .device_info(ordinal)
                .map_err(|error| cuda_error("device metadata query", error))?;

            devices.push(DeviceInfo {
                id: DeviceId::new(ordinal as u32),
                name: raw.name,
                total_memory_bytes: raw.total_memory_bytes,
            });
        }

        Ok(devices)
    }

    fn create_context(&self, device: DeviceId) -> Result<Box<dyn crate::Context>> {
        let api = {
            let state = self.lock_state()?;
            state.api.clone().ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidState,
                    "NVIDIA driver must be initialized before context creation",
                )
            })?
        };

        let count = api
            .device_count()
            .map_err(|error| cuda_error("cuDeviceGetCount", error))?;
        if count < 0 {
            return Err(Error::new(
                ErrorKind::Driver,
                format!("cuDeviceGetCount returned an invalid negative device count: {count}"),
            ));
        }

        let ordinal = i32::try_from(device.index()).map_err(|_| {
            Error::new(
                ErrorKind::NotFound,
                "logical NVIDIA device ID is out of range",
            )
        })?;
        if ordinal >= count {
            return Err(Error::new(
                ErrorKind::NotFound,
                format!("logical NVIDIA device {} is not available", device.index()),
            ));
        }

        let handle = api
            .create_context(ordinal)
            .map_err(|error| cuda_error("context creation", error))?;

        Ok(Box::new(NvidiaContext::new(api, handle, device)))
    }
}

pub(super) fn cuda_error(operation: &str, error: CudaError) -> Error {
    let mut message = format!("{operation} failed with CUDA error code {}", error.code);

    if let Some(name) = error.name {
        message.push_str(&format!(" ({name})"));
    }

    if let Some(description) = error.description {
        message.push_str(&format!(": {description}"));
    }

    Error::new(ErrorKind::Driver, message)
}
