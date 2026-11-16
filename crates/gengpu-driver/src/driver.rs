//! Backend-neutral driver interface.

use gengpu_core::{DeviceId, DeviceInfo, Error, ErrorKind, Result};

use crate::context::Context;

/// Interface implemented by a concrete GPU driver backend.
///
/// Implementations own their backend-specific loader state and resolved
/// function pointers. Native API handles and error codes must not cross this
/// interface.
pub trait Driver {
    /// Initializes the driver backend.
    ///
    /// Initialization must be idempotent for a given driver instance:
    /// repeated calls after successful initialization must succeed without
    /// repeating backend initialization.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend cannot be initialized.
    fn initialize(&self) -> Result<()>;

    /// Enumerates the devices exposed by the initialized backend.
    ///
    /// The returned identifiers are logical GenGPU identifiers. They are not
    /// persistent hardware identifiers.
    ///
    /// # Errors
    ///
    /// Returns an error with kind InvalidState if initialization has not
    /// completed successfully, or a backend error if enumeration fails.
    fn enumerate_devices(&self) -> Result<Vec<DeviceInfo>>;

    /// Creates an owned execution context for a previously enumerated device.
    ///
    /// Context creation is explicit and does not occur during device
    /// enumeration. The returned context retains the backend resources needed
    /// for its lifetime, even if this driver value is dropped.
    ///
    /// # Errors
    ///
    /// Returns InvalidState if the driver has not been initialized,
    /// NotFound if the logical device is not available, or a backend error if
    /// context creation fails.
    fn create_context(&self, device: DeviceId) -> Result<Box<dyn Context>> {
        let _ = device;
        Err(Error::new(
            ErrorKind::Unsupported,
            "this driver backend does not implement context creation",
        ))
    }
}
