//! Backend-neutral driver interface.

use gengpu_core::{DeviceInfo, Result};

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
}
