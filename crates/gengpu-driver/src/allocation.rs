//! Backend-neutral ownership of a device-memory allocation.

use std::any::Any;
use std::sync::Arc;

use gengpu_core::{DeviceId, Error, ErrorKind, Result};

/// Opaque address in a GPU device's address space.
///
/// This is not a host pointer and must not be dereferenced by host code.
/// Its value is meaningful only for the device and live allocation that
/// produced it.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DeviceAddress(u64);

impl DeviceAddress {
    /// Returns the opaque numeric device address.
    pub fn as_u64(self) -> u64 {
        self.0
    }

    // Used by concrete backend implementations introduced in PH1-005C.
    #[allow(dead_code)]
    pub(crate) fn from_raw(value: u64) -> Self {
        Self(value)
    }
}

/// An owned allocation in a GPU device's memory space.
///
/// The allocation retains an owner token so backend context state can outlive
/// the public Context wrapper. The backend-provided cleanup callback is
/// invoked by explicit deallocation or, best-effort, by Drop.
pub struct DeviceAllocation {
    device_id: DeviceId,
    address: DeviceAddress,
    size_bytes: usize,
    cleanup: Option<Box<dyn FnMut() -> Result<()> + Send + 'static>>,
    _owner: Arc<dyn Any + Send + Sync>,
}

impl DeviceAllocation {
    /// Creates an allocation from backend-owned state.
    ///
    /// This constructor is crate-private: callers obtain allocations only
    /// through Context::allocate.
    pub(crate) fn new(
        device_id: DeviceId,
        address: DeviceAddress,
        size_bytes: usize,
        owner: Arc<dyn Any + Send + Sync>,
        cleanup: impl FnMut() -> Result<()> + Send + 'static,
    ) -> Result<Self> {
        if size_bytes == 0 {
            return Err(Error::new(
                ErrorKind::InvalidArgument,
                "device allocation size must be greater than zero",
            ));
        }

        Ok(Self {
            device_id,
            address,
            size_bytes,
            cleanup: Some(Box::new(cleanup)),
            _owner: owner,
        })
    }

    /// Returns the logical device that owns this allocation.
    pub fn device_id(&self) -> DeviceId {
        self.device_id
    }

    /// Returns the opaque device address.
    pub fn device_address(&self) -> DeviceAddress {
        self.address
    }

    /// Returns the allocation size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// Releases this allocation and reports backend cleanup failures.
    ///
    /// If cleanup fails, ownership state is retained so the caller may retry.
    /// Calling this method after a successful deallocation is idempotent.
    pub fn deallocate(&mut self) -> Result<()> {
        let Some(cleanup) = self.cleanup.as_mut() else {
            return Ok(());
        };

        cleanup()?;
        self.cleanup = None;
        Ok(())
    }

    /// Returns whether this allocation has been successfully deallocated.
    pub fn is_deallocated(&self) -> bool {
        self.cleanup.is_none()
    }
}

impl Drop for DeviceAllocation {
    fn drop(&mut self) {
        if let Some(cleanup) = self.cleanup.as_mut() {
            // Drop cannot report errors and must never panic. The owner token
            // remains alive throughout this best-effort cleanup attempt.
            let _ = cleanup();
            self.cleanup = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use gengpu_core::{DeviceId, Error, ErrorKind};

    use super::{DeviceAddress, DeviceAllocation};

    fn allocation(
        cleanup: impl FnMut() -> gengpu_core::Result<()> + Send + 'static,
    ) -> DeviceAllocation {
        DeviceAllocation::new(
            DeviceId::new(3),
            DeviceAddress::from_raw(0xCAFE),
            4096,
            Arc::new(()),
            cleanup,
        )
        .expect("allocation should be constructed")
    }

    #[test]
    fn exposes_device_metadata_without_exposing_a_host_pointer() {
        let allocation = allocation(|| Ok(()));

        assert_eq!(allocation.device_id(), DeviceId::new(3));
        assert_eq!(allocation.device_address().as_u64(), 0xCAFE);
        assert_eq!(allocation.size_bytes(), 4096);
        assert!(!allocation.is_deallocated());
    }

    #[test]
    fn rejects_zero_byte_allocations() {
        let result = DeviceAllocation::new(
            DeviceId::new(0),
            DeviceAddress::from_raw(1),
            0,
            Arc::new(()),
            || Ok(()),
        );

        assert_eq!(
            result.err().expect("zero-byte allocation must fail").kind(),
            ErrorKind::InvalidArgument
        );
    }

    #[test]
    fn explicit_deallocation_is_idempotent() {
        let calls = Arc::new(Mutex::new(0));
        let observed = Arc::clone(&calls);
        let mut allocation = allocation(move || {
            *observed.lock().unwrap() += 1;
            Ok(())
        });

        allocation.deallocate().unwrap();
        allocation.deallocate().unwrap();

        assert!(allocation.is_deallocated());
        assert_eq!(*calls.lock().unwrap(), 1);
    }

    #[test]
    fn failed_deallocation_retains_state_for_retry() {
        let calls = Arc::new(Mutex::new(0));
        let observed = Arc::clone(&calls);
        let mut allocation = allocation(move || {
            let mut calls = observed.lock().unwrap();
            *calls += 1;
            if *calls == 1 {
                Err(Error::new(ErrorKind::Driver, "simulated free failure"))
            } else {
                Ok(())
            }
        });

        assert_eq!(
            allocation.deallocate().unwrap_err().kind(),
            ErrorKind::Driver
        );
        assert!(!allocation.is_deallocated());
        allocation.deallocate().unwrap();

        assert!(allocation.is_deallocated());
        assert_eq!(*calls.lock().unwrap(), 2);
    }

    #[test]
    fn drop_attempts_best_effort_cleanup_once() {
        let calls = Arc::new(Mutex::new(0));
        let observed = Arc::clone(&calls);
        drop(allocation(move || {
            *observed.lock().unwrap() += 1;
            Ok(())
        }));

        assert_eq!(*calls.lock().unwrap(), 1);
    }

    #[test]
    fn drop_does_not_panic_when_cleanup_fails() {
        let calls = Arc::new(Mutex::new(0));
        let observed = Arc::clone(&calls);
        drop(allocation(move || {
            *observed.lock().unwrap() += 1;
            Err(Error::new(ErrorKind::Driver, "simulated free failure"))
        }));

        assert_eq!(*calls.lock().unwrap(), 1);
    }
}
