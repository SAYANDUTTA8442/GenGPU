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

    pub(crate) fn from_raw(value: u64) -> Self {
        Self(value)
    }
}

/// Result of a backend cleanup attempt.
///
/// A backend can report an error after the resource has already been
/// irreversibly released (for example, when restoring thread context fails).
/// Such a result must disarm the cleanup callback to prevent double-free.
pub(crate) enum CleanupOutcome {
    Released(Result<()>),
    Retain(Error),
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
    cleanup: Option<Box<dyn FnMut() -> CleanupOutcome + Send + 'static>>,
    on_drop_failure: Option<Box<dyn FnOnce() + Send + 'static>>,
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
        cleanup: impl FnMut() -> CleanupOutcome + Send + 'static,
        on_drop_failure: impl FnOnce() + Send + 'static,
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
            on_drop_failure: Some(Box::new(on_drop_failure)),
            _owner: owner,
        })
    }

    /// Returns whether this allocation belongs to the exact backend owner.
    ///
    /// Device identity is deliberately not considered: separate contexts on
    /// the same physical device must not be allowed to access one another's
    /// allocations. The owner token remains opaque to callers.
    pub(crate) fn is_owned_by<T>(&self, owner: &Arc<T>) -> bool
    where
        T: Any + Send + Sync,
    {
        Arc::as_ptr(&self._owner) as *const () == Arc::as_ptr(owner) as *const ()
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
    /// If the resource remains live after cleanup fails, ownership state is
    /// retained so the caller may retry. If the resource was released but a
    /// later restoration step failed, the error is reported and the callback
    /// is disarmed to prevent a second release attempt.
    ///
    /// Calling this method after a completed deallocation is idempotent.
    pub fn deallocate(&mut self) -> Result<()> {
        let Some(cleanup) = self.cleanup.as_mut() else {
            return Ok(());
        };

        match cleanup() {
            CleanupOutcome::Released(result) => {
                self.cleanup = None;
                result
            }
            CleanupOutcome::Retain(error) => Err(error),
        }
    }

    /// Returns whether this allocation's device resource has been released.
    pub fn is_deallocated(&self) -> bool {
        self.cleanup.is_none()
    }
}

impl Drop for DeviceAllocation {
    fn drop(&mut self) {
        if let Some(cleanup) = self.cleanup.as_mut() {
            // Drop cannot report errors. If the resource remains live after
            // this best-effort attempt, transfer cleanup responsibility to the
            // owning context before releasing the owner token.
            if matches!(cleanup(), CleanupOutcome::Retain(_)) {
                self.cleanup = None;
                if let Some(on_drop_failure) = self.on_drop_failure.take() {
                    on_drop_failure();
                }
            } else {
                self.cleanup = None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::sync::{Arc, Mutex};

    use gengpu_core::{DeviceId, Error, ErrorKind};

    use super::{CleanupOutcome, DeviceAddress, DeviceAllocation};

    fn allocation(cleanup: impl FnMut() -> CleanupOutcome + Send + 'static) -> DeviceAllocation {
        DeviceAllocation::new(
            DeviceId::new(3),
            DeviceAddress::from_raw(0xCAFE),
            4096,
            Arc::new(()),
            cleanup,
            || {},
        )
        .expect("allocation should be constructed")
    }

    #[test]
    fn exposes_device_metadata_without_exposing_a_host_pointer() {
        let allocation = allocation(|| CleanupOutcome::Released(Ok(())));

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
            || CleanupOutcome::Released(Ok(())),
            || {},
        );

        assert_eq!(
            result.err().expect("zero-byte allocation must fail").kind(),
            ErrorKind::InvalidArgument
        );
    }

    #[test]
    fn allocation_owner_identity_requires_the_exact_owner() {
        let owner = Arc::new(String::from("context owner"));
        let erased_owner: Arc<dyn Any + Send + Sync> = owner.clone();
        let allocation = DeviceAllocation::new(
            DeviceId::new(3),
            DeviceAddress::from_raw(0xCAFE),
            64,
            erased_owner,
            || CleanupOutcome::Released(Ok(())),
            || {},
        )
        .unwrap();

        let another_owner = Arc::new(String::from("context owner"));

        assert!(allocation.is_owned_by(&owner));
        assert!(!allocation.is_owned_by(&another_owner));
    }

    #[test]
    fn explicit_deallocation_is_idempotent() {
        let calls = Arc::new(Mutex::new(0));
        let observed = Arc::clone(&calls);
        let mut allocation = allocation(move || {
            *observed.lock().unwrap() += 1;
            CleanupOutcome::Released(Ok(()))
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
                CleanupOutcome::Retain(Error::new(ErrorKind::Driver, "simulated free failure"))
            } else {
                CleanupOutcome::Released(Ok(()))
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
    fn released_resource_disarms_cleanup_even_when_restoration_reports_error() {
        let calls = Arc::new(Mutex::new(0));
        let observed = Arc::clone(&calls);
        let mut allocation = allocation(move || {
            *observed.lock().unwrap() += 1;
            CleanupOutcome::Released(Err(Error::new(
                ErrorKind::Driver,
                "simulated context restoration failure",
            )))
        });

        assert_eq!(
            allocation.deallocate().unwrap_err().kind(),
            ErrorKind::Driver
        );
        assert!(allocation.is_deallocated());
        allocation.deallocate().unwrap();
        assert_eq!(*calls.lock().unwrap(), 1);
    }

    #[test]
    fn drop_transfers_failed_cleanup_to_the_owner() {
        let cleanup_calls = Arc::new(Mutex::new(0));
        let observed_cleanup = Arc::clone(&cleanup_calls);
        let orphan_calls = Arc::new(Mutex::new(0));
        let observed_orphan = Arc::clone(&orphan_calls);

        drop(
            DeviceAllocation::new(
                DeviceId::new(3),
                DeviceAddress::from_raw(0xCAFE),
                4096,
                Arc::new(()),
                move || {
                    *observed_cleanup.lock().unwrap() += 1;
                    CleanupOutcome::Retain(Error::new(
                        ErrorKind::Driver,
                        "simulated persistent free failure",
                    ))
                },
                move || {
                    *observed_orphan.lock().unwrap() += 1;
                },
            )
            .unwrap(),
        );

        assert_eq!(*cleanup_calls.lock().unwrap(), 1);
        assert_eq!(*orphan_calls.lock().unwrap(), 1);
    }

    #[test]
    fn successful_drop_does_not_transfer_cleanup_to_the_owner() {
        let orphan_calls = Arc::new(Mutex::new(0));
        let observed_orphan = Arc::clone(&orphan_calls);

        drop(
            DeviceAllocation::new(
                DeviceId::new(3),
                DeviceAddress::from_raw(0xCAFE),
                4096,
                Arc::new(()),
                || CleanupOutcome::Released(Ok(())),
                move || {
                    *observed_orphan.lock().unwrap() += 1;
                },
            )
            .unwrap(),
        );

        assert_eq!(*orphan_calls.lock().unwrap(), 0);
    }

    #[test]
    fn drop_attempts_best_effort_cleanup_once() {
        let calls = Arc::new(Mutex::new(0));
        let observed = Arc::clone(&calls);
        drop(allocation(move || {
            *observed.lock().unwrap() += 1;
            CleanupOutcome::Released(Ok(()))
        }));

        assert_eq!(*calls.lock().unwrap(), 1);
    }

    #[test]
    fn drop_does_not_panic_when_cleanup_fails() {
        let calls = Arc::new(Mutex::new(0));
        let observed = Arc::clone(&calls);
        drop(allocation(move || {
            *observed.lock().unwrap() += 1;
            CleanupOutcome::Retain(Error::new(ErrorKind::Driver, "simulated free failure"))
        }));

        assert_eq!(*calls.lock().unwrap(), 1);
    }
}
