//! Backend-neutral GPU execution context contracts.

use std::marker::PhantomData;
use std::rc::Rc;

use gengpu_core::{DeviceId, Error, ErrorKind, Result};

use crate::{DeviceAllocation, KernelArg, KernelFunction, LaunchConfig, Module};

/// An owned execution context associated with one logical GPU device.
///
/// Contexts may be moved between threads, but are not shareable concurrently.
/// Implementations must ensure that operations which temporarily make a
/// context current restore the calling thread's previous current context.
pub trait Context: Send {
    /// Returns the logical device associated with this context.
    fn device_id(&self) -> DeviceId;

    /// Makes this context current on the calling thread for the lifetime of
    /// the returned guard.
    ///
    /// The mutable borrow held by the guard prevents this context from being
    /// closed or activated again until the guard is finished or dropped.
    fn make_current(&mut self) -> Result<CurrentContextGuard<'_>>;

    /// Allocates device memory owned by this context.
    ///
    /// The returned allocation retains the backend state required to keep
    /// its context alive. Explicit context closure must fail while allocations
    /// remain live. A zero-byte request is invalid.
    ///
    /// Backends that do not yet implement device allocation return
    /// ErrorKind::Unsupported.
    fn allocate(&self, size_bytes: usize) -> Result<DeviceAllocation> {
        let _ = size_bytes;
        Err(Error::new(
            ErrorKind::Unsupported,
            "this driver backend does not implement device memory allocation",
        ))
    }

    /// Copies bytes from host memory into an allocation owned by this context.
    ///
    /// The allocation is mutably borrowed for the complete operation. The
    /// offset is measured in bytes from the beginning of the allocation.
    /// Backends that do not yet implement host-to-device transfers return
    /// ErrorKind::Unsupported.
    fn copy_to_device(
        &self,
        destination: &mut DeviceAllocation,
        offset: usize,
        source: &[u8],
    ) -> Result<()> {
        let _ = (destination, offset, source);
        Err(Error::new(
            ErrorKind::Unsupported,
            "this driver backend does not implement host-to-device transfers",
        ))
    }

    /// Copies bytes from an allocation owned by this context into host memory.
    ///
    /// The allocation is mutably borrowed for the complete operation. The
    /// offset is measured in bytes from the beginning of the allocation.
    /// Backends that do not yet implement device-to-host transfers return
    /// ErrorKind::Unsupported.
    fn copy_from_device(
        &self,
        source: &mut DeviceAllocation,
        offset: usize,
        destination: &mut [u8],
    ) -> Result<()> {
        let _ = (source, offset, destination);
        Err(Error::new(
            ErrorKind::Unsupported,
            "this driver backend does not implement device-to-host transfers",
        ))
    }

    /// Loads PTX text as a module owned by this context.
    ///
    /// Empty PTX and PTX containing an embedded NUL byte are rejected before
    /// backend loading. Backends that do not yet implement module loading
    /// return ErrorKind::Unsupported.
    fn load_ptx(&self, ptx: &str) -> Result<Box<dyn Module>> {
        crate::module::validate_ptx(ptx)?;
        Err(Error::new(
            ErrorKind::Unsupported,
            "this driver backend does not implement PTX module loading",
        ))
    }

    /// Launches a kernel function owned by this exact execution context.
    ///
    /// The function's parent module and every device-pointer argument must
    /// belong to this exact context; matching logical device identifiers are
    /// insufficient. Implementations must reject ownership mismatches before
    /// submitting work to the backend.
    ///
    /// The operation is synchronous: after a successful launch submission,
    /// implementations must wait for the context's work to complete before
    /// returning success. Argument storage and device allocations remain
    /// borrowed for the complete operation, including synchronization.
    ///
    /// Argument order and representation must match the PTX function's
    /// parameter declarations. This API cannot verify arbitrary PTX
    /// signatures because compiler-generated parameter metadata is not yet
    /// available. Device-specific grid, block, and shared-memory limits are
    /// validated by the backend/driver.
    ///
    /// Backends that do not yet implement kernel launch return
    /// ErrorKind::Unsupported.
    fn launch(
        &self,
        function: &KernelFunction<'_>,
        config: LaunchConfig,
        args: &[KernelArg<'_>],
    ) -> Result<()> {
        let _ = (function, config, args);
        Err(Error::new(
            ErrorKind::Unsupported,
            "this driver backend does not implement kernel launch",
        ))
    }

    /// Synchronizes this context without changing the caller's observable
    /// current-context state.
    ///
    /// Implementations must temporarily activate the context if required and
    /// restore the previously current context before returning.
    fn synchronize(&mut self) -> Result<()>;

    /// Explicitly destroys this context.
    ///
    /// On success, the context becomes closed and subsequent operations must
    /// return an InvalidState error. If destruction fails, the implementation
    /// must retain enough ownership state for the caller to retry.
    fn close(&mut self) -> Result<()>;
}

/// A scoped association between a context and the calling host thread.
///
/// The guard holds an exclusive borrow of its context, so the context cannot
/// be closed or mutably used while the guard is alive. It is deliberately
/// neither Send nor Sync. Its cleanup callback is retained after a failed
/// explicit finish so Drop can make a best-effort second attempt to restore
/// the thread's previous current context.
pub struct CurrentContextGuard<'a> {
    cleanup: Option<Box<dyn FnMut() -> Result<()> + 'a>>,
    _context_borrow: PhantomData<&'a mut ()>,
    _thread_bound: PhantomData<Rc<()>>,
}

impl<'a> CurrentContextGuard<'a> {
    /// Creates a guard with backend-provided cleanup while exclusively
    /// borrowing the associated context for the guard's lifetime.
    ///
    /// This constructor is crate-private: external callers obtain guards only
    /// through Context::make_current.
    pub(crate) fn new<T: ?Sized>(
        _context: &'a mut T,
        cleanup: impl FnMut() -> Result<()> + 'a,
    ) -> Self {
        Self {
            cleanup: Some(Box::new(cleanup)),
            _context_borrow: PhantomData,
            _thread_bound: PhantomData,
        }
    }

    /// Restores the previous current-context state and reports cleanup errors.
    ///
    /// If cleanup fails, the guard's Drop implementation makes one
    /// best-effort retry when the backend determines that retry is safe.
    /// A backend may suppress retry after detecting a context-stack mismatch
    /// where another pop could disturb unrelated thread state.
    pub fn finish(mut self) -> Result<()> {
        let result = match self.cleanup.as_mut() {
            Some(cleanup) => cleanup(),
            None => Ok(()),
        };

        if result.is_ok() {
            self.cleanup = None;
        }

        result
    }
}

impl Drop for CurrentContextGuard<'_> {
    fn drop(&mut self) {
        if let Some(cleanup) = self.cleanup.as_mut() {
            let _ = cleanup();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::cell::Cell;
    use std::rc::Rc;
    use std::sync::Arc;

    use gengpu_core::{DeviceId, Error, ErrorKind, Result};

    use crate::allocation::CleanupOutcome;
    use crate::{
        DeviceAddress, DeviceAllocation, KernelArg, KernelFunction, LaunchConfig, Module,
    };

    use super::{Context, CurrentContextGuard};

    struct StubModule;

    impl Module for StubModule {
        fn get_function<'module>(&'module self, _name: &str) -> Result<KernelFunction<'module>> {
            Ok(KernelFunction::from_backend_handle(self, 7_u64))
        }

        fn unload(&mut self) -> Result<()> {
            Ok(())
        }
    }

    struct StubContext;

    impl Context for StubContext {
        fn device_id(&self) -> DeviceId {
            DeviceId::new(0)
        }

        fn make_current(&mut self) -> Result<CurrentContextGuard<'_>> {
            Ok(CurrentContextGuard::new(self, || Ok(())))
        }

        fn synchronize(&mut self) -> Result<()> {
            Ok(())
        }

        fn close(&mut self) -> Result<()> {
            Ok(())
        }
    }

    fn test_allocation() -> DeviceAllocation {
        let owner: Arc<dyn Any + Send + Sync> = Arc::new(());
        DeviceAllocation::new(
            DeviceId::new(0),
            DeviceAddress::from_raw(0xCAFE),
            16,
            owner,
            || CleanupOutcome::Released(Ok(())),
            || {},
        )
        .unwrap()
    }

    #[test]
    fn transfer_methods_default_to_unsupported() {
        let context = StubContext;
        let mut allocation = test_allocation();
        let mut destination = [0_u8; 4];

        let to_device = context
            .copy_to_device(&mut allocation, 0, &[1, 2, 3, 4])
            .unwrap_err();
        let from_device = context
            .copy_from_device(&mut allocation, 0, &mut destination)
            .unwrap_err();

        assert_eq!(to_device.kind(), ErrorKind::Unsupported);
        assert_eq!(from_device.kind(), ErrorKind::Unsupported);
    }

    #[test]
    fn ptx_loading_defaults_to_unsupported_for_valid_source() {
        let context = StubContext;
        let error = context
            .load_ptx(".version 7.0\n.target sm_50")
            .err()
            .expect("valid PTX should reach the unsupported backend");

        assert_eq!(error.kind(), ErrorKind::Unsupported);
    }

    #[test]
    fn ptx_loading_rejects_empty_source_before_backend_dispatch() {
        let context = StubContext;
        let error = context
            .load_ptx("")
            .err()
            .expect("empty PTX must be rejected");

        assert_eq!(error.kind(), ErrorKind::InvalidArgument);
    }

    #[test]
    fn ptx_loading_rejects_embedded_nul_before_backend_dispatch() {
        let context = StubContext;
        let error = context
            .load_ptx(".version 7.0\0.target sm_50")
            .err()
            .expect("embedded NUL must be rejected");

        assert_eq!(error.kind(), ErrorKind::InvalidArgument);
    }

    #[test]
    fn kernel_launch_defaults_to_unsupported_for_valid_request() {
        let context = StubContext;
        let module = StubModule;
        let function = module.get_function("entry").unwrap();
        let config = LaunchConfig::new(
            crate::Dim3::new(1, 1, 1).unwrap(),
            crate::Dim3::new(32, 1, 1).unwrap(),
            0,
        )
        .unwrap();

        let error = context
            .launch(&function, config, &[])
            .expect_err("stub backend must not claim to support kernel launch");

        assert_eq!(error.kind(), ErrorKind::Unsupported);
    }

    #[test]
    fn kernel_launch_api_accepts_scalar_and_device_pointer_arguments() {
        let context = StubContext;
        let module = StubModule;
        let function = module.get_function("entry").unwrap();
        let config = LaunchConfig::new(
            crate::Dim3::new(1, 1, 1).unwrap(),
            crate::Dim3::new(32, 1, 1).unwrap(),
            0,
        )
        .unwrap();
        let scalar = 42_u32;
        let scalar_argument = KernelArg::scalar(&scalar);
        let mut allocation = test_allocation();
        let pointer_argument = KernelArg::device_ptr(&mut allocation).unwrap();
        let arguments = [scalar_argument, pointer_argument];

        let error = context
            .launch(&function, config, &arguments)
            .expect_err("stub backend must report launch as unsupported");

        assert_eq!(error.kind(), ErrorKind::Unsupported);
    }

    #[test]
    fn finishing_guard_runs_cleanup_once() {
        let attempts = Rc::new(Cell::new(0));
        let observed = Rc::clone(&attempts);
        let mut context = ();
        let guard = CurrentContextGuard::new(&mut context, move || {
            observed.set(observed.get() + 1);
            Ok(())
        });

        guard.finish().expect("cleanup should succeed");
        assert_eq!(attempts.get(), 1);
    }

    #[test]
    fn dropping_guard_runs_best_effort_cleanup() {
        let attempts = Rc::new(Cell::new(0));
        let observed = Rc::clone(&attempts);
        let mut context = ();
        let guard = CurrentContextGuard::new(&mut context, move || {
            observed.set(observed.get() + 1);
            Ok(())
        });

        drop(guard);
        assert_eq!(attempts.get(), 1);
    }

    #[test]
    fn failed_finish_retains_cleanup_for_drop_retry() {
        let attempts = Rc::new(Cell::new(0));
        let observed = Rc::clone(&attempts);
        let mut context = ();
        let guard = CurrentContextGuard::new(&mut context, move || {
            observed.set(observed.get() + 1);
            if observed.get() == 1 {
                Err(Error::new(ErrorKind::Driver, "simulated pop failure"))
            } else {
                Ok(())
            }
        });

        assert!(guard.finish().is_err());
        assert_eq!(attempts.get(), 2);
    }
}
