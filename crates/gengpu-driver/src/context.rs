//! Backend-neutral GPU execution context contracts.

use std::marker::PhantomData;
use std::rc::Rc;

use gengpu_core::{DeviceId, Result};

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
    /// best-effort retry before the guard is discarded.
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
    use std::cell::Cell;
    use std::rc::Rc;

    use gengpu_core::{Error, ErrorKind};

    use super::CurrentContextGuard;

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
