//! Backend-neutral PTX module and kernel-function contracts.

use std::any::Any;
use std::marker::PhantomData;
use std::rc::Rc;

use gengpu_core::{Error, ErrorKind, Result};

/// A loaded GPU program module owned by exactly one execution context.
///
/// A module retains the backend resources required for its lifetime. Kernel
/// function handles borrowed from a module cannot outlive that module.
pub trait Module: Send {
    /// Resolves a named kernel function in this module.
    ///
    /// An empty name or a name containing an embedded NUL byte is invalid.
    /// Implementations return ErrorKind::NotFound when the name is not
    /// present in the loaded module.
    fn get_function<'module>(&'module self, name: &str) -> Result<KernelFunction<'module>>;

    /// Explicitly unloads this module.
    ///
    /// Implementations must retain enough state after a failed unload for the
    /// caller to retry. A successful unload is idempotent.
    fn unload(&mut self) -> Result<()>;
}

/// An opaque function handle whose lifetime is bounded by its parent module.
///
/// The backend handle is intentionally type-erased. Backend implementations
/// may retrieve their own handle type with backend_handle. The module
/// borrow prevents safe code from retaining this function after the module
/// has been dropped. This type is neither Send nor Sync.
///
/// The borrow is enforced by the Rust type system:
///
/// ```compile_fail
/// use gengpu_core::Result;
/// use gengpu_driver::{KernelFunction, Module};
///
/// struct FakeModule;
///
/// impl Module for FakeModule {
///     fn get_function<'a>(&'a self, _name: &str) -> Result<KernelFunction<'a>> {
///         Ok(KernelFunction::from_backend_handle(self, 7_u64))
///     }
///
///     fn unload(&mut self) -> Result<()> {
///         Ok(())
///     }
/// }
///
/// fn invalid() -> KernelFunction<'static> {
///     let module = FakeModule;
///     module.get_function("entry").unwrap()
/// }
/// ```
pub struct KernelFunction<'module> {
    backend_handle: Box<dyn Any + Send + Sync>,
    _module_borrow: PhantomData<&'module dyn Module>,
    _thread_bound: PhantomData<Rc<()>>,
}

impl<'module> KernelFunction<'module> {
    /// Wraps a backend-specific function handle and ties it to its module.
    ///
    /// This constructor is public so alternative driver backends can
    /// implement the backend-neutral Module trait. The handle remains opaque
    /// to callers that do not know its concrete backend type.
    pub fn from_backend_handle<T>(module: &'module dyn Module, handle: T) -> Self
    where
        T: Any + Send + Sync,
    {
        let _ = module;
        Self {
            backend_handle: Box::new(handle),
            _module_borrow: PhantomData,
            _thread_bound: PhantomData,
        }
    }

    /// Returns the backend-specific handle when the requested type matches.
    ///
    /// This is intended for driver-backend implementations. It does not
    /// transfer ownership of the native handle.
    pub fn backend_handle<T: Any>(&self) -> Option<&T> {
        self.backend_handle.downcast_ref()
    }
}

/// Validates PTX text before it is passed to a backend API.
pub(crate) fn validate_ptx(ptx: &str) -> Result<()> {
    if ptx.is_empty() {
        return Err(Error::new(
            ErrorKind::InvalidArgument,
            "PTX source must not be empty",
        ));
    }

    if ptx.as_bytes().contains(&0) {
        return Err(Error::new(
            ErrorKind::InvalidArgument,
            "PTX source must not contain an embedded NUL byte",
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{validate_ptx, KernelFunction, Module};
    use gengpu_core::{ErrorKind, Result};

    struct StubModule;

    impl Module for StubModule {
        fn get_function<'module>(&'module self, _name: &str) -> Result<KernelFunction<'module>> {
            Ok(KernelFunction::from_backend_handle(self, 42_u64))
        }

        fn unload(&mut self) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn accepts_non_empty_ptx_without_embedded_nul() {
        assert!(validate_ptx(".version 7.0\n.target sm_50").is_ok());
    }

    #[test]
    fn rejects_empty_ptx() {
        assert_eq!(
            validate_ptx("").unwrap_err().kind(),
            ErrorKind::InvalidArgument
        );
    }

    #[test]
    fn rejects_ptx_with_embedded_nul() {
        assert_eq!(
            validate_ptx(".version 7.0\0.target sm_50")
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidArgument
        );
    }

    #[test]
    fn kernel_function_keeps_backend_handle_opaque_and_typed() {
        let module = StubModule;
        let function = module.get_function("entry").unwrap();

        assert_eq!(function.backend_handle::<u64>(), Some(&42_u64));
        assert!(function.backend_handle::<u32>().is_none());
    }
}
