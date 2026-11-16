//! NVIDIA PTX module ownership and kernel-function resolution.

use std::ffi::CString;
use std::sync::{Arc, Mutex};

use gengpu_core::{Error, ErrorKind, Result};

use crate::{KernelFunction, Module};

use super::context::{combine_errors, pop_expected_context_retry, ContextInner};
use super::driver::cuda_error;

const CUDA_ERROR_NOT_FOUND: i32 = 500;

/// Native CUDA function handle retained as an opaque KernelFunction payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CudaFunctionHandle(pub usize);

/// A PTX module owned by one NVIDIA context.
///
/// The shared ContextInner keeps both the native CUDA context and loaded
/// driver API alive for the module's lifetime.
pub(super) struct NvidiaModule {
    inner: Arc<ContextInner>,
    handle: Mutex<Option<usize>>,
}

impl NvidiaModule {
    pub(super) fn new(inner: Arc<ContextInner>, handle: usize) -> Self {
        Self {
            inner,
            handle: Mutex::new(Some(handle)),
        }
    }

    fn module_handle(&self) -> Result<usize> {
        self.inner.handle()?;
        self.handle
            .lock()
            .map_err(|_| {
                Error::new(
                    ErrorKind::Internal,
                    "NVIDIA module handle mutex is poisoned",
                )
            })?
            .ok_or_else(|| Error::new(ErrorKind::InvalidState, "NVIDIA module is unloaded"))
    }

    fn unload_native(&mut self) -> Result<()> {
        let module_handle = {
            let handle = self.handle.lock().map_err(|_| {
                Error::new(
                    ErrorKind::Internal,
                    "NVIDIA module handle mutex is poisoned",
                )
            })?;
            match *handle {
                Some(handle) => handle,
                None => return Ok(()),
            }
        };
        let context_handle = self.inner.handle()?;
        let api = Arc::clone(self.inner.api());

        api.push_current(context_handle)
            .map_err(|error| cuda_error("cuCtxPushCurrent_v2", error))?;

        let unload_result = api
            .unload_module(module_handle)
            .map_err(|error| cuda_error("cuModuleUnload", error));
        let restoration = pop_expected_context_retry(&api, context_handle);

        match (unload_result, restoration) {
            (Ok(()), Ok(())) => {
                *self.handle.lock().map_err(|_| {
                    Error::new(
                        ErrorKind::Internal,
                        "NVIDIA module handle mutex is poisoned",
                    )
                })? = None;
                Ok(())
            }
            (Err(error), Ok(())) => Err(error),
            (Ok(()), Err(error)) => {
                // The module is already released. Never retry unload merely
                // because restoring the caller's context stack failed.
                *self.handle.lock().map_err(|_| {
                    Error::new(
                        ErrorKind::Internal,
                        "NVIDIA module handle mutex is poisoned",
                    )
                })? = None;
                Err(error)
            }
            (Err(error), Err(restore_error)) => Err(combine_errors(
                error,
                restore_error,
                "restoring the previous current context after module unload failed",
            )),
        }
    }
}

impl Module for NvidiaModule {
    fn get_function<'module>(&'module self, name: &str) -> Result<KernelFunction<'module>> {
        crate::module::validate_function_name(name)?;
        let name = CString::new(name).map_err(|_| {
            Error::new(
                ErrorKind::InvalidArgument,
                "kernel function name must not contain an embedded NUL byte",
            )
        })?;
        let module_handle = self.module_handle()?;
        let context_handle = self.inner.handle()?;
        let api = Arc::clone(self.inner.api());

        api.push_current(context_handle)
            .map_err(|error| cuda_error("cuCtxPushCurrent_v2", error))?;

        let lookup_result = api.get_function(module_handle, &name).map_err(|error| {
            if error.code == CUDA_ERROR_NOT_FOUND {
                let diagnostic = cuda_error("cuModuleGetFunction", error);
                Error::new(
                    ErrorKind::NotFound,
                    format!(
                        "kernel function '{}' was not found in the NVIDIA module: {}",
                        name.to_string_lossy(),
                        diagnostic.message()
                    ),
                )
            } else {
                cuda_error("cuModuleGetFunction", error)
            }
        });
        let restoration = pop_expected_context_retry(&api, context_handle);

        let function_handle = match (lookup_result, restoration) {
            (Ok(function), Ok(())) => function,
            (Err(error), Ok(())) => return Err(error),
            (Err(error), Err(restore_error)) => {
                return Err(combine_errors(
                    error,
                    restore_error,
                    "restoring context after kernel function lookup",
                ));
            }
            (Ok(_), Err(restore_error)) => return Err(restore_error),
        };

        Ok(KernelFunction::from_backend_handle(
            self,
            CudaFunctionHandle(function_handle),
        ))
    }

    fn unload(&mut self) -> Result<()> {
        self.unload_native()
    }
}

impl Drop for NvidiaModule {
    fn drop(&mut self) {
        // C3 adds context-owned orphan tracking for failed module cleanup.
        // Until then, context destruction remains the final reclamation
        // boundary if this best-effort unload cannot release the module.
        let _ = self.unload_native();
    }
}
