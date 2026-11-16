//! Private, injectable interface to the CUDA Driver API.

use std::ffi::CStr;

/// Diagnostic information returned by a failed CUDA Driver API call.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct CudaError {
    pub code: i32,
    pub name: Option<String>,
    pub description: Option<String>,
}

impl CudaError {
    pub fn new(code: i32) -> Self {
        Self {
            code,
            name: None,
            description: None,
        }
    }
}

/// Device properties needed by the backend-neutral driver interface.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct RawDeviceInfo {
    pub name: String,
    pub total_memory_bytes: u64,
}

/// Narrow abstraction over the CUDA calls used by this backend.
///
/// This trait is private so tests can inject a deterministic implementation
/// without requiring an installed NVIDIA driver or physical GPU.
pub(super) trait CudaApi: Send + Sync {
    fn initialize(&self) -> std::result::Result<(), CudaError>;
    fn driver_version(&self) -> std::result::Result<i32, CudaError>;
    fn device_count(&self) -> std::result::Result<i32, CudaError>;
    fn device_info(&self, ordinal: i32) -> std::result::Result<RawDeviceInfo, CudaError>;
    fn create_context(&self, ordinal: i32) -> std::result::Result<usize, CudaError>;
    fn destroy_context(&self, handle: usize) -> std::result::Result<(), CudaError>;
    fn push_current(&self, handle: usize) -> std::result::Result<(), CudaError>;
    fn pop_current(&self) -> std::result::Result<usize, CudaError>;
    fn synchronize(&self) -> std::result::Result<(), CudaError>;
    fn copy_to_device(&self, address: u64, source: &[u8]) -> std::result::Result<(), CudaError>;
    fn copy_from_device(
        &self,
        address: u64,
        destination: &mut [u8],
    ) -> std::result::Result<(), CudaError>;
    fn allocate_memory(&self, size_bytes: usize) -> std::result::Result<u64, CudaError>;
    fn free_memory(&self, address: u64) -> std::result::Result<(), CudaError>;
    fn load_module(&self, ptx: &CStr) -> std::result::Result<usize, CudaError>;
    fn unload_module(&self, module: usize) -> std::result::Result<(), CudaError>;
    fn get_function(&self, module: usize, name: &CStr) -> std::result::Result<usize, CudaError>;
}

/// Loads an implementation of the private CUDA API.
pub(super) trait CudaLoader: Send + Sync {
    fn load(&self) -> gengpu_core::Result<Box<dyn CudaApi>>;
}
