//! Private, injectable interface to the CUDA Driver API.

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
}

/// Loads an implementation of the private CUDA API.
pub(super) trait CudaLoader: Send + Sync {
    fn load(&self) -> gengpu_core::Result<Box<dyn CudaApi>>;
}
