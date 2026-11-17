//! Runtime loading and private CUDA Driver API function table.

use std::ffi::{c_char, c_int, c_uint, c_void, CStr};
use std::ptr;

use gengpu_core::{Error, ErrorKind, Result};
use libloading::{Library, Symbol};

use super::api::{CudaApi, CudaError, CudaLoader, RawDeviceInfo};

type CuResult = c_int;
type CuDevice = c_int;
type CuContext = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;

type CuInit = unsafe extern "C" fn(flags: c_uint) -> CuResult;
type CuDriverGetVersion = unsafe extern "C" fn(driver_version: *mut c_int) -> CuResult;
type CuDeviceGetCount = unsafe extern "C" fn(count: *mut c_int) -> CuResult;
type CuDeviceGet = unsafe extern "C" fn(device: *mut CuDevice, ordinal: c_int) -> CuResult;
type CuDeviceGetName =
    unsafe extern "C" fn(name: *mut c_char, length: c_int, device: CuDevice) -> CuResult;
type CuDeviceTotalMem = unsafe extern "C" fn(bytes: *mut usize, device: CuDevice) -> CuResult;
type CuCtxCreate =
    unsafe extern "C" fn(context: *mut CuContext, flags: c_uint, device: CuDevice) -> CuResult;
type CuCtxDestroy = unsafe extern "C" fn(context: CuContext) -> CuResult;
type CuCtxPopCurrent = unsafe extern "C" fn(context: *mut CuContext) -> CuResult;
type CuCtxPushCurrent = unsafe extern "C" fn(context: CuContext) -> CuResult;
type CuCtxSynchronize = unsafe extern "C" fn() -> CuResult;
type CuMemcpyHtoD =
    unsafe extern "C" fn(destination: u64, source: *const c_void, bytesize: usize) -> CuResult;
type CuMemcpyDtoH =
    unsafe extern "C" fn(destination: *mut c_void, source: u64, bytesize: usize) -> CuResult;
type CuMemAlloc = unsafe extern "C" fn(address: *mut u64, bytesize: usize) -> CuResult;
type CuMemFree = unsafe extern "C" fn(address: u64) -> CuResult;
type CuModuleLoadData =
    unsafe extern "C" fn(module: *mut CuModule, image: *const c_void) -> CuResult;
type CuModuleUnload = unsafe extern "C" fn(module: CuModule) -> CuResult;
type CuModuleGetFunction = unsafe extern "C" fn(
    function: *mut CuFunction,
    module: CuModule,
    name: *const c_char,
) -> CuResult;
type CuLaunchKernel = unsafe extern "C" fn(
    function: CuFunction,
    grid_x: c_uint,
    grid_y: c_uint,
    grid_z: c_uint,
    block_x: c_uint,
    block_y: c_uint,
    block_z: c_uint,
    shared_mem_bytes: c_uint,
    stream: *mut c_void,
    kernel_params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> CuResult;
type CuGetErrorName = unsafe extern "C" fn(error: CuResult, name: *mut *const c_char) -> CuResult;
type CuGetErrorString =
    unsafe extern "C" fn(error: CuResult, description: *mut *const c_char) -> CuResult;

const CUDA_DEVICE_NAME_BUFFER_LENGTH: c_int = 256;

/// Owns the loaded library for at least as long as its copied function pointers.
struct DynamicCudaApi {
    _library: Library,
    cu_init: CuInit,
    cu_driver_get_version: CuDriverGetVersion,
    cu_device_get_count: CuDeviceGetCount,
    cu_device_get: CuDeviceGet,
    cu_device_get_name: CuDeviceGetName,
    cu_device_total_mem: CuDeviceTotalMem,
    cu_ctx_create: CuCtxCreate,
    cu_ctx_destroy: CuCtxDestroy,
    cu_ctx_pop_current: CuCtxPopCurrent,
    cu_ctx_push_current: CuCtxPushCurrent,
    cu_ctx_synchronize: CuCtxSynchronize,
    cu_memcpy_htod: CuMemcpyHtoD,
    cu_memcpy_dtoh: CuMemcpyDtoH,
    cu_mem_alloc: CuMemAlloc,
    cu_mem_free: CuMemFree,
    cu_module_load_data: CuModuleLoadData,
    cu_module_unload: CuModuleUnload,
    cu_module_get_function: CuModuleGetFunction,
    cu_launch_kernel: CuLaunchKernel,
    cu_get_error_name: Option<CuGetErrorName>,
    cu_get_error_string: Option<CuGetErrorString>,
}

impl DynamicCudaApi {
    #[cfg(target_os = "linux")]
    fn load() -> Result<Self> {
        // Loading a shared library executes its platform loader logic. The
        // library is explicitly selected and all resolved symbols are private.
        let library = unsafe { Library::new("libcuda.so.1") }.map_err(|error| {
            Error::new(
                ErrorKind::NotFound,
                format!("failed to load NVIDIA driver library libcuda.so.1: {error}"),
            )
        })?;

        let cu_init = required_symbol::<CuInit>(&library, b"cuInit\0")?;
        let cu_driver_get_version =
            required_symbol::<CuDriverGetVersion>(&library, b"cuDriverGetVersion\0")?;
        let cu_device_get_count =
            required_symbol::<CuDeviceGetCount>(&library, b"cuDeviceGetCount\0")?;
        let cu_device_get = required_symbol::<CuDeviceGet>(&library, b"cuDeviceGet\0")?;
        let cu_device_get_name =
            required_symbol::<CuDeviceGetName>(&library, b"cuDeviceGetName\0")?;
        let cu_device_total_mem =
            required_symbol::<CuDeviceTotalMem>(&library, b"cuDeviceTotalMem_v2\0")?;
        let cu_ctx_create = required_symbol::<CuCtxCreate>(&library, b"cuCtxCreate_v2\0")?;
        let cu_ctx_destroy = required_symbol::<CuCtxDestroy>(&library, b"cuCtxDestroy_v2\0")?;
        let cu_ctx_pop_current =
            required_symbol::<CuCtxPopCurrent>(&library, b"cuCtxPopCurrent_v2\0")?;
        let cu_ctx_push_current =
            required_symbol::<CuCtxPushCurrent>(&library, b"cuCtxPushCurrent_v2\0")?;
        let cu_ctx_synchronize =
            required_symbol::<CuCtxSynchronize>(&library, b"cuCtxSynchronize\0")?;
        let cu_memcpy_htod = required_symbol::<CuMemcpyHtoD>(&library, b"cuMemcpyHtoD_v2\0")?;
        let cu_memcpy_dtoh = required_symbol::<CuMemcpyDtoH>(&library, b"cuMemcpyDtoH_v2\0")?;

        let cu_mem_alloc = required_symbol::<CuMemAlloc>(&library, b"cuMemAlloc_v2\0")?;
        let cu_mem_free = required_symbol::<CuMemFree>(&library, b"cuMemFree_v2\0")?;
        let cu_module_load_data =
            required_symbol::<CuModuleLoadData>(&library, b"cuModuleLoadData\0")?;
        let cu_module_unload = required_symbol::<CuModuleUnload>(&library, b"cuModuleUnload\0")?;
        let cu_module_get_function =
            required_symbol::<CuModuleGetFunction>(&library, b"cuModuleGetFunction\0")?;
        let cu_launch_kernel =
            required_symbol::<CuLaunchKernel>(&library, b"cuLaunchKernel\0")?;

        // Error-description helpers improve diagnostics but are not required
        // for correct operation of the backend.
        let cu_get_error_name = optional_symbol::<CuGetErrorName>(&library, b"cuGetErrorName\0");
        let cu_get_error_string =
            optional_symbol::<CuGetErrorString>(&library, b"cuGetErrorString\0");

        Ok(Self {
            _library: library,
            cu_init,
            cu_driver_get_version,
            cu_device_get_count,
            cu_device_get,
            cu_device_get_name,
            cu_device_total_mem,
            cu_ctx_create,
            cu_ctx_destroy,
            cu_ctx_pop_current,
            cu_ctx_push_current,
            cu_ctx_synchronize,
            cu_memcpy_htod,
            cu_memcpy_dtoh,
            cu_mem_alloc,
            cu_mem_free,
            cu_module_load_data,
            cu_module_unload,
            cu_module_get_function,
            cu_launch_kernel,
            cu_get_error_name,
            cu_get_error_string,
        })
    }

    #[cfg(not(target_os = "linux"))]
    fn load() -> Result<Self> {
        Err(Error::new(
            ErrorKind::Unsupported,
            "the NVIDIA driver loader currently supports Linux only",
        ))
    }

    fn translate_error(&self, code: CuResult) -> CudaError {
        let mut error = CudaError::new(code);

        if let Some(get_error_name) = self.cu_get_error_name {
            let mut name = ptr::null();
            // CUDA writes a pointer to a driver-owned, null-terminated string.
            let result = unsafe { get_error_name(code, &mut name) };
            if result == 0 && !name.is_null() {
                error.name = Some(
                    unsafe { CStr::from_ptr(name) }
                        .to_string_lossy()
                        .into_owned(),
                );
            }
        }

        if let Some(get_error_string) = self.cu_get_error_string {
            let mut description = ptr::null();
            // CUDA writes a pointer to a driver-owned, null-terminated string.
            let result = unsafe { get_error_string(code, &mut description) };
            if result == 0 && !description.is_null() {
                error.description = Some(
                    unsafe { CStr::from_ptr(description) }
                        .to_string_lossy()
                        .into_owned(),
                );
            }
        }

        error
    }

    fn check(&self, result: CuResult) -> std::result::Result<(), CudaError> {
        if result == 0 {
            Ok(())
        } else {
            Err(self.translate_error(result))
        }
    }

    fn destroy_after_failed_detach(&self, context: CuContext, error: &mut CudaError) {
        if let Err(cleanup_error) = self.check(unsafe { (self.cu_ctx_destroy)(context) }) {
            let detail = format!(
                "best-effort context destruction also failed with CUDA error code {}",
                cleanup_error.code
            );
            error.description = Some(match error.description.take() {
                Some(existing) => format!("{existing}; {detail}"),
                None => detail,
            });
        }
    }
}

impl CudaApi for DynamicCudaApi {
    fn initialize(&self) -> std::result::Result<(), CudaError> {
        self.check(unsafe { (self.cu_init)(0) })
    }

    fn driver_version(&self) -> std::result::Result<i32, CudaError> {
        let mut version = 0;
        self.check(unsafe { (self.cu_driver_get_version)(&mut version) })?;
        Ok(version)
    }

    fn device_count(&self) -> std::result::Result<i32, CudaError> {
        let mut count = 0;
        self.check(unsafe { (self.cu_device_get_count)(&mut count) })?;
        Ok(count)
    }

    fn device_info(&self, ordinal: i32) -> std::result::Result<RawDeviceInfo, CudaError> {
        let mut device = 0;
        self.check(unsafe { (self.cu_device_get)(&mut device, ordinal) })?;

        let mut name = [0; CUDA_DEVICE_NAME_BUFFER_LENGTH as usize];
        self.check(unsafe {
            (self.cu_device_get_name)(name.as_mut_ptr(), CUDA_DEVICE_NAME_BUFFER_LENGTH, device)
        })?;
        let name = unsafe { CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        let mut total_memory = 0_usize;
        self.check(unsafe { (self.cu_device_total_mem)(&mut total_memory, device) })?;

        Ok(RawDeviceInfo {
            name,
            total_memory_bytes: total_memory as u64,
        })
    }

    fn create_context(&self, ordinal: i32) -> std::result::Result<usize, CudaError> {
        let mut device = 0;
        self.check(unsafe { (self.cu_device_get)(&mut device, ordinal) })?;

        let mut context = ptr::null_mut();
        self.check(unsafe { (self.cu_ctx_create)(&mut context, 0, device) })?;
        if context.is_null() {
            let mut error = CudaError::new(-1);
            error.description = Some(String::from(
                "cuCtxCreate_v2 succeeded but returned a null context handle",
            ));
            return Err(error);
        }

        // cuCtxCreate_v2 makes the new context current on this thread. Pop it
        // immediately so creation returns a detached, owned context.
        let mut popped = ptr::null_mut();
        if let Err(mut error) = self.check(unsafe { (self.cu_ctx_pop_current)(&mut popped) }) {
            self.destroy_after_failed_detach(context, &mut error);
            return Err(error);
        }

        if popped != context {
            let mut error = CudaError::new(-1);
            error.description = Some(String::from(
                "cuCtxPopCurrent_v2 returned a context different from the one just created",
            ));

            // Restore the unexpected popped context before cleaning up ours.
            if !popped.is_null() {
                if let Err(push_error) = self.check(unsafe { (self.cu_ctx_push_current)(popped) }) {
                    error.description = Some(format!(
                        "{}; restoring the prior context failed with CUDA error code {}",
                        error
                            .description
                            .as_deref()
                            .unwrap_or("context stack mismatch"),
                        push_error.code
                    ));
                }
            }

            self.destroy_after_failed_detach(context, &mut error);
            return Err(error);
        }

        Ok(context as usize)
    }

    fn destroy_context(&self, handle: usize) -> std::result::Result<(), CudaError> {
        self.check(unsafe { (self.cu_ctx_destroy)(handle as CuContext) })
    }

    fn push_current(&self, handle: usize) -> std::result::Result<(), CudaError> {
        self.check(unsafe { (self.cu_ctx_push_current)(handle as CuContext) })
    }

    fn pop_current(&self) -> std::result::Result<usize, CudaError> {
        let mut context = ptr::null_mut();
        self.check(unsafe { (self.cu_ctx_pop_current)(&mut context) })?;
        Ok(context as usize)
    }

    fn synchronize(&self) -> std::result::Result<(), CudaError> {
        self.check(unsafe { (self.cu_ctx_synchronize)() })
    }

    fn copy_to_device(&self, address: u64, source: &[u8]) -> std::result::Result<(), CudaError> {
        self.check(unsafe { (self.cu_memcpy_htod)(address, source.as_ptr().cast(), source.len()) })
    }

    fn copy_from_device(
        &self,
        address: u64,
        destination: &mut [u8],
    ) -> std::result::Result<(), CudaError> {
        self.check(unsafe {
            (self.cu_memcpy_dtoh)(destination.as_mut_ptr().cast(), address, destination.len())
        })
    }

    fn allocate_memory(&self, size_bytes: usize) -> std::result::Result<u64, CudaError> {
        let mut address = 0_u64;
        self.check(unsafe { (self.cu_mem_alloc)(&mut address, size_bytes) })?;
        Ok(address)
    }

    fn free_memory(&self, address: u64) -> std::result::Result<(), CudaError> {
        self.check(unsafe { (self.cu_mem_free)(address) })
    }

    fn load_module(&self, ptx: &CStr) -> std::result::Result<usize, CudaError> {
        let mut module = ptr::null_mut();
        self.check(unsafe { (self.cu_module_load_data)(&mut module, ptx.as_ptr().cast()) })?;
        if module.is_null() {
            let mut error = CudaError::new(-1);
            error.description = Some(String::from(
                "cuModuleLoadData succeeded but returned a null module handle",
            ));
            return Err(error);
        }
        Ok(module as usize)
    }

    fn unload_module(&self, module: usize) -> std::result::Result<(), CudaError> {
        self.check(unsafe { (self.cu_module_unload)(module as CuModule) })
    }

    fn get_function(&self, module: usize, name: &CStr) -> std::result::Result<usize, CudaError> {
        let mut function = ptr::null_mut();
        self.check(unsafe {
            (self.cu_module_get_function)(&mut function, module as CuModule, name.as_ptr())
        })?;
        if function.is_null() {
            let mut error = CudaError::new(500);
            error.description = Some(String::from(
                "cuModuleGetFunction succeeded but returned a null function handle",
            ));
            return Err(error);
        }
        Ok(function as usize)
    }

    fn launch_kernel(
        &self,
        function: usize,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem_bytes: u32,
        kernel_params: &[*mut c_void],
    ) -> std::result::Result<(), CudaError> {
        let kernel_params = if kernel_params.is_empty() {
            ptr::null_mut()
        } else {
            kernel_params.as_ptr().cast_mut()
        };

        self.check(unsafe {
            (self.cu_launch_kernel)(
                function as CuFunction,
                grid[0],
                grid[1],
                grid[2],
                block[0],
                block[1],
                block[2],
                shared_mem_bytes,
                ptr::null_mut(),
                kernel_params,
                ptr::null_mut(),
            )
        })
    }
}

/// Production loader for the NVIDIA CUDA Driver API.
pub(super) struct DynamicCudaLoader;

impl CudaLoader for DynamicCudaLoader {
    fn load(&self) -> Result<Box<dyn CudaApi>> {
        Ok(Box::new(DynamicCudaApi::load()?))
    }
}

fn required_symbol<T: Copy>(library: &Library, name: &'static [u8]) -> Result<T> {
    // The symbol is copied as a function pointer; the library remains alive
    // in DynamicCudaApi for the entire lifetime of every call through it.
    let symbol: Symbol<'_, T> = unsafe { library.get(name) }.map_err(|error| {
        let symbol_name = std::str::from_utf8(&name[..name.len() - 1]).unwrap_or("unknown");
        Error::new(
            ErrorKind::Unsupported,
            format!("NVIDIA driver library is missing required symbol {symbol_name}: {error}"),
        )
    })?;
    Ok(*symbol)
}

fn optional_symbol<T: Copy>(library: &Library, name: &'static [u8]) -> Option<T> {
    // Optional diagnostic symbols do not affect backend functionality.
    unsafe { library.get::<T>(name).ok().map(|symbol| *symbol) }
}
