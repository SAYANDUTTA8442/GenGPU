//! Runtime loading and private CUDA Driver API function table.

use std::ffi::{c_char, c_int, c_uint, CStr};
use std::ptr;

use gengpu_core::{Error, ErrorKind, Result};
use libloading::{Library, Symbol};

use super::api::{CudaApi, CudaError, CudaLoader, RawDeviceInfo};

type CuResult = c_int;
type CuDevice = c_int;

type CuInit = unsafe extern "C" fn(flags: c_uint) -> CuResult;
type CuDriverGetVersion = unsafe extern "C" fn(driver_version: *mut c_int) -> CuResult;
type CuDeviceGetCount = unsafe extern "C" fn(count: *mut c_int) -> CuResult;
type CuDeviceGet = unsafe extern "C" fn(device: *mut CuDevice, ordinal: c_int) -> CuResult;
type CuDeviceGetName =
    unsafe extern "C" fn(name: *mut c_char, length: c_int, device: CuDevice) -> CuResult;
type CuDeviceTotalMem = unsafe extern "C" fn(bytes: *mut usize, device: CuDevice) -> CuResult;
type CuGetErrorName =
    unsafe extern "C" fn(error: CuResult, name: *mut *const c_char) -> CuResult;
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
                error.name = Some(unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned());
            }
        }

        if let Some(get_error_string) = self.cu_get_error_string {
            let mut description = ptr::null();
            // CUDA writes a pointer to a driver-owned, null-terminated string.
            let result = unsafe { get_error_string(code, &mut description) };
            if result == 0 && !description.is_null() {
                error.description =
                    Some(unsafe { CStr::from_ptr(description) }.to_string_lossy().into_owned());
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
            (self.cu_device_get_name)(
                name.as_mut_ptr(),
                CUDA_DEVICE_NAME_BUFFER_LENGTH,
                device,
            )
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
