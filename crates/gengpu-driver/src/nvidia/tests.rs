//! GPU-independent tests for NVIDIA driver and context lifecycle behavior.

use std::ffi::CStr;
use std::sync::{Arc, Mutex};

use gengpu_core::{DeviceId, Error, ErrorKind};

use crate::Driver;

use super::api::{CudaApi, CudaError, CudaLoader, RawDeviceInfo};
use super::driver::DriverState;
use super::NvidiaDriver;

#[derive(Default)]
struct MockState {
    initialize_calls: usize,
    driver_version: i32,
    devices: Vec<RawDeviceInfo>,
    initialize_error: Option<CudaError>,
    driver_version_error: Option<CudaError>,
    device_count_error: Option<CudaError>,
    device_info_error: Option<CudaError>,
    context_create_error: Option<CudaError>,
    context_destroy_error: Option<CudaError>,
    context_create_calls: usize,
    context_destroy_calls: usize,
    destroyed_contexts: Vec<usize>,
    context_stack: Vec<usize>,
    push_calls: usize,
    pop_calls: usize,
    push_error: Option<CudaError>,
    pop_error: Option<CudaError>,
    pop_error_remaining: usize,
    pop_override: Option<usize>,
    synchronize_calls: usize,
    synchronize_error: Option<CudaError>,
    copy_to_device_calls: usize,
    copy_from_device_calls: usize,
    copy_to_device_error: Option<CudaError>,
    copy_from_device_error: Option<CudaError>,
    device_buffers: Vec<(u64, Vec<u8>)>,
    allocation_calls: usize,
    free_calls: usize,
    allocation_error: Option<CudaError>,
    allocation_address: Option<u64>,
    free_error: Option<CudaError>,
    free_error_remaining: usize,
    allocated_addresses: Vec<u64>,
    freed_addresses: Vec<u64>,
    module_load_calls: usize,
    module_unload_calls: usize,
    function_lookup_calls: usize,
    module_load_error: Option<CudaError>,
    module_unload_error: Option<CudaError>,
    function_lookup_error: Option<CudaError>,
    loaded_ptx: Option<String>,
    looked_up_function: Option<String>,
    unloaded_modules: Vec<usize>,
}

#[derive(Clone, Default)]
struct MockCudaApi {
    state: Arc<Mutex<MockState>>,
}

impl CudaApi for MockCudaApi {
    fn initialize(&self) -> std::result::Result<(), CudaError> {
        let mut state = self.state.lock().expect("mock state lock");
        state.initialize_calls += 1;
        match &state.initialize_error {
            Some(error) => Err(error.clone()),
            None => Ok(()),
        }
    }

    fn driver_version(&self) -> std::result::Result<i32, CudaError> {
        let state = self.state.lock().expect("mock state lock");
        match &state.driver_version_error {
            Some(error) => Err(error.clone()),
            None => Ok(state.driver_version),
        }
    }

    fn device_count(&self) -> std::result::Result<i32, CudaError> {
        let state = self.state.lock().expect("mock state lock");
        match &state.device_count_error {
            Some(error) => Err(error.clone()),
            None => Ok(state.devices.len() as i32),
        }
    }

    fn device_info(&self, ordinal: i32) -> std::result::Result<RawDeviceInfo, CudaError> {
        let state = self.state.lock().expect("mock state lock");
        if let Some(error) = &state.device_info_error {
            return Err(error.clone());
        }
        state
            .devices
            .get(ordinal as usize)
            .cloned()
            .ok_or_else(|| CudaError::new(101))
    }

    fn create_context(&self, ordinal: i32) -> std::result::Result<usize, CudaError> {
        let mut state = self.state.lock().expect("mock state lock");
        state.context_create_calls += 1;
        if let Some(error) = &state.context_create_error {
            return Err(error.clone());
        }
        Ok(10_000 + state.context_create_calls - 1 + ordinal as usize)
    }

    fn destroy_context(&self, handle: usize) -> std::result::Result<(), CudaError> {
        let mut state = self.state.lock().expect("mock state lock");
        state.context_destroy_calls += 1;
        if let Some(error) = &state.context_destroy_error {
            return Err(error.clone());
        }
        state.destroyed_contexts.push(handle);
        Ok(())
    }

    fn push_current(&self, handle: usize) -> std::result::Result<(), CudaError> {
        let mut state = self.state.lock().expect("mock state lock");
        state.push_calls += 1;
        if let Some(error) = state.push_error.take() {
            return Err(error);
        }
        state.context_stack.push(handle);
        Ok(())
    }

    fn pop_current(&self) -> std::result::Result<usize, CudaError> {
        let mut state = self.state.lock().expect("mock state lock");
        state.pop_calls += 1;
        if state.pop_error_remaining > 0 {
            state.pop_error_remaining -= 1;
            return Err(state
                .pop_error
                .clone()
                .unwrap_or_else(|| CudaError::new(709)));
        }
        let popped = state.context_stack.pop().unwrap_or(0);
        Ok(state.pop_override.take().unwrap_or(popped))
    }

    fn copy_to_device(&self, address: u64, source: &[u8]) -> std::result::Result<(), CudaError> {
        let mut state = self.state.lock().expect("mock state lock");
        state.copy_to_device_calls += 1;
        if let Some(error) = &state.copy_to_device_error {
            return Err(error.clone());
        }
        let length = u64::try_from(source.len()).expect("test buffer length fits u64");
        let end = address
            .checked_add(length)
            .ok_or_else(|| CudaError::new(700))?;
        let Some((base, buffer)) = state.device_buffers.iter_mut().find(|(base, buffer)| {
            address >= *base && end <= base.saturating_add(u64::try_from(buffer.len()).unwrap())
        }) else {
            return Err(CudaError::new(700));
        };
        let start = usize::try_from(address - *base).expect("test offset fits usize");
        buffer[start..start + source.len()].copy_from_slice(source);
        Ok(())
    }

    fn copy_from_device(
        &self,
        address: u64,
        destination: &mut [u8],
    ) -> std::result::Result<(), CudaError> {
        let mut state = self.state.lock().expect("mock state lock");
        state.copy_from_device_calls += 1;
        if let Some(error) = &state.copy_from_device_error {
            return Err(error.clone());
        }
        let length = u64::try_from(destination.len()).expect("test buffer length fits u64");
        let end = address
            .checked_add(length)
            .ok_or_else(|| CudaError::new(700))?;
        let Some((base, buffer)) = state.device_buffers.iter().find(|(base, buffer)| {
            address >= *base && end <= base.saturating_add(u64::try_from(buffer.len()).unwrap())
        }) else {
            return Err(CudaError::new(700));
        };
        let start = usize::try_from(address - *base).expect("test offset fits usize");
        destination.copy_from_slice(&buffer[start..start + destination.len()]);
        Ok(())
    }

    fn allocate_memory(&self, size_bytes: usize) -> std::result::Result<u64, CudaError> {
        let mut state = self.state.lock().expect("mock state lock");
        state.allocation_calls += 1;
        if let Some(error) = &state.allocation_error {
            return Err(error.clone());
        }
        let address = state.allocation_address.unwrap_or_else(|| {
            0x10000 + (state.allocation_calls as u64 * 0x1000) + size_bytes as u64
        });
        state.allocated_addresses.push(address);
        state.device_buffers.push((address, vec![0; size_bytes]));
        Ok(address)
    }

    fn free_memory(&self, address: u64) -> std::result::Result<(), CudaError> {
        let mut state = self.state.lock().expect("mock state lock");
        state.free_calls += 1;
        if state.free_error_remaining > 0 {
            state.free_error_remaining -= 1;
            return Err(state
                .free_error
                .clone()
                .unwrap_or_else(|| CudaError::new(709)));
        }
        if let Some(error) = &state.free_error {
            return Err(error.clone());
        }
        state.freed_addresses.push(address);
        Ok(())
    }

    fn synchronize(&self) -> std::result::Result<(), CudaError> {
        let mut state = self.state.lock().expect("mock state lock");
        state.synchronize_calls += 1;
        match &state.synchronize_error {
            Some(error) => Err(error.clone()),
            None => Ok(()),
        }
    }

    fn load_module(&self, ptx: &CStr) -> std::result::Result<usize, CudaError> {
        let mut state = self.state.lock().expect("mock state lock");
        state.module_load_calls += 1;
        if let Some(error) = &state.module_load_error {
            return Err(error.clone());
        }
        state.loaded_ptx = Some(ptx.to_string_lossy().into_owned());
        Ok(20_000 + state.module_load_calls)
    }

    fn unload_module(&self, module: usize) -> std::result::Result<(), CudaError> {
        let mut state = self.state.lock().expect("mock state lock");
        state.module_unload_calls += 1;
        if let Some(error) = &state.module_unload_error {
            return Err(error.clone());
        }
        state.unloaded_modules.push(module);
        Ok(())
    }

    fn get_function(
        &self,
        _module: usize,
        name: &CStr,
    ) -> std::result::Result<usize, CudaError> {
        let mut state = self.state.lock().expect("mock state lock");
        state.function_lookup_calls += 1;
        state.looked_up_function = Some(name.to_string_lossy().into_owned());
        if let Some(error) = &state.function_lookup_error {
            return Err(error.clone());
        }
        Ok(30_000 + state.function_lookup_calls)
    }
}

struct MockLoader {
    api: MockCudaApi,
    load_error: bool,
}

impl CudaLoader for MockLoader {
    fn load(&self) -> gengpu_core::Result<Box<dyn CudaApi>> {
        if self.load_error {
            return Err(Error::new(
                ErrorKind::NotFound,
                "mock CUDA library is unavailable",
            ));
        }
        Ok(Box::new(self.api.clone()))
    }
}

fn driver_with_mock(state: MockState) -> (NvidiaDriver, MockCudaApi) {
    let api = MockCudaApi {
        state: Arc::new(Mutex::new(state)),
    };
    let driver = NvidiaDriver {
        loader: Box::new(MockLoader {
            api: api.clone(),
            load_error: false,
        }),
        state: Mutex::new(DriverState::default()),
    };
    (driver, api)
}

fn initialized_driver() -> (NvidiaDriver, MockCudaApi) {
    let (driver, api) = driver_with_mock(MockState {
        devices: vec![RawDeviceInfo {
            name: String::from("GPU A"),
            total_memory_bytes: 1024,
        }],
        ..MockState::default()
    });
    driver.initialize().unwrap();
    (driver, api)
}

#[test]
fn initialization_is_idempotent() {
    let (driver, api) = driver_with_mock(MockState {
        driver_version: 12080,
        ..MockState::default()
    });

    driver.initialize().unwrap();
    driver.initialize().unwrap();

    assert_eq!(api.state.lock().unwrap().initialize_calls, 1);
    assert_eq!(driver.driver_version().unwrap(), 12080);
}

#[test]
fn enumeration_before_initialization_returns_invalid_state() {
    let (driver, _) = driver_with_mock(MockState::default());

    let error = driver.enumerate_devices().unwrap_err();

    assert_eq!(error.kind(), ErrorKind::InvalidState);
}

#[test]
fn zero_devices_is_a_valid_empty_result() {
    let (driver, _) = driver_with_mock(MockState::default());

    driver.initialize().unwrap();

    assert!(driver.enumerate_devices().unwrap().is_empty());
}

#[test]
fn enumeration_returns_device_metadata_and_logical_ids() {
    let (driver, _) = driver_with_mock(MockState {
        devices: vec![
            RawDeviceInfo {
                name: String::from("GPU A"),
                total_memory_bytes: 8 * 1024 * 1024 * 1024,
            },
            RawDeviceInfo {
                name: String::from("GPU B"),
                total_memory_bytes: 16 * 1024 * 1024 * 1024,
            },
        ],
        ..MockState::default()
    });

    driver.initialize().unwrap();
    let devices = driver.enumerate_devices().unwrap();

    assert_eq!(devices.len(), 2);
    assert_eq!(devices[0].id, DeviceId::new(0));
    assert_eq!(devices[0].name, "GPU A");
    assert_eq!(devices[0].total_memory_bytes, 8 * 1024 * 1024 * 1024);
    assert_eq!(devices[1].id, DeviceId::new(1));
    assert_eq!(devices[1].name, "GPU B");
    assert_eq!(devices[1].total_memory_bytes, 16 * 1024 * 1024 * 1024);
}

#[test]
fn initialization_failure_does_not_mark_driver_initialized() {
    let (driver, _) = driver_with_mock(MockState {
        initialize_error: Some(CudaError::new(3)),
        ..MockState::default()
    });

    let error = driver.initialize().unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Driver);
    assert_eq!(
        driver.enumerate_devices().unwrap_err().kind(),
        ErrorKind::InvalidState
    );
}

#[test]
fn device_query_failure_is_translated_to_driver_error() {
    let (driver, _) = driver_with_mock(MockState {
        devices: vec![RawDeviceInfo {
            name: String::from("GPU A"),
            total_memory_bytes: 1024,
        }],
        device_info_error: Some(CudaError {
            code: 101,
            name: Some(String::from("CUDA_ERROR_INVALID_DEVICE")),
            description: Some(String::from("invalid device ordinal")),
        }),
        ..MockState::default()
    });

    driver.initialize().unwrap();
    let error = driver.enumerate_devices().unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Driver);
    assert!(error.message().contains("CUDA error code 101"));
    assert!(error.message().contains("CUDA_ERROR_INVALID_DEVICE"));
    assert!(error.message().contains("invalid device ordinal"));
}

#[test]
fn driver_version_query_failure_does_not_mark_driver_initialized() {
    let (driver, _) = driver_with_mock(MockState {
        driver_version_error: Some(CudaError::new(35)),
        ..MockState::default()
    });

    let error = driver.initialize().unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Driver);
    assert_eq!(
        driver.enumerate_devices().unwrap_err().kind(),
        ErrorKind::InvalidState
    );
}

#[test]
fn driver_version_is_unavailable_before_initialization() {
    let (driver, _) = driver_with_mock(MockState::default());

    assert_eq!(
        driver.driver_version().unwrap_err().kind(),
        ErrorKind::InvalidState
    );
}

#[test]
fn library_load_failure_is_reported_without_panicking() {
    let api = MockCudaApi::default();
    let driver = NvidiaDriver {
        loader: Box::new(MockLoader {
            api,
            load_error: true,
        }),
        state: Mutex::new(DriverState::default()),
    };

    let error = driver.initialize().unwrap_err();

    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(
        driver.enumerate_devices().unwrap_err().kind(),
        ErrorKind::InvalidState
    );
}

#[test]
fn context_creation_requires_initialized_driver() {
    let (driver, _) = driver_with_mock(MockState::default());

    let error = driver
        .create_context(DeviceId::new(0))
        .err()
        .expect("context creation should fail");

    assert_eq!(error.kind(), ErrorKind::InvalidState);
}

#[test]
fn context_creation_rejects_unknown_device() {
    let (driver, api) = driver_with_mock(MockState {
        devices: vec![RawDeviceInfo {
            name: String::from("GPU A"),
            total_memory_bytes: 1024,
        }],
        ..MockState::default()
    });
    driver.initialize().unwrap();

    let error = driver
        .create_context(DeviceId::new(1))
        .err()
        .expect("unknown device should be rejected");

    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(api.state.lock().unwrap().context_create_calls, 0);
}

#[test]
fn context_creation_and_explicit_close_manage_native_lifetime() {
    let (driver, api) = driver_with_mock(MockState {
        devices: vec![RawDeviceInfo {
            name: String::from("GPU A"),
            total_memory_bytes: 1024,
        }],
        ..MockState::default()
    });
    driver.initialize().unwrap();

    let mut context = driver.create_context(DeviceId::new(0)).unwrap();
    assert_eq!(context.device_id(), DeviceId::new(0));
    context.close().unwrap();
    context.close().unwrap();

    let state = api.state.lock().unwrap();
    assert_eq!(state.context_create_calls, 1);
    assert_eq!(state.context_destroy_calls, 1);
    assert_eq!(state.destroyed_contexts, vec![10_000]);
}

#[test]
fn context_retains_backend_after_driver_is_dropped() {
    let (driver, api) = driver_with_mock(MockState {
        devices: vec![RawDeviceInfo {
            name: String::from("GPU A"),
            total_memory_bytes: 1024,
        }],
        ..MockState::default()
    });
    driver.initialize().unwrap();

    let mut context = driver.create_context(DeviceId::new(0)).unwrap();
    drop(driver);
    context.close().unwrap();

    assert_eq!(api.state.lock().unwrap().destroyed_contexts, vec![10_000]);
}

#[test]
fn failed_context_destruction_retains_handle_for_retry() {
    let (driver, api) = driver_with_mock(MockState {
        devices: vec![RawDeviceInfo {
            name: String::from("GPU A"),
            total_memory_bytes: 1024,
        }],
        context_destroy_error: Some(CudaError::new(709)),
        ..MockState::default()
    });
    driver.initialize().unwrap();

    let mut context = driver.create_context(DeviceId::new(0)).unwrap();
    assert_eq!(context.close().unwrap_err().kind(), ErrorKind::Driver);
    api.state.lock().unwrap().context_destroy_error = None;
    context.close().unwrap();

    let state = api.state.lock().unwrap();
    assert_eq!(state.context_destroy_calls, 2);
    assert_eq!(state.destroyed_contexts, vec![10_000]);
}

#[test]
fn dropping_open_context_attempts_best_effort_destruction() {
    let (driver, api) = driver_with_mock(MockState {
        devices: vec![RawDeviceInfo {
            name: String::from("GPU A"),
            total_memory_bytes: 1024,
        }],
        ..MockState::default()
    });
    driver.initialize().unwrap();

    drop(driver.create_context(DeviceId::new(0)).unwrap());

    assert_eq!(api.state.lock().unwrap().destroyed_contexts, vec![10_000]);
}

#[test]
fn context_creation_driver_errors_are_translated() {
    let (driver, _) = driver_with_mock(MockState {
        devices: vec![RawDeviceInfo {
            name: String::from("GPU A"),
            total_memory_bytes: 1024,
        }],
        context_create_error: Some(CudaError {
            code: 999,
            name: Some(String::from("CUDA_ERROR_UNKNOWN")),
            description: Some(String::from("simulated create failure")),
        }),
        ..MockState::default()
    });
    driver.initialize().unwrap();

    let error = driver
        .create_context(DeviceId::new(0))
        .err()
        .expect("context creation should fail");

    assert_eq!(error.kind(), ErrorKind::Driver);
    assert!(error.message().contains("CUDA_ERROR_UNKNOWN"));
    assert!(error.message().contains("simulated create failure"));
}

#[test]
fn make_current_pushes_context_and_finish_pops_it() {
    let (driver, api) = initialized_driver();
    let mut context = driver.create_context(DeviceId::new(0)).unwrap();

    let guard = context.make_current().unwrap();
    assert_eq!(api.state.lock().unwrap().context_stack, vec![10_000]);

    guard.finish().unwrap();

    let state = api.state.lock().unwrap();
    assert!(state.context_stack.is_empty());
    assert_eq!(state.push_calls, 1);
    assert_eq!(state.pop_calls, 1);
}

#[test]
fn nested_context_guards_restore_the_previous_context() {
    let (driver, api) = initialized_driver();
    let mut first = driver.create_context(DeviceId::new(0)).unwrap();
    let mut second = driver.create_context(DeviceId::new(0)).unwrap();

    let first_guard = first.make_current().unwrap();
    let second_guard = second.make_current().unwrap();
    assert_eq!(
        api.state.lock().unwrap().context_stack,
        vec![10_000, 10_001]
    );

    second_guard.finish().unwrap();
    assert_eq!(api.state.lock().unwrap().context_stack, vec![10_000]);

    first_guard.finish().unwrap();
    assert!(api.state.lock().unwrap().context_stack.is_empty());
}

#[test]
fn dropping_current_context_guard_pops_context() {
    let (driver, api) = initialized_driver();
    let mut context = driver.create_context(DeviceId::new(0)).unwrap();

    drop(context.make_current().unwrap());

    assert!(api.state.lock().unwrap().context_stack.is_empty());
}

#[test]
fn failed_guard_finish_retries_pop_during_drop() {
    let (driver, api) = driver_with_mock(MockState {
        devices: vec![RawDeviceInfo {
            name: String::from("GPU A"),
            total_memory_bytes: 1024,
        }],
        pop_error: Some(CudaError::new(709)),
        pop_error_remaining: 1,
        ..MockState::default()
    });
    driver.initialize().unwrap();
    let mut context = driver.create_context(DeviceId::new(0)).unwrap();

    let error = context.make_current().unwrap().finish().unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Driver);
    let state = api.state.lock().unwrap();
    assert!(state.context_stack.is_empty());
    assert_eq!(state.pop_calls, 2);
}

#[test]
fn unexpected_popped_context_is_reported_and_restoration_is_attempted() {
    let (driver, api) = driver_with_mock(MockState {
        devices: vec![RawDeviceInfo {
            name: String::from("GPU A"),
            total_memory_bytes: 1024,
        }],
        pop_override: Some(12_345),
        ..MockState::default()
    });
    driver.initialize().unwrap();
    let mut context = driver.create_context(DeviceId::new(0)).unwrap();

    let error = context.make_current().unwrap().finish().unwrap_err();

    assert_eq!(error.kind(), ErrorKind::InvalidState);
    assert!(error.message().contains("expected 0x2710"));
    let state = api.state.lock().unwrap();
    // One push activates the requested context; the second restores the
    // unexpected context returned by the pop operation.
    assert_eq!(state.push_calls, 2);
    assert_eq!(state.pop_calls, 1);
}

#[test]
fn failed_push_does_not_create_a_guard_or_call_synchronization() {
    let (driver, api) = driver_with_mock(MockState {
        devices: vec![RawDeviceInfo {
            name: String::from("GPU A"),
            total_memory_bytes: 1024,
        }],
        push_error: Some(CudaError::new(709)),
        ..MockState::default()
    });
    driver.initialize().unwrap();
    let mut context = driver.create_context(DeviceId::new(0)).unwrap();

    let error = context
        .make_current()
        .err()
        .expect("context activation should fail");

    assert_eq!(error.kind(), ErrorKind::Driver);
    let state = api.state.lock().unwrap();
    assert!(state.context_stack.is_empty());
    assert_eq!(state.synchronize_calls, 0);
}

#[test]
fn synchronize_temporarily_activates_context_and_restores_previous_context() {
    let (driver, api) = initialized_driver();
    let mut first = driver.create_context(DeviceId::new(0)).unwrap();
    let mut second = driver.create_context(DeviceId::new(0)).unwrap();

    let first_guard = first.make_current().unwrap();
    second.synchronize().unwrap();

    let state = api.state.lock().unwrap();
    assert_eq!(state.context_stack, vec![10_000]);
    assert_eq!(state.synchronize_calls, 1);
    assert_eq!(state.push_calls, 2);
    assert_eq!(state.pop_calls, 1);
    drop(state);

    first_guard.finish().unwrap();
    assert!(api.state.lock().unwrap().context_stack.is_empty());
}

#[test]
fn synchronization_failure_still_restores_previous_context() {
    let (driver, api) = driver_with_mock(MockState {
        devices: vec![RawDeviceInfo {
            name: String::from("GPU A"),
            total_memory_bytes: 1024,
        }],
        synchronize_error: Some(CudaError::new(999)),
        ..MockState::default()
    });
    driver.initialize().unwrap();
    let mut context = driver.create_context(DeviceId::new(0)).unwrap();

    let error = context.synchronize().unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Driver);
    assert!(error.message().contains("999"));
    let state = api.state.lock().unwrap();
    assert!(state.context_stack.is_empty());
    assert_eq!(state.synchronize_calls, 1);
    assert_eq!(state.pop_calls, 1);
}

#[test]
fn synchronization_error_remains_primary_when_restoration_fails() {
    let (driver, api) = driver_with_mock(MockState {
        devices: vec![RawDeviceInfo {
            name: String::from("GPU A"),
            total_memory_bytes: 1024,
        }],
        synchronize_error: Some(CudaError::new(999)),
        pop_error: Some(CudaError::new(709)),
        pop_error_remaining: 1,
        ..MockState::default()
    });
    driver.initialize().unwrap();
    let mut context = driver.create_context(DeviceId::new(0)).unwrap();

    let error = context.synchronize().unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Driver);
    assert!(error.message().contains("999"));
    assert!(error
        .message()
        .contains("restoring the previous current context"));
    let state = api.state.lock().unwrap();
    assert!(state.context_stack.is_empty());
    assert_eq!(state.pop_calls, 2);
}

#[test]
fn synchronization_restoration_failure_is_reported_when_sync_succeeds() {
    let (driver, api) = driver_with_mock(MockState {
        devices: vec![RawDeviceInfo {
            name: String::from("GPU A"),
            total_memory_bytes: 1024,
        }],
        pop_error: Some(CudaError::new(709)),
        pop_error_remaining: 1,
        ..MockState::default()
    });
    driver.initialize().unwrap();
    let mut context = driver.create_context(DeviceId::new(0)).unwrap();

    let error = context.synchronize().unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Driver);
    assert!(error.message().contains("cuCtxPopCurrent_v2"));
    let state = api.state.lock().unwrap();
    assert!(state.context_stack.is_empty());
    assert_eq!(state.synchronize_calls, 1);
    assert_eq!(state.pop_calls, 2);
}

#[test]
fn device_allocation_uses_context_and_returns_metadata() {
    let (driver, api) = initialized_driver();
    let context = driver.create_context(DeviceId::new(0)).unwrap();

    let allocation = context.allocate(4096).unwrap();

    assert_eq!(allocation.device_id(), DeviceId::new(0));
    assert_eq!(allocation.size_bytes(), 4096);
    assert_eq!(
        allocation.device_address().as_u64(),
        0x10000 + 0x1000 + 4096
    );
    let state = api.state.lock().unwrap();
    assert_eq!(state.allocation_calls, 1);
    assert_eq!(state.context_stack, Vec::<usize>::new());
    assert_eq!(state.push_calls, 1);
    assert_eq!(state.pop_calls, 1);
}

#[test]
fn zero_byte_device_allocation_is_rejected_before_cuda_call() {
    let (driver, api) = initialized_driver();
    let context = driver.create_context(DeviceId::new(0)).unwrap();

    let error = context
        .allocate(0)
        .err()
        .expect("zero-byte allocation must fail");

    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
    assert_eq!(api.state.lock().unwrap().allocation_calls, 0);
}

#[test]
fn allocation_failure_restores_previous_context_stack() {
    let (driver, api) = driver_with_mock(MockState {
        devices: vec![RawDeviceInfo {
            name: String::from("GPU A"),
            total_memory_bytes: 1024,
        }],
        allocation_error: Some(CudaError::new(2)),
        ..MockState::default()
    });
    driver.initialize().unwrap();
    let context = driver.create_context(DeviceId::new(0)).unwrap();

    let error = context.allocate(128).err().expect("allocation must fail");

    assert_eq!(error.kind(), ErrorKind::Driver);
    let state = api.state.lock().unwrap();
    assert!(state.context_stack.is_empty());
    assert_eq!(state.allocation_calls, 1);
    assert_eq!(state.pop_calls, 1);
}

#[test]
fn live_allocation_prevents_explicit_context_close_until_deallocated() {
    let (driver, api) = initialized_driver();
    let mut context = driver.create_context(DeviceId::new(0)).unwrap();
    let mut allocation = context.allocate(256).unwrap();

    assert_eq!(context.close().unwrap_err().kind(), ErrorKind::InvalidState);
    allocation.deallocate().unwrap();
    context.close().unwrap();

    let state = api.state.lock().unwrap();
    assert_eq!(state.freed_addresses.len(), 1);
    assert_eq!(state.context_destroy_calls, 1);
}

#[test]
fn allocation_keeps_context_alive_after_context_wrapper_is_dropped() {
    let (driver, api) = initialized_driver();
    let mut allocation = {
        let context = driver.create_context(DeviceId::new(0)).unwrap();
        context.allocate(512).unwrap()
    };

    assert_eq!(api.state.lock().unwrap().context_destroy_calls, 0);
    allocation.deallocate().unwrap();
    assert_eq!(api.state.lock().unwrap().freed_addresses.len(), 1);
    drop(allocation);
    assert_eq!(api.state.lock().unwrap().context_destroy_calls, 1);
}

#[test]
fn failed_device_free_retains_allocation_for_retry() {
    let (driver, api) = driver_with_mock(MockState {
        devices: vec![RawDeviceInfo {
            name: String::from("GPU A"),
            total_memory_bytes: 1024,
        }],
        free_error: Some(CudaError::new(709)),
        free_error_remaining: 1,
        ..MockState::default()
    });
    driver.initialize().unwrap();
    let context = driver.create_context(DeviceId::new(0)).unwrap();
    let mut allocation = context.allocate(128).unwrap();

    assert_eq!(
        allocation.deallocate().unwrap_err().kind(),
        ErrorKind::Driver
    );
    assert!(!allocation.is_deallocated());
    api.state.lock().unwrap().free_error = None;
    allocation.deallocate().unwrap();

    assert!(allocation.is_deallocated());
    let state = api.state.lock().unwrap();
    assert_eq!(state.free_calls, 2);
    assert_eq!(state.freed_addresses.len(), 1);
}

#[test]
fn dropping_device_allocation_frees_memory_and_then_context() {
    let (driver, api) = initialized_driver();
    let allocation = {
        let context = driver.create_context(DeviceId::new(0)).unwrap();
        context.allocate(64).unwrap()
    };

    drop(allocation);

    let state = api.state.lock().unwrap();
    assert_eq!(state.freed_addresses.len(), 1);
    assert_eq!(state.context_destroy_calls, 1);
}

#[test]
fn free_error_is_reported_and_context_stack_is_restored() {
    let (driver, api) = driver_with_mock(MockState {
        devices: vec![RawDeviceInfo {
            name: String::from("GPU A"),
            total_memory_bytes: 1024,
        }],
        free_error: Some(CudaError::new(709)),
        ..MockState::default()
    });
    driver.initialize().unwrap();
    let context = driver.create_context(DeviceId::new(0)).unwrap();
    let mut allocation = context.allocate(64).unwrap();

    let error = allocation.deallocate().unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Driver);
    let state = api.state.lock().unwrap();
    assert!(state.context_stack.is_empty());
    assert_eq!(state.push_calls, 2);
    assert_eq!(state.pop_calls, 2);
}

#[test]
fn successful_free_is_not_retried_when_context_restoration_needs_retry() {
    let (driver, api) = driver_with_mock(MockState {
        devices: vec![RawDeviceInfo {
            name: String::from("GPU A"),
            total_memory_bytes: 1024,
        }],
        ..MockState::default()
    });
    driver.initialize().unwrap();
    let context = driver.create_context(DeviceId::new(0)).unwrap();
    let mut allocation = context.allocate(96).unwrap();
    {
        let mut state = api.state.lock().unwrap();
        state.pop_error = Some(CudaError::new(709));
        state.pop_error_remaining = 1;
    }

    let error = allocation.deallocate().unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Driver);
    assert!(allocation.is_deallocated());
    let state = api.state.lock().unwrap();
    assert_eq!(state.free_calls, 1);
    assert_eq!(state.freed_addresses.len(), 1);
    assert!(state.context_stack.is_empty());
    assert_eq!(state.pop_calls, 3);
}

#[test]
fn allocation_is_freed_when_initial_context_restoration_requires_retry() {
    let (driver, api) = driver_with_mock(MockState {
        devices: vec![RawDeviceInfo {
            name: String::from("GPU A"),
            total_memory_bytes: 1024,
        }],
        pop_error: Some(CudaError::new(709)),
        pop_error_remaining: 1,
        ..MockState::default()
    });
    driver.initialize().unwrap();
    let context = driver.create_context(DeviceId::new(0)).unwrap();

    let error = context
        .allocate(80)
        .err()
        .expect("allocation must report the initial restoration failure");

    assert_eq!(error.kind(), ErrorKind::Driver);
    let state = api.state.lock().unwrap();
    assert_eq!(state.allocation_calls, 1);
    assert_eq!(state.free_calls, 1);
    assert_eq!(state.freed_addresses.len(), 1);
    assert!(state.context_stack.is_empty());
    assert_eq!(state.pop_calls, 3);
}

#[test]
fn dropped_allocation_transfers_failed_free_to_context_for_close_retry() {
    let (driver, api) = initialized_driver();
    let mut context = driver.create_context(DeviceId::new(0)).unwrap();
    let allocation = context.allocate(128).unwrap();
    let address = allocation.device_address().as_u64();

    api.state.lock().unwrap().free_error = Some(CudaError::new(709));
    drop(allocation);

    {
        let state = api.state.lock().unwrap();
        assert_eq!(state.free_calls, 1);
        assert!(state.freed_addresses.is_empty());
        assert_eq!(state.context_destroy_calls, 0);
    }

    api.state.lock().unwrap().free_error = None;
    context.close().unwrap();

    let state = api.state.lock().unwrap();
    assert_eq!(state.free_calls, 2);
    assert_eq!(state.freed_addresses, vec![address]);
    assert_eq!(state.context_destroy_calls, 1);
}

#[test]
fn context_close_retains_orphan_after_free_failure_for_later_retry() {
    let (driver, api) = initialized_driver();
    let mut context = driver.create_context(DeviceId::new(0)).unwrap();
    let allocation = context.allocate(256).unwrap();

    api.state.lock().unwrap().free_error = Some(CudaError::new(709));
    drop(allocation);

    assert_eq!(context.close().unwrap_err().kind(), ErrorKind::Driver);
    assert_eq!(api.state.lock().unwrap().context_destroy_calls, 0);

    api.state.lock().unwrap().free_error = None;
    context.close().unwrap();

    let state = api.state.lock().unwrap();
    assert_eq!(state.free_calls, 3);
    assert_eq!(state.freed_addresses.len(), 1);
    assert_eq!(state.context_destroy_calls, 1);
}

#[test]
fn dropping_context_releases_orphaned_allocations_before_destroying_context() {
    let (driver, api) = initialized_driver();
    let context = {
        let context = driver.create_context(DeviceId::new(0)).unwrap();
        let allocation = context.allocate(512).unwrap();
        api.state.lock().unwrap().free_error = Some(CudaError::new(709));
        drop(allocation);
        context
    };

    drop(context);

    let state = api.state.lock().unwrap();
    // The context Drop path retries the orphan free, then destroys the
    // context as the final CUDA resource-reclamation boundary.
    assert_eq!(state.free_calls, 2);
    assert!(state.freed_addresses.is_empty());
    assert_eq!(state.context_destroy_calls, 1);
    drop(state);
}

#[test]
fn failed_cleanup_of_unreturned_allocation_is_tracked_by_context() {
    let (driver, api) = driver_with_mock(MockState {
        devices: vec![RawDeviceInfo {
            name: String::from("GPU A"),
            total_memory_bytes: 1024,
        }],
        pop_error: Some(CudaError::new(709)),
        pop_error_remaining: 1,
        free_error: Some(CudaError::new(2)),
        ..MockState::default()
    });
    driver.initialize().unwrap();
    let mut context = driver.create_context(DeviceId::new(0)).unwrap();

    let error = context
        .allocate(80)
        .err()
        .expect("allocation should report failed restoration and cleanup");
    assert_eq!(error.kind(), ErrorKind::Driver);

    {
        let state = api.state.lock().unwrap();
        assert_eq!(state.allocation_calls, 1);
        assert_eq!(state.free_calls, 1);
        assert!(state.freed_addresses.is_empty());
        assert_eq!(state.context_destroy_calls, 0);
    }

    api.state.lock().unwrap().free_error = None;
    context.close().unwrap();

    let state = api.state.lock().unwrap();
    assert_eq!(state.free_calls, 2);
    assert_eq!(state.freed_addresses.len(), 1);
    assert_eq!(state.context_destroy_calls, 1);
}

#[test]
fn host_device_transfers_round_trip_bytes_with_offsets() {
    let (driver, api) = initialized_driver();
    let context = driver.create_context(DeviceId::new(0)).unwrap();
    let mut allocation = context.allocate(16).unwrap();

    context
        .copy_to_device(&mut allocation, 3, &[10, 20, 30, 40])
        .unwrap();
    let mut received = [0_u8; 4];
    context
        .copy_from_device(&mut allocation, 3, &mut received)
        .unwrap();

    assert_eq!(received, [10, 20, 30, 40]);
    let state = api.state.lock().unwrap();
    assert_eq!(state.copy_to_device_calls, 1);
    assert_eq!(state.copy_from_device_calls, 1);
    assert_eq!(state.synchronize_calls, 2);
    assert_eq!(state.push_calls, 3); // allocation plus two transfers
    assert_eq!(state.pop_calls, 3);
    assert!(state.context_stack.is_empty());
}

#[test]
fn transfer_rejects_allocation_from_another_context_on_the_same_device() {
    let (driver, api) = initialized_driver();
    let first_context = driver.create_context(DeviceId::new(0)).unwrap();
    let second_context = driver.create_context(DeviceId::new(0)).unwrap();
    let mut allocation = first_context.allocate(16).unwrap();

    let error = second_context
        .copy_to_device(&mut allocation, 0, &[1, 2, 3])
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
    let state = api.state.lock().unwrap();
    assert_eq!(state.copy_to_device_calls, 0);
    assert_eq!(state.synchronize_calls, 0);
    assert_eq!(state.push_calls, 1); // allocation only
}

#[test]
fn transfer_rejects_out_of_bounds_and_overflow_before_cuda_copy() {
    let (driver, api) = initialized_driver();
    let context = driver.create_context(DeviceId::new(0)).unwrap();
    let mut allocation = context.allocate(8).unwrap();

    let out_of_bounds = context
        .copy_to_device(&mut allocation, 7, &[1, 2])
        .unwrap_err();
    let overflow = context
        .copy_to_device(&mut allocation, usize::MAX, &[1])
        .unwrap_err();

    assert_eq!(out_of_bounds.kind(), ErrorKind::InvalidArgument);
    assert_eq!(overflow.kind(), ErrorKind::InvalidArgument);
    let state = api.state.lock().unwrap();
    assert_eq!(state.copy_to_device_calls, 0);
    assert_eq!(state.synchronize_calls, 0);
    assert_eq!(state.push_calls, 1); // allocation only
}

#[test]
fn zero_length_transfers_validate_ranges_without_cuda_calls() {
    let (driver, api) = initialized_driver();
    let context = driver.create_context(DeviceId::new(0)).unwrap();
    let mut allocation = context.allocate(8).unwrap();

    context.copy_to_device(&mut allocation, 8, &[]).unwrap();
    let mut empty = [];
    context
        .copy_from_device(&mut allocation, 8, &mut empty)
        .unwrap();

    let invalid = context.copy_to_device(&mut allocation, 9, &[]).unwrap_err();
    assert_eq!(invalid.kind(), ErrorKind::InvalidArgument);

    let state = api.state.lock().unwrap();
    assert_eq!(state.copy_to_device_calls, 0);
    assert_eq!(state.copy_from_device_calls, 0);
    assert_eq!(state.synchronize_calls, 0);
    assert_eq!(state.push_calls, 1); // allocation only
    assert_eq!(state.pop_calls, 1);
}

#[test]
fn failed_native_copy_is_reported_and_context_is_restored_without_sync() {
    let (driver, api) = driver_with_mock(MockState {
        devices: vec![RawDeviceInfo {
            name: String::from("GPU A"),
            total_memory_bytes: 1024,
        }],
        copy_to_device_error: Some(CudaError::new(999)),
        ..MockState::default()
    });
    driver.initialize().unwrap();
    let context = driver.create_context(DeviceId::new(0)).unwrap();
    let mut allocation = context.allocate(8).unwrap();

    let error = context
        .copy_to_device(&mut allocation, 0, &[1, 2, 3])
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Driver);
    assert!(error.message().contains("cuMemcpyHtoD_v2"));
    let state = api.state.lock().unwrap();
    assert_eq!(state.copy_to_device_calls, 1);
    assert_eq!(state.synchronize_calls, 0);
    assert!(state.context_stack.is_empty());
    assert_eq!(state.pop_calls, 2); // allocation and failed transfer restoration
}

#[test]
fn transfer_synchronization_failure_is_reported_after_copy_and_context_restore() {
    let (driver, api) = driver_with_mock(MockState {
        devices: vec![RawDeviceInfo {
            name: String::from("GPU A"),
            total_memory_bytes: 1024,
        }],
        synchronize_error: Some(CudaError::new(999)),
        ..MockState::default()
    });
    driver.initialize().unwrap();
    let context = driver.create_context(DeviceId::new(0)).unwrap();
    let mut allocation = context.allocate(8).unwrap();

    let error = context
        .copy_from_device(&mut allocation, 0, &mut [0_u8; 4])
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Driver);
    assert!(error.message().contains("cuCtxSynchronize"));
    let state = api.state.lock().unwrap();
    assert_eq!(state.copy_from_device_calls, 1);
    assert_eq!(state.synchronize_calls, 1);
    assert!(state.context_stack.is_empty());
    assert_eq!(state.pop_calls, 2);
}

#[test]
fn transfer_reports_context_restoration_failure_after_retry() {
    let (driver, api) = initialized_driver();
    let context = driver.create_context(DeviceId::new(0)).unwrap();
    let mut allocation = context.allocate(8).unwrap();
    {
        let mut state = api.state.lock().unwrap();
        state.pop_error = Some(CudaError::new(709));
        state.pop_error_remaining = 1;
    }

    let error = context
        .copy_to_device(&mut allocation, 0, &[1, 2, 3, 4])
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Driver);
    assert!(error.message().contains("cuCtxPopCurrent_v2"));
    let state = api.state.lock().unwrap();
    assert_eq!(state.copy_to_device_calls, 1);
    assert_eq!(state.synchronize_calls, 1);
    assert_eq!(state.pop_calls, 3); // allocation pop plus failed and retry pops
    assert!(state.context_stack.is_empty());
}

#[test]
fn deallocated_memory_cannot_be_used_for_transfers() {
    let (driver, api) = initialized_driver();
    let context = driver.create_context(DeviceId::new(0)).unwrap();
    let mut allocation = context.allocate(8).unwrap();
    allocation.deallocate().unwrap();

    let error = context
        .copy_to_device(&mut allocation, 0, &[1])
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::InvalidState);
    let state = api.state.lock().unwrap();
    assert_eq!(state.copy_to_device_calls, 0);
    assert_eq!(state.synchronize_calls, 0);
}

#[test]
fn transfer_rejects_device_address_overflow_before_cuda_copy() {
    let (driver, api) = driver_with_mock(MockState {
        devices: vec![RawDeviceInfo {
            name: String::from("GPU A"),
            total_memory_bytes: 1024,
        }],
        allocation_address: Some(u64::MAX - 3),
        ..MockState::default()
    });
    driver.initialize().unwrap();
    let context = driver.create_context(DeviceId::new(0)).unwrap();
    let mut allocation = context.allocate(8).unwrap();

    let error = context
        .copy_to_device(&mut allocation, 4, &[1])
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
    let state = api.state.lock().unwrap();
    assert_eq!(state.copy_to_device_calls, 0);
    assert_eq!(state.synchronize_calls, 0);
}

#[test]
fn ptx_module_loads_inside_and_restores_owning_context() {
    let (driver, api) = initialized_driver();
    let context = driver.create_context(DeviceId::new(0)).unwrap();
    let mut module = context
        .load_ptx(".version 7.0\n.target sm_50")
        .unwrap();

    {
        let state = api.state.lock().unwrap();
        assert_eq!(state.module_load_calls, 1);
        assert_eq!(state.loaded_ptx.as_deref(), Some(".version 7.0\n.target sm_50"));
        assert!(state.context_stack.is_empty());
        assert_eq!(state.push_calls, 1);
        assert_eq!(state.pop_calls, 1);
    }

    module.unload().unwrap();
    assert_eq!(api.state.lock().unwrap().module_unload_calls, 1);
}

#[test]
fn ptx_module_load_failure_restores_context_and_translates_error() {
    let (driver, api) = driver_with_mock(MockState {
        devices: vec![RawDeviceInfo {
            name: String::from("GPU A"),
            total_memory_bytes: 1024,
        }],
        module_load_error: Some(CudaError::new(200)),
        ..MockState::default()
    });
    driver.initialize().unwrap();
    let context = driver.create_context(DeviceId::new(0)).unwrap();

    let error = context
        .load_ptx(".version 7.0\n.target sm_50")
        .err()
        .expect("module load failure must be returned");

    assert_eq!(error.kind(), ErrorKind::Driver);
    assert!(error.message().contains("cuModuleLoadData"));
    let state = api.state.lock().unwrap();
    assert_eq!(state.module_load_calls, 1);
    assert_eq!(state.module_unload_calls, 0);
    assert!(state.context_stack.is_empty());
    assert_eq!(state.push_calls, 1);
    assert_eq!(state.pop_calls, 1);
}

#[test]
fn ptx_module_is_cleaned_up_when_context_restoration_fails_after_load() {
    let (driver, api) = initialized_driver();
    let context = driver.create_context(DeviceId::new(0)).unwrap();
    {
        let mut state = api.state.lock().unwrap();
        state.pop_error = Some(CudaError::new(709));
        state.pop_error_remaining = 1;
    }

    let error = context
        .load_ptx(".version 7.0\n.target sm_50")
        .err()
        .expect("context restoration failure must be returned");

    assert_eq!(error.kind(), ErrorKind::Driver);
    let state = api.state.lock().unwrap();
    assert_eq!(state.module_load_calls, 1);
    assert_eq!(state.module_unload_calls, 1);
    assert_eq!(state.push_calls, 2);
    assert_eq!(state.pop_calls, 3);
    assert!(state.context_stack.is_empty());
}

#[test]
fn module_function_lookup_returns_an_opaque_backend_handle() {
    let (driver, api) = initialized_driver();
    let context = driver.create_context(DeviceId::new(0)).unwrap();
    let mut module = context
        .load_ptx(".version 7.0\n.target sm_50")
        .unwrap();

    let function = module.get_function("kernel_entry").unwrap();
    assert_eq!(
        function
            .backend_handle::<super::module::CudaFunctionHandle>()
            .map(|handle| handle.0),
        Some(30_001)
    );
    assert_eq!(
        api.state.lock().unwrap().looked_up_function.as_deref(),
        Some("kernel_entry")
    );
    drop(function);

    module.unload().unwrap();
}

#[test]
fn module_function_lookup_rejects_invalid_names_before_cuda_call() {
    let (driver, api) = initialized_driver();
    let context = driver.create_context(DeviceId::new(0)).unwrap();
    let module = context
        .load_ptx(".version 7.0\n.target sm_50")
        .unwrap();

    let empty = module
        .get_function("")
        .err()
        .expect("empty function name must be rejected");
    let embedded_nul = module
        .get_function("kernel\0suffix")
        .err()
        .expect("embedded NUL must be rejected");

    assert_eq!(empty.kind(), ErrorKind::InvalidArgument);
    assert_eq!(embedded_nul.kind(), ErrorKind::InvalidArgument);
    assert_eq!(api.state.lock().unwrap().function_lookup_calls, 0);
}

#[test]
fn missing_module_function_maps_cuda_not_found_to_gen_gpu_not_found() {
    let (driver, api) = driver_with_mock(MockState {
        devices: vec![RawDeviceInfo {
            name: String::from("GPU A"),
            total_memory_bytes: 1024,
        }],
        function_lookup_error: Some(CudaError::new(500)),
        ..MockState::default()
    });
    driver.initialize().unwrap();
    let context = driver.create_context(DeviceId::new(0)).unwrap();
    let module = context
        .load_ptx(".version 7.0\n.target sm_50")
        .unwrap();

    let error = module
        .get_function("missing_kernel")
        .err()
        .expect("missing kernel function must return NotFound");

    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(error.message().contains("missing_kernel"));
    let state = api.state.lock().unwrap();
    assert_eq!(state.function_lookup_calls, 1);
    assert!(state.context_stack.is_empty());
}

#[test]
fn module_function_lookup_preserves_driver_errors_and_restores_context() {
    let (driver, api) = driver_with_mock(MockState {
        devices: vec![RawDeviceInfo {
            name: String::from("GPU A"),
            total_memory_bytes: 1024,
        }],
        function_lookup_error: Some(CudaError::new(700)),
        ..MockState::default()
    });
    driver.initialize().unwrap();
    let context = driver.create_context(DeviceId::new(0)).unwrap();
    let module = context
        .load_ptx(".version 7.0\n.target sm_50")
        .unwrap();

    let error = module
        .get_function("kernel_entry")
        .err()
        .expect("native function lookup failure must be returned");

    assert_eq!(error.kind(), ErrorKind::Driver);
    assert!(error.message().contains("cuModuleGetFunction"));
    let state = api.state.lock().unwrap();
    assert_eq!(state.function_lookup_calls, 1);
    assert!(state.context_stack.is_empty());
    assert_eq!(state.push_calls, 2);
    assert_eq!(state.pop_calls, 2);
}

#[test]
fn function_lookup_reports_context_restoration_failure_after_retry() {
    let (driver, api) = initialized_driver();
    let context = driver.create_context(DeviceId::new(0)).unwrap();
    let module = context
        .load_ptx(".version 7.0\n.target sm_50")
        .unwrap();
    {
        let mut state = api.state.lock().unwrap();
        state.pop_error = Some(CudaError::new(709));
        state.pop_error_remaining = 1;
    }

    let error = module
        .get_function("kernel_entry")
        .err()
        .expect("restoration failure must be returned");

    assert_eq!(error.kind(), ErrorKind::Driver);
    assert!(error.message().contains("cuCtxPopCurrent_v2"));
    let state = api.state.lock().unwrap();
    assert_eq!(state.function_lookup_calls, 1);
    assert!(state.context_stack.is_empty());
    assert_eq!(state.push_calls, 2);
    assert_eq!(state.pop_calls, 3);
}

#[test]
fn ptx_module_load_does_not_call_cuda_when_context_push_fails() {
    let (driver, api) = driver_with_mock(MockState {
        devices: vec![RawDeviceInfo {
            name: String::from("GPU A"),
            total_memory_bytes: 1024,
        }],
        push_error: Some(CudaError::new(709)),
        ..MockState::default()
    });
    driver.initialize().unwrap();
    let context = driver.create_context(DeviceId::new(0)).unwrap();

    let error = context
        .load_ptx(".version 7.0\n.target sm_50")
        .err()
        .expect("context push failure must be returned");

    assert_eq!(error.kind(), ErrorKind::Driver);
    let state = api.state.lock().unwrap();
    assert_eq!(state.module_load_calls, 0);
    assert_eq!(state.module_unload_calls, 0);
    assert_eq!(state.push_calls, 1);
    assert_eq!(state.pop_calls, 0);
}

#[test]
fn missing_function_error_remains_primary_when_context_restoration_fails() {
    let (driver, api) = driver_with_mock(MockState {
        devices: vec![RawDeviceInfo {
            name: String::from("GPU A"),
            total_memory_bytes: 1024,
        }],
        function_lookup_error: Some(CudaError::new(500)),
        ..MockState::default()
    });
    driver.initialize().unwrap();
    let context = driver.create_context(DeviceId::new(0)).unwrap();
    let module = context
        .load_ptx(".version 7.0\n.target sm_50")
        .unwrap();
    {
        let mut state = api.state.lock().unwrap();
        state.pop_error = Some(CudaError::new(709));
        state.pop_error_remaining = 1;
    }

    let error = module
        .get_function("missing_kernel")
        .err()
        .expect("missing function must remain an error");

    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(error.message().contains("missing_kernel"));
    assert!(error.message().contains("cuCtxPopCurrent_v2"));
    let state = api.state.lock().unwrap();
    assert_eq!(state.function_lookup_calls, 1);
    assert!(state.context_stack.is_empty());
}
