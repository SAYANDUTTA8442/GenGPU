//! GPU-independent tests for NVIDIA driver lifecycle and error handling.

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
    assert_eq!(driver.enumerate_devices().unwrap_err().kind(), ErrorKind::InvalidState);
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
    assert_eq!(driver.enumerate_devices().unwrap_err().kind(), ErrorKind::InvalidState);
}

#[test]
fn driver_version_is_unavailable_before_initialization() {
    let (driver, _) = driver_with_mock(MockState::default());

    assert_eq!(driver.driver_version().unwrap_err().kind(), ErrorKind::InvalidState);
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
    assert_eq!(driver.enumerate_devices().unwrap_err().kind(), ErrorKind::InvalidState);
}
