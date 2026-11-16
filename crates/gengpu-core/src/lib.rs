//! Shared foundational types for the GenGPU project.
//!
//! This crate contains backend-independent contracts shared by the driver,
//! runtime, and compiler components.

mod device;
mod error;

pub use device::{DeviceId, DeviceInfo};
pub use error::{Error, ErrorKind, Result};
