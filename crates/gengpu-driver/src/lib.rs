//! Low-level GPU driver integration for GenGPU.
//!
//! This crate defines a backend-neutral driver contract and contains
//! backend-specific implementations. Native backend handles and API types
//! must remain behind this crate's public abstractions.

mod driver;

/// NVIDIA-specific driver backend.
pub mod nvidia;

pub use driver::Driver;
