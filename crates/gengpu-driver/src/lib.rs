//! Low-level GPU driver integration for GenGPU.
//!
//! This crate defines a backend-neutral driver contract and contains
//! backend-specific implementations. Native backend handles and API types
//! must remain behind the crate's public abstractions.

mod allocation;
mod argument;
mod context;
mod driver;
mod launch;
mod module;

/// NVIDIA-specific driver backend.
pub mod nvidia;

pub use allocation::{DeviceAddress, DeviceAllocation};
pub use argument::{KernelArg, SupportedScalar};
pub use context::{Context, CurrentContextGuard};
pub use driver::Driver;
pub use launch::{Dim3, LaunchConfig};
pub use module::{KernelFunction, Module};
