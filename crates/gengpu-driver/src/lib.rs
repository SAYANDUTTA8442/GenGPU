//! Low-level GPU driver integration for GenGPU.
//!
//! This crate defines a backend-neutral driver contract and contains
//! backend-specific implementations. Native backend handles and API types
//! must remain behind the crate's public abstractions.

mod context;
mod driver;

/// NVIDIA-specific driver backend.
pub mod nvidia;

pub use context::{Context, CurrentContextGuard};
pub use driver::Driver;
