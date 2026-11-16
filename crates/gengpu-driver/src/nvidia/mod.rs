//! NVIDIA CUDA Driver API backend.
//!
//! The backend loads the NVIDIA driver library at runtime. CUDA-native types,
//! handles, function pointers, and unsafe FFI remain private to this module.

mod api;
mod driver;
mod loader;

pub use driver::NvidiaDriver;

#[cfg(test)]
mod tests;
