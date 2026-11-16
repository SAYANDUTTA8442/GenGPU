//! NVIDIA CUDA Driver API backend boundary.
//!
//! The concrete implementation and dynamic library loading are intentionally
//! deferred to a later subphase. NVIDIA-specific handles, function pointers,
//! and native error codes will remain private to this backend; callers will
//! interact through the crate-level Driver contract.
