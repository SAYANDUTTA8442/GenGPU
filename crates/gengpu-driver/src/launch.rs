//! Validated backend-neutral kernel launch dimensions and configuration.

use gengpu_core::{Error, ErrorKind, Result};

/// Three-dimensional unsigned launch dimensions.
///
/// Every dimension must be non-zero. Device- and architecture-specific
/// maximums are intentionally left to the backend/driver.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Dim3 {
    x: u32,
    y: u32,
    z: u32,
}

impl Dim3 {
    /// Creates dimensions after rejecting zero-valued components.
    pub fn new(x: u32, y: u32, z: u32) -> Result<Self> {
        if x == 0 || y == 0 || z == 0 {
            return Err(Error::new(
                ErrorKind::InvalidArgument,
                "launch dimensions must all be non-zero",
            ));
        }

        Ok(Self { x, y, z })
    }

    /// Returns the x dimension.
    pub fn x(self) -> u32 {
        self.x
    }

    /// Returns the y dimension.
    pub fn y(self) -> u32 {
        self.y
    }

    /// Returns the z dimension.
    pub fn z(self) -> u32 {
        self.z
    }

    /// Returns the checked product of all three dimensions.
    pub fn checked_volume(self) -> Result<u64> {
        u64::from(self.x)
            .checked_mul(u64::from(self.y))
            .and_then(|xy| xy.checked_mul(u64::from(self.z)))
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidArgument,
                    "launch dimension product overflows u64",
                )
            })
    }
}

/// Validated grid, block, and dynamic shared-memory launch configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LaunchConfig {
    grid: Dim3,
    block: Dim3,
    shared_mem_bytes: u32,
}

impl LaunchConfig {
    /// Creates a launch configuration.
    ///
    /// Grid and block dimensions must be non-zero. The block-thread product
    /// is checked for overflow. Device-specific launch limits are validated
    /// by the CUDA driver because this type does not assume hardware limits.
    pub fn new(grid: Dim3, block: Dim3, shared_mem_bytes: u32) -> Result<Self> {
        block.checked_volume()?;

        Ok(Self {
            grid,
            block,
            shared_mem_bytes,
        })
    }

    /// Returns the grid dimensions.
    pub fn grid(self) -> Dim3 {
        self.grid
    }

    /// Returns the block dimensions.
    pub fn block(self) -> Dim3 {
        self.block
    }

    /// Returns dynamic shared memory in bytes.
    pub fn shared_mem_bytes(self) -> u32 {
        self.shared_mem_bytes
    }

    /// Returns the checked number of threads per block.
    pub fn threads_per_block(self) -> Result<u64> {
        self.block.checked_volume()
    }
}

#[cfg(test)]
mod tests {
    use super::{Dim3, LaunchConfig};
    use gengpu_core::ErrorKind;

    #[test]
    fn accepts_valid_dimensions() {
        let dimensions = Dim3::new(16, 8, 2).unwrap();

        assert_eq!(dimensions.x(), 16);
        assert_eq!(dimensions.y(), 8);
        assert_eq!(dimensions.z(), 2);
        assert_eq!(dimensions.checked_volume().unwrap(), 256);
    }

    #[test]
    fn rejects_zero_dimension_components() {
        for dimensions in [(0, 1, 1), (1, 0, 1), (1, 1, 0)] {
            let error = Dim3::new(dimensions.0, dimensions.1, dimensions.2).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::InvalidArgument);
        }
    }

    #[test]
    fn rejects_dimension_product_overflow() {
        let dimensions = Dim3 {
            x: u32::MAX,
            y: u32::MAX,
            z: u32::MAX,
        };

        assert_eq!(
            dimensions.checked_volume().unwrap_err().kind(),
            ErrorKind::InvalidArgument
        );
    }

    #[test]
    fn validates_block_thread_product_when_building_config() {
        let grid = Dim3::new(1, 1, 1).unwrap();
        let block = Dim3 {
            x: u32::MAX,
            y: u32::MAX,
            z: u32::MAX,
        };

        assert_eq!(
            LaunchConfig::new(grid, block, 0).unwrap_err().kind(),
            ErrorKind::InvalidArgument
        );
    }

    #[test]
    fn stores_shared_memory_size_in_bytes() {
        let config = LaunchConfig::new(
            Dim3::new(2, 3, 4).unwrap(),
            Dim3::new(8, 4, 1).unwrap(),
            2048,
        )
        .unwrap();

        assert_eq!(config.grid(), Dim3::new(2, 3, 4).unwrap());
        assert_eq!(config.block(), Dim3::new(8, 4, 1).unwrap());
        assert_eq!(config.shared_mem_bytes(), 2048);
        assert_eq!(config.threads_per_block().unwrap(), 32);
    }
}
