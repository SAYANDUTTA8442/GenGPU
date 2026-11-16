//! Backend-independent GPU device metadata and identifiers.

/// Process-local logical identifier for a GPU device.
///
/// A `DeviceId` identifies a device within the current GenGPU process. It is
/// not a persistent hardware identifier and must not be assumed to remain
/// stable across process runs or device-enumeration changes.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DeviceId(u32);

impl DeviceId {
    /// Creates a logical device identifier from its enumeration index.
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    /// Returns the device's enumeration index.
    pub const fn index(self) -> u32 {
        self.0
    }
}

/// Backend-independent information about a GPU device.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceInfo {
    /// Process-local logical identifier for this device.
    pub id: DeviceId,
    /// Human-readable device name reported by the backend.
    pub name: String,
    /// Total device memory, in bytes.
    pub total_memory_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::{DeviceId, DeviceInfo};

    #[test]
    fn device_id_preserves_its_index() {
        let id = DeviceId::new(7);

        assert_eq!(id.index(), 7);
    }

    #[test]
    fn device_ids_support_value_equality_and_hashing() {
        use std::collections::HashSet;

        let first = DeviceId::new(2);
        let second = DeviceId::new(2);
        let third = DeviceId::new(3);
        let mut ids = HashSet::new();
        ids.insert(first);

        assert_eq!(first, second);
        assert_ne!(first, third);
        assert!(ids.contains(&second));
        assert!(!ids.contains(&third));
    }

    #[test]
    fn device_info_stores_backend_independent_metadata() {
        let info = DeviceInfo {
            id: DeviceId::new(0),
            name: String::from("Example GPU"),
            total_memory_bytes: 8 * 1024 * 1024 * 1024,
        };

        assert_eq!(info.id.index(), 0);
        assert_eq!(info.name, "Example GPU");
        assert_eq!(info.total_memory_bytes, 8 * 1024 * 1024 * 1024);
    }
}
