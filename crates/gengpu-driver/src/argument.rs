//! Backend-neutral kernel argument representations.

use std::ffi::c_void;
use std::marker::PhantomData;

use gengpu_core::{Error, ErrorKind, Result};

use crate::DeviceAllocation;

/// Sealed set of scalar types supported by the initial kernel argument ABI.
///
/// This trait is intentionally sealed. Implementations are limited to the
/// primitive integer and floating-point types documented below.
///
/// Supported types: `u8`, `i8`, `u16`, `i16`, `u32`, `i32`,
/// `u64`, `i64`, `f32`, and `f64`.
///
/// External implementations are rejected by the private sealing supertrait:
///
/// ```compile_fail
/// use gengpu_driver::SupportedScalar;
///
/// struct MyScalar(u32);
///
/// impl SupportedScalar for MyScalar {}
/// ```
pub trait SupportedScalar: private::Sealed + Copy {}

mod private {
    pub trait Sealed {}

    macro_rules! impl_supported_scalar {
        ($($ty:ty),+ $(,)?) => {
            $(
                impl Sealed for $ty {}
                impl super::SupportedScalar for $ty {}
            )+
        };
    }

    impl_supported_scalar!(u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);
}

enum KernelArgValue<'a> {
    Scalar(*mut c_void),
    DevicePointer {
        allocation: &'a mut DeviceAllocation,
        address: u64,
    },
}

/// One kernel argument borrowed for the duration of a launch.
///
/// Construct scalar arguments with [`KernelArg::scalar`] and device-pointer
/// arguments with [`KernelArg::device_ptr`]. The caller is responsible for
/// matching argument order and representation to the PTX parameter
/// declarations; GenGPU does not yet have signature metadata to verify this.
pub struct KernelArg<'a> {
    value: KernelArgValue<'a>,
    _borrow: PhantomData<&'a ()>,
}

impl<'a> KernelArg<'a> {
    /// Passes a supported scalar value by value.
    ///
    /// The scalar is borrowed, not copied. Its storage must remain valid and
    /// at a stable address until the synchronous launch operation completes.
    pub fn scalar<T: SupportedScalar>(value: &'a T) -> Self {
        Self {
            value: KernelArgValue::Scalar(value as *const T as *mut c_void),
            _borrow: PhantomData,
        }
    }

    /// Passes a live GenGPU device allocation's address by value.
    ///
    /// The mutable borrow prevents safe deallocation or another mutable use
    /// of this allocation while the argument is retained. Arbitrary numeric
    /// addresses and host pointers cannot be supplied through this API.
    pub fn device_ptr(allocation: &'a mut DeviceAllocation) -> Result<Self> {
        if allocation.is_deallocated() {
            return Err(Error::new(
                ErrorKind::InvalidState,
                "cannot create a kernel argument from a deallocated device allocation",
            ));
        }

        let address = allocation.device_address().as_u64();
        Ok(Self {
            value: KernelArgValue::DevicePointer {
                allocation,
                address,
            },
            _borrow: PhantomData,
        })
    }

    /// Returns the allocation backing a device-pointer argument.
    ///
    /// This is crate-private so the driver backend can validate exact context
    /// ownership before launching.
    pub(crate) fn device_allocation(&self) -> Option<&DeviceAllocation> {
        match &self.value {
            KernelArgValue::Scalar(_) => None,
            KernelArgValue::DevicePointer { allocation, .. } => Some(allocation),
        }
    }

    /// Returns a pointer to the argument value for CUDA's kernelParams array.
    ///
    /// The pointer is valid while this KernelArg remains at a stable address
    /// and its borrow is alive. Launch code must build the pointer array only
    /// after the argument slice is finalized and must not move its elements
    /// until the synchronous launch and synchronization have completed.
    pub(crate) fn kernel_param_ptr(&self) -> *mut c_void {
        match &self.value {
            KernelArgValue::Scalar(pointer) => *pointer,
            KernelArgValue::DevicePointer { address, .. } => address as *const u64 as *mut c_void,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::KernelArg;
    use crate::{DeviceAddress, DeviceAllocation};
    use gengpu_core::{DeviceId, ErrorKind};
    use std::any::Any;
    use std::sync::Arc;

    fn allocation(deallocated: bool) -> DeviceAllocation {
        let owner: Arc<dyn Any + Send + Sync> = Arc::new(());
        let mut allocation = DeviceAllocation::new(
            DeviceId::new(0),
            DeviceAddress::from_raw(0x1234_5678),
            64,
            owner,
            || crate::allocation::CleanupOutcome::Released(Ok(())),
            || {},
        )
        .unwrap();
        if deallocated {
            allocation.deallocate().unwrap();
        }
        allocation
    }

    #[test]
    fn scalar_argument_points_to_the_original_value() {
        let value = 0x1234_u32;
        let argument = KernelArg::scalar(&value);
        let pointer = argument.kernel_param_ptr().cast::<u32>();

        // SAFETY: the argument borrows value for its lifetime and the test
        // reads the pointer before either value or argument is dropped.
        assert_eq!(unsafe { *pointer }, value);
    }

    #[test]
    fn device_pointer_argument_stores_the_device_address_by_value() {
        let mut allocation = allocation(false);
        let argument = KernelArg::device_ptr(&mut allocation).unwrap();
        let pointer = argument.kernel_param_ptr().cast::<u64>();

        // SAFETY: the address field remains stable while argument is borrowed.
        assert_eq!(unsafe { *pointer }, 0x1234_5678);
        assert!(argument.device_allocation().is_some());
    }

    #[test]
    fn rejects_deallocated_device_allocation() {
        let mut allocation = allocation(false);
        allocation.deallocate().unwrap();

        let result = KernelArg::device_ptr(&mut allocation);
        assert!(matches!(result, Err(error) if error.kind() == ErrorKind::InvalidState));
    }

    #[test]
    fn supports_each_documented_scalar_type() {
        let a = 1_u8;
        let b = 2_i8;
        let c = 3_u16;
        let d = 4_i16;
        let e = 5_u32;
        let f = 6_i32;
        let g = 7_u64;
        let h = 8_i64;
        let i = 9.0_f32;
        let j = 10.0_f64;

        let args = [
            KernelArg::scalar(&a),
            KernelArg::scalar(&b),
            KernelArg::scalar(&c),
            KernelArg::scalar(&d),
            KernelArg::scalar(&e),
            KernelArg::scalar(&f),
            KernelArg::scalar(&g),
            KernelArg::scalar(&h),
            KernelArg::scalar(&i),
            KernelArg::scalar(&j),
        ];

        assert_eq!(args.len(), 10);
    }

    #[test]
    fn scalar_and_pointer_arguments_have_distinct_internal_kinds() {
        let value = 7_u32;
        let scalar = KernelArg::scalar(&value);
        let mut allocation = allocation(false);
        let pointer = KernelArg::device_ptr(&mut allocation).unwrap();

        assert!(scalar.device_allocation().is_none());
        assert!(pointer.device_allocation().is_some());
    }
}
