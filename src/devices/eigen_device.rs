use device::Device;
use primitiv_sys as _primitiv;
use std::ptr::{self, NonNull};
use ApiResult;
use Wrap;

/// Device class for the Eigen3 backend.
#[derive(Debug)]
pub struct Eigen {
    inner: NonNull<_primitiv::primitivDevice_t>,
    owned: bool,
}

impl_device!(Eigen);

impl Eigen {
    /// Creates a Eigen object.
    pub fn new() -> Self {
        unsafe {
            let mut device_ptr: *mut _primitiv::primitivDevice_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCreateEigenDevice(&mut device_ptr));
            Eigen::from_raw(device_ptr, true)
        }
    }

    /// Creates a Eigen object.
    pub fn new_with_seed(rng_seed: u32) -> Self {
        unsafe {
            let mut device_ptr: *mut _primitiv::primitivDevice_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCreateEigenDeviceWithSeed(
                rng_seed,
                &mut device_ptr,
            ));
            Eigen::from_raw(device_ptr, true)
        }
    }
}
