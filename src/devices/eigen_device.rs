use primitiv_sys as _primitiv;
use std::ptr;
use ApiResult;
use device::Device;
use Wrap;

/// Device class for the Eigen3 backend.
#[derive(Debug)]
pub struct Eigen {
    inner: *mut _primitiv::primitivDevice_t,
    owned: bool,
}

impl_device!(Eigen);

impl Eigen {
    /// Creates a Eigen object.
    pub fn new() -> Self {
        unsafe {
            let mut device_ptr: *mut _primitiv::primitivDevice_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCreateEigenDevice(&mut device_ptr));
            assert!(!device_ptr.is_null());
            Eigen {
                inner: device_ptr,
                owned: true,
            }
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
            assert!(!device_ptr.is_null());
            Eigen {
                inner: device_ptr,
                owned: true,
            }
        }
    }
}
