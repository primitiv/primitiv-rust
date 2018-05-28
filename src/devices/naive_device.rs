use device::Device;
use primitiv_sys as _primitiv;
use std::ptr;
use ApiResult;
use Wrap;

/// Device class for the naive function implementations on CPU.
#[derive(Debug)]
pub struct Naive {
    inner: *mut _primitiv::primitivDevice_t,
    owned: bool,
}

impl_device!(Naive);

impl Naive {
    /// Creates a Naive object.
    pub fn new() -> Self {
        unsafe {
            let mut device_ptr: *mut _primitiv::primitivDevice_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCreateNaiveDevice(&mut device_ptr));
            assert!(!device_ptr.is_null());
            Naive {
                inner: device_ptr,
                owned: true,
            }
        }
    }

    /// Creates a Naive object.
    pub fn new_with_seed(rng_seed: u32) -> Self {
        unsafe {
            let mut device_ptr: *mut _primitiv::primitivDevice_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCreateNaiveDeviceWithSeed(
                rng_seed,
                &mut device_ptr,
            ));
            assert!(!device_ptr.is_null());
            Naive {
                inner: device_ptr,
                owned: true,
            }
        }
    }
}
