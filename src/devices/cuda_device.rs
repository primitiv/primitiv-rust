use primitiv_sys as _primitiv;
use std::ptr;
use ApiResult;
use device::Device;
use Wrap;

/// Device class for CUDA.
#[derive(Debug)]
pub struct CUDA {
    inner: *mut _primitiv::primitivDevice_t,
    owned: bool,
}

impl_device!(CUDA);

impl CUDA {
    /// Creates a new CUDA device.
    pub fn new(device_id: u32) -> Self {
        unsafe {
            let mut device_ptr: *mut _primitiv::primitivDevice_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCreateCudaDevice(
                device_id,
                &mut device_ptr,
            ));
            assert!(!device_ptr.is_null());
            CUDA {
                inner: device_ptr,
                owned: true,
            }
        }
    }
    /// Creates a new CUDA device.
    pub fn new_with_seed(device_id: u32, rng_seed: u32) -> Self {
        unsafe {
            let mut device_ptr: *mut _primitiv::primitivDevice_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCreateCudaDeviceWithSeed(
                device_id,
                rng_seed,
                &mut device_ptr,
            ));
            assert!(!device_ptr.is_null());
            CUDA {
                inner: device_ptr,
                owned: true,
            }
        }
    }

    /// Retrieves the number of active hardwares.
    pub fn num_devices() -> u32 {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(_primitiv::primitivGetNumCudaDevices(&mut retval as *mut _));
            retval
        }
    }
}
