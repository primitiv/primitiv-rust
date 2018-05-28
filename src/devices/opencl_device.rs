use device::Device;
use primitiv_sys as _primitiv;
use std::ptr;
use ApiResult;
use Wrap;

/// Device class for OpenCL.
#[derive(Debug)]
pub struct OpenCL {
    inner: *mut _primitiv::primitivDevice_t,
    owned: bool,
}

impl_device!(OpenCL);

impl OpenCL {
    /// Creates a new OpenCL device.
    pub fn new(platform_id: u32, device_id: u32) -> Self {
        unsafe {
            let mut device_ptr: *mut _primitiv::primitivDevice_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCreateOpenCLDevice(
                platform_id,
                device_id,
                &mut device_ptr,
            ));
            assert!(!device_ptr.is_null());
            OpenCL {
                inner: device_ptr,
                owned: true,
            }
        }
    }
    /// Creates a new OpenCL device.
    pub fn new_with_seed(platform_id: u32, device_id: u32, rng_seed: u32) -> Self {
        unsafe {
            let mut device_ptr: *mut _primitiv::primitivDevice_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCreateOpenCLDeviceWithSeed(
                platform_id,
                device_id,
                rng_seed,
                &mut device_ptr,
            ));
            assert!(!device_ptr.is_null());
            OpenCL {
                inner: device_ptr,
                owned: true,
            }
        }
    }

    /// Retrieves the number of active platforms.
    pub fn num_platforms() -> u32 {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(_primitiv::primitivGetNumOpenCLPlatforms(
                &mut retval as *mut _,
            ));
            retval
        }
    }

    /// Retrieves the number of active devices on the specified platform.
    pub fn num_devices(platform_id: u32) -> u32 {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(_primitiv::primitivGetNumOpenCLDevices(
                platform_id,
                &mut retval as *mut _,
            ));
            retval
        }
    }
}
