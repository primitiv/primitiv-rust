use primitiv_sys as _primitiv;
use device;
use Status;
use Wrap;

#[derive(Debug)]
pub struct CUDA {
    inner: *mut _primitiv::primitiv_Device,
}

impl_wrap!(CUDA, primitiv_Device);
impl_drop!(CUDA, safe_primitiv_Naive_delete);

impl device::Device for CUDA {}

impl CUDA {
    pub fn new(device_id: u32) -> Self {
        let mut status = Status::new();
        unsafe {
            let inner = _primitiv::safe_primitiv_CUDA_new(device_id, status.as_inner_mut_ptr());
            status.into_result().unwrap();
            assert!(!inner.is_null());
            CUDA { inner: inner }
        }
    }

    pub fn num_devices() -> u32 {
        let mut status = Status::new();
        unsafe {
            let num_devices = _primitiv::safe_primitiv_CUDA_num_devices(status.as_inner_mut_ptr());
            status.into_result().unwrap();
            num_devices
        }
    }

    pub fn dump_description(&self) {
        let mut status = Status::new();
        unsafe {
            _primitiv::safe_primitiv_CUDA_dump_description(
                self.as_inner_ptr(),
                status.as_inner_mut_ptr(),
            );
            status.into_result().unwrap();
        }
    }
}
