use primitiv_sys as _primitiv;
use Status;
use Wrap;

use std::ptr;

/// `Device` trait
pub trait Device: Wrap<_primitiv::primitiv_Device> {}

#[derive(Debug)]
pub struct AnyDevice {
    inner: *mut _primitiv::primitiv_Device,
    owned: bool,
}

impl_wrap!(AnyDevice, primitiv_Device);
impl_drop!(AnyDevice, primitiv_Device_delete);

impl Device for AnyDevice {}

/*
pub fn get_default() -> Result<AnyDevice> {
    // let mut status = Status::new();
    unsafe {
        let mut device_ptr: *mut _primitiv::primitiv_Device = ptr::null_mut();
        let status = _primitiv::primitiv_Device_get_default(&mut device_ptr);
        // let device = AnyDevice::from_inner_ptr(
        //     // status.as_inner_mut_ptr(),
        // ));
        // status.into_result().unwrap();
        // device
        assert!(!inner.is_null());
        AnyDevice { inner: device_ptr }
    }
}
*/

pub fn set_default<D: Device>(device: &mut D) {
    unsafe {
        _primitiv::primitiv_Device_set_default(device.as_inner_mut_ptr());
    }
}
