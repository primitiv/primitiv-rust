extern crate primitiv_sys as _primitiv;

use Wrap;

pub trait Device: Wrap<_primitiv::primitiv_Device> {}

#[derive(Debug)]
pub struct AnyDevice {
    inner: *mut _primitiv::primitiv_Device,
}

impl_wrap!(AnyDevice, primitiv_Device);

impl Drop for AnyDevice {
    fn drop(&mut self) {
        // An AnyDevice instance does not call internal destructor
        // because its inner is actually reference and it does not own a device instance.
    }
}

impl Device for AnyDevice {}

pub fn get_default() -> AnyDevice {
    unsafe { AnyDevice::from_inner_ptr(_primitiv::primitiv_Device_get_default()) }
}

pub fn set_default<D: Device>(device: &mut D) {
    unsafe {
        _primitiv::primitiv_Device_set_default(device.as_inner_mut_ptr());
    }
}
