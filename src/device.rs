use primitiv_sys as _primitiv;
use Status;
use Wrap;

use std::ptr;

/// `Device` trait
pub trait Device: Wrap<_primitiv::primitiv_Device> {}

macro_rules! impl_device {
    ($name:ident) => {
        impl_wrap!($name, primitiv_Device);
        impl_drop!($name, primitiv_Device_delete);
        impl Device for $name {}
    }
}

#[derive(Debug)]
pub struct AnyDevice {
    inner: *mut _primitiv::primitiv_Device,
    owned: bool,
}

impl_device!(AnyDevice);

pub fn set_default<D: Device>(device: &mut D) {
    unsafe {
        _primitiv::primitiv_Device_set_default(device.as_inner_mut_ptr());
    }
}
