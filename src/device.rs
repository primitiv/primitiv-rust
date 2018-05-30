use primitiv_sys as _primitiv;
use ApiResult;
use Wrap;

/// `Device` trait
pub trait Device: Wrap<_primitiv::primitivDevice_t> {}

macro_rules! impl_device {
    ($name:ident) => {
        impl_wrap!($name, primitivDevice_t);
        impl_drop!($name, primitivDeleteDevice);
        impl Device for $name {}
    };
}

#[allow(dead_code)]
pub fn set_default<D: Device + ?Sized>(device: &mut D) {
    unsafe {
        check_api_status!(_primitiv::primitivSetDefaultDevice(device.as_mut_ptr()));
    }
}
