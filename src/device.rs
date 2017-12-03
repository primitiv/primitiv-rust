extern crate primitiv_sys as _primitiv;

use Wrap;

pub trait Device: Wrap<_primitiv::primitiv_Device> {
    fn set_default<T: Device>(device: &T) {
        unsafe {
            _primitiv::primitiv_Device_set_default(device.as_inner_mut_ptr());
        }
    }
}
