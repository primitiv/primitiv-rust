use primitiv_sys as _primitiv;
use Status;
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
    let mut status = Status::new();
    unsafe {
        let device = AnyDevice::from_inner_ptr(_primitiv::safe_primitiv_Device_get_default(
            status.as_inner_mut_ptr(),
        ));
        status.into_result().unwrap();
        device
    }
}

pub fn set_default<D: Device>(device: &mut D) {
    let mut status = Status::new();
    unsafe {
        _primitiv::safe_primitiv_Device_set_default(
            device.as_inner_mut_ptr(),
            status.as_inner_mut_ptr(),
        );
        status.into_result().unwrap();
    }
}
