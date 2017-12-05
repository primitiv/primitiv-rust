extern crate primitiv_sys as _primitiv;

use std::ptr;
use Device;
use Initializer;
use Shape;
use Wrap;

#[derive(Debug)]
pub struct Parameter {
    inner: *mut _primitiv::primitiv_Parameter,
}

impl_wrap!(Parameter, primitiv_Parameter);
impl_new!(Parameter, primitiv_Parameter_new);
impl_drop!(Parameter, primitiv_Parameter_delete);

impl Parameter {
    pub fn from_values<T: Device>(shape: &Shape, value: &[u64], device: Option<&T>) -> Self {
        unsafe {
            Parameter {
                inner: _primitiv::primitiv_Parameter_new_with_values(
                    shape.as_inner_ptr(),
                    value.as_ptr() as *const _,
                    value.len(),
                    match device {
                        Some(device) => device.as_inner_mut_ptr(),
                        None => ptr::null_mut(),
                    },
                ),
            }
        }
    }
    pub fn from_initializer<D: Device, I: Initializer>(
        shape: &Shape,
        initializer: &I,
        device: Option<&D>,
    ) -> Self {
        unsafe {
            Parameter {
                inner: _primitiv::primitiv_Parameter_new_with_initializer(
                    shape.as_inner_ptr(),
                    initializer.as_inner_ptr(),
                    match device {
                        Some(device) => device.as_inner_mut_ptr(),
                        None => ptr::null_mut(),
                    },
                ),
            }
        }
    }
}
