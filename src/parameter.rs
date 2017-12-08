use primitiv_sys as _primitiv;
use std::ptr;
use device::{AnyDevice, Device};
use Initializer;
use Shape;
use Status;
use Wrap;

#[derive(Debug)]
pub struct Parameter {
    inner: *mut _primitiv::primitiv_Parameter,
}

impl_wrap!(Parameter, primitiv_Parameter);
impl_new!(Parameter, safe_primitiv_Parameter_new);
impl_drop!(Parameter, safe_primitiv_Parameter_delete);

impl Parameter {
    pub fn from_values(shape: &Shape, value: &[f32]) -> Self {
        Self::from_values_with_device::<AnyDevice>(shape, value, None)
    }

    pub fn from_values_with_device<D: Device>(
        shape: &Shape,
        value: &[f32],
        device: Option<&mut D>,
    ) -> Self {
        let mut status = Status::new();
        unsafe {
            let inner = _primitiv::safe_primitiv_Parameter_new_with_values(
                shape.as_inner_ptr(),
                value.as_ptr() as *const _,
                value.len(),
                device.map(|d| d.as_inner_mut_ptr()).unwrap_or(
                    ptr::null_mut(),
                ),
                status.as_inner_mut_ptr(),
            );
            status.into_result().unwrap();
            assert!(!inner.is_null());
            Parameter { inner: inner }
        }
    }

    pub fn from_initializer<I: Initializer>(shape: &Shape, initializer: &I) -> Self {
        Self::from_initializer_with_device::<AnyDevice, I>(shape, initializer, None)
    }

    pub fn from_initializer_with_device<D: Device, I: Initializer>(
        shape: &Shape,
        initializer: &I,
        device: Option<&mut D>,
    ) -> Self {
        let mut status = Status::new();
        unsafe {
            let inner = _primitiv::safe_primitiv_Parameter_new_with_initializer(
                shape.as_inner_ptr(),
                initializer.as_inner_ptr(),
                device.map(|d| d.as_inner_mut_ptr()).unwrap_or(
                    ptr::null_mut(),
                ),
                status.as_inner_mut_ptr(),
            );
            status.into_result().unwrap();
            assert!(!inner.is_null());
            Parameter { inner: inner }
        }
    }
}
