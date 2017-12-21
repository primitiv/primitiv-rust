use primitiv_sys as _primitiv;
use std::ptr;
use ApiResult;
use device::{AnyDevice, Device};
use Initializer;
use Result;
use Shape;
use Wrap;

#[derive(Debug)]
pub struct Parameter {
    inner: *mut _primitiv::primitiv_Parameter,
    owned: bool,
}

impl_wrap!(Parameter, primitiv_Parameter);
impl_new!(Parameter, primitiv_Parameter_new);
impl_drop!(Parameter, primitiv_Parameter_delete);

impl Parameter {
    pub fn from_values<S: Into<Shape>>(shape: S, value: &[f32]) -> Result<Self> {
        Self::from_values_with_device::<S, AnyDevice>(shape, value, None)
    }

    pub fn from_values_with_device<S: Into<Shape>, D: Device>(
        shape: S,
        value: &[f32],
        device: Option<&mut D>,
    ) -> Result<Self> {
        unsafe {
            let mut parameter_ptr: *mut _primitiv::primitiv_Parameter = ptr::null_mut();
            Result::from_api_status(
                _primitiv::primitiv_Parameter_new_with_values(
                    shape.into().as_inner_ptr(),
                    value.as_ptr() as *const _,
                    value.len(),
                    device.map(|d| d.as_inner_mut_ptr()).unwrap_or(
                        ptr::null_mut(),
                    ),
                    &mut parameter_ptr,
                ),
                parameter_ptr,
            ).map(|ptr| {
                Parameter {
                    inner: ptr,
                    owned: true,
                }
            })
        }
    }

    pub fn from_initializer<S: Into<Shape>, I: Initializer>(
        shape: S,
        initializer: &I,
    ) -> Result<Self> {
        Self::from_initializer_with_device::<S, AnyDevice, I>(shape, initializer, None)
    }

    pub fn from_initializer_with_device<S: Into<Shape>, D: Device, I: Initializer>(
        shape: S,
        initializer: &I,
        device: Option<&mut D>,
    ) -> Result<Self> {
        unsafe {
            let mut parameter_ptr: *mut _primitiv::primitiv_Parameter = ptr::null_mut();
            Result::from_api_status(
                _primitiv::primitiv_Parameter_new_with_initializer(
                    shape.into().as_inner_ptr(),
                    initializer.as_inner_ptr(),
                    device.map(|d| d.as_inner_mut_ptr()).unwrap_or(
                        ptr::null_mut(),
                    ),
                    &mut parameter_ptr,
                ),
                parameter_ptr,
            ).map(|ptr| {
                Parameter {
                    inner: ptr,
                    owned: true,
                }
            })
        }
    }
}
