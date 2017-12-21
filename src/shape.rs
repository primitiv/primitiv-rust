use primitiv_sys as _primitiv;
use std::ptr;
use Result;
use ApiResult;
use Wrap;

#[derive(Debug)]
pub struct Shape {
    inner: *mut _primitiv::primitiv_Shape,
}

impl_wrap_owned!(Shape, primitiv_Shape);
impl_drop!(Shape, primitiv_Shape_delete);

impl Shape {
    pub fn new() -> Self {
        unsafe { Shape { inner: _primitiv::primitiv_Shape_new() } }
    }

    pub fn from_dims(dims: &[u32], batch: u32) -> Result<Self> {
        unsafe {
            let mut shape_ptr: *mut _primitiv::primitiv_Shape = ptr::null_mut();
            Result::from_api_status(
                _primitiv::primitiv_Shape_new_with_dims(
                    dims.as_ptr() as *const _,
                    dims.len(),
                    batch,
                    &mut shape_ptr,
                ),
                shape_ptr,
            ).map(|ptr| {
                assert!(!ptr.is_null());
                Shape { inner: ptr }
            })
        }
    }

    pub fn size(&self) -> usize {
        unsafe { _primitiv::primitiv_Shape_size(self.as_inner_ptr()) as usize }
    }
}

macro_rules! impl_array_into_shape {
    ($num:expr) => {
        impl Into<Shape> for [u32; $num] {
            fn into(self) -> Shape {
                Shape::from_dims(&self, 1).unwrap()
            }
        }
    }
}
impl_array_into_shape!(0);
impl_array_into_shape!(1);
impl_array_into_shape!(2);
impl_array_into_shape!(3);
impl_array_into_shape!(4);
impl_array_into_shape!(5);
impl_array_into_shape!(6);
impl_array_into_shape!(7);
impl_array_into_shape!(8);

macro_rules! impl_tuple_into_shape {
    ($num:expr) => {
        impl Into<Shape> for ([u32; $num], u32) {
            fn into(self) -> Shape {
                Shape::from_dims(&self.0, self.1).unwrap()
            }
        }
    }
}
impl_tuple_into_shape!(0);
impl_tuple_into_shape!(1);
impl_tuple_into_shape!(2);
impl_tuple_into_shape!(3);
impl_tuple_into_shape!(4);
impl_tuple_into_shape!(5);
impl_tuple_into_shape!(6);
impl_tuple_into_shape!(7);
impl_tuple_into_shape!(8);
