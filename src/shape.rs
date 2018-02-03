use primitiv_sys as _primitiv;
use std::ptr;
use ApiResult;
use Result;
use Wrap;

#[derive(Debug)]
pub struct Shape {
    inner: *mut _primitiv::primitivShape_t,
}

impl_wrap_owned!(Shape, primitivShape_t);
impl_drop!(Shape, primitivDeleteShape);

impl Shape {
    pub fn new() -> Self {
        unsafe {
            let mut shape_ptr: *mut _primitiv::primitivShape_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCreateShape(&mut shape_ptr));
            assert!(!shape_ptr.is_null());
            Shape { inner: shape_ptr }
        }
    }

    pub fn from_dims(dims: &[u32], batch: u32) -> Self {
        unsafe {
            let mut shape_ptr: *mut _primitiv::primitivShape_t = ptr::null_mut();
            check_api_status!(
                _primitiv::primitivCreateShapeWithDims(
                    dims.as_ptr() as *const _,
                    dims.len(),
                    batch,
                    &mut shape_ptr,
                )
            );
            //     shape_ptr,
            // ).map(|ptr| {
            //     assert!(!ptr.is_null());
            //     Shape { inner: ptr }
            // })
            Shape { inner: shape_ptr }
        }
    }

    /*
    pub fn size(&self) -> usize {
        unsafe { _primitiv::primitiv_Shape_size(self.as_inner_ptr()) as usize }
    }
    */
}

macro_rules! impl_array_into_shape {
    ($num:expr) => {
        impl Into<Shape> for [u32; $num] {
            fn into(self) -> Shape {
                Shape::from_dims(&self, 1)
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
                Shape::from_dims(&self.0, self.1)
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
