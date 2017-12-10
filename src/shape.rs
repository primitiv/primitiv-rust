use primitiv_sys as _primitiv;
use Status;
use Wrap;

#[derive(Debug)]
pub struct Shape {
    inner: *mut _primitiv::primitiv_Shape,
}

impl_wrap!(Shape, primitiv_Shape);
impl_new!(Shape, safe_primitiv_Shape_new);
impl_drop!(Shape, safe_primitiv_Shape_delete);

impl Shape {
    pub fn from_dims(dims: &[u32], batch: u32) -> Self {
        let mut status = Status::new();
        unsafe {
            let inner = _primitiv::safe_primitiv_Shape_new_with_dims(
                dims.as_ptr() as *const _,
                dims.len(),
                batch,
                status.as_inner_mut_ptr(),
            );
            status.into_result().unwrap();
            assert!(!inner.is_null());
            Shape { inner: inner }
        }
    }

    pub fn size(&self) -> usize {
        let mut status = Status::new();
        unsafe {
            let size =
                _primitiv::safe_primitiv_Shape_size(self.as_inner_ptr(), status.as_inner_mut_ptr());
            status.into_result().unwrap();
            size as usize
        }
    }
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
