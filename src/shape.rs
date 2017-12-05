extern crate primitiv_sys as _primitiv;

use Wrap;

#[derive(Debug)]
pub struct Shape {
    inner: *mut _primitiv::primitiv_Shape,
}

impl_wrap!(Shape, primitiv_Shape);
impl_new!(Shape, primitiv_Shape_new);
impl_drop!(Shape, primitiv_Shape_delete);

impl Shape {
    pub fn from_dims(dims: &[u32], batch: u32) -> Self {
        unsafe {
            Shape {
                inner: _primitiv::primitiv_Shape_new_with_dims(
                    dims.as_ptr() as *const _,
                    dims.len(),
                    batch,
                ),
            }
        }
    }

    pub fn size(&self) -> u32 {
        unsafe { _primitiv::primitiv_Shape_size(self.as_inner_ptr()) }
    }
}
