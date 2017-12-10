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
