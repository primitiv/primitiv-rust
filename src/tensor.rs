use primitiv_sys as _primitiv;
use std::ptr;

use ApiResult;
use Result;
use Status;
use Wrap;

#[derive(Debug)]
pub struct Tensor {
    inner: *mut _primitiv::primitivTensor_t,
    owned: bool,
}

impl_wrap!(Tensor, primitivTensor_t);
// impl_new!(Tensor, primitiv_Tensor_new);
impl_drop!(Tensor, primitivDeleteTensor);

impl Tensor {

    /// Creates a new Graph object.
    pub fn new() -> Self {
        unsafe {
            let mut inner: *mut _primitiv::primitivTensor_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCreateTensor(&mut inner));
            assert!(!inner.is_null());
            Tensor { inner: inner, owned: true }
        }
    }
}
