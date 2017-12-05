extern crate primitiv_sys as _primitiv;

use Optimizer;
use Wrap;

#[derive(Debug)]
pub struct SGD {
    inner: *mut _primitiv::primitiv_Optimizer,
}

impl_wrap!(SGD, primitiv_Optimizer);
impl_drop!(SGD, primitiv_SGD_delete);

impl Optimizer for SGD {}

impl SGD {
    pub fn new(eta: f32) -> Self {
        unsafe {
            let inner = _primitiv::primitiv_SGD_new_with_eta(eta);
            assert!(!inner.is_null());
            SGD { inner: inner }
        }
    }
}
