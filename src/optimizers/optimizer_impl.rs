use primitiv_sys as _primitiv;
use Optimizer;
use Status;
use Wrap;

#[derive(Debug)]
pub struct SGD {
    inner: *mut _primitiv::primitiv_Optimizer,
}

impl_wrap!(SGD, primitiv_Optimizer);
impl_drop!(SGD, safe_primitiv_SGD_delete);

impl Optimizer for SGD {}

impl SGD {
    pub fn new(eta: f32) -> Self {
        let mut status = Status::new();
        unsafe {
            let inner = _primitiv::safe_primitiv_SGD_new(eta, status.as_inner_mut_ptr());
            status.into_result().unwrap();
            assert!(!inner.is_null());
            SGD { inner: inner }
        }
    }
}
