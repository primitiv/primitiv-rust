extern crate primitiv_sys as _primitiv;

use Wrap;
use optimizer;

#[derive(Debug)]
pub struct SGD {
    inner: *mut _primitiv::primitiv_Optimizer,
}

impl_wrap!(SGD, primitiv_Optimizer);
impl_drop!(SGD, primitiv_SGD_delete);

impl optimizer::Optimizer for SGD {}
