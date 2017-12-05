extern crate primitiv_sys as _primitiv;

use Wrap;

#[derive(Debug)]
pub struct Tensor {
    inner: *mut _primitiv::primitiv_Tensor,
}

impl_wrap!(Tensor, primitiv_Tensor);
// impl_new!(Tensor, primitiv_Tensor_new);
// impl_drop!(Tensor, primitiv_Tensor_delete);
