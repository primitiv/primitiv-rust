extern crate primitiv_sys as _primitiv;

use Wrap;
use device;

#[derive(Debug)]
pub struct Naive {
    inner: *mut _primitiv::primitiv_Device,
}

impl_wrap!(Naive, primitiv_Device);
impl_new!(Naive, primitiv_Naive_new);
impl_drop!(Naive, primitiv_Naive_delete);

impl device::Device for Naive {}
