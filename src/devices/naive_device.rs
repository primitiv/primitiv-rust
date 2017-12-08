use primitiv_sys as _primitiv;
use device;
use Status;
use Wrap;

#[derive(Debug)]
pub struct Naive {
    inner: *mut _primitiv::primitiv_Device,
}

impl_wrap!(Naive, primitiv_Device);
impl_new!(Naive, safe_primitiv_Naive_new);
impl_drop!(Naive, safe_primitiv_Naive_delete);

impl device::Device for Naive {}
