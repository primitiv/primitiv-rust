use primitiv_sys as _primitiv;
use Device;
use Wrap;

/// Device class for the naive function implementations on CPU.
#[derive(Debug)]
pub struct Naive {
    inner: *mut _primitiv::primitiv_Device,
    owned: bool,
}

impl_device!(Naive);
impl_new!(Naive, primitiv_devices_Naive_new);
