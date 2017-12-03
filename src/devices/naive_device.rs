extern crate primitiv_sys as _primitiv;

use Wrap;
use device;

#[derive(Debug)]
pub struct Naive {
    inner: *mut _primitiv::primitiv_Device,
}

impl Drop for Naive {
    fn drop(&mut self) {
        unsafe {
            _primitiv::primitiv_Naive_delete(self.inner);
        }
    }
}

impl Wrap<_primitiv::primitiv_Device> for Naive {
    fn as_inner_ptr(&self) -> *const _primitiv::primitiv_Device {
        self.inner
    }

    fn as_inner_mut_ptr(&self) -> *mut _primitiv::primitiv_Device {
        self.inner
    }
}

impl device::Device for Naive {}

impl Naive {
    pub fn new() -> Self {
        unsafe {
            let inner = _primitiv::primitiv_Naive_new();
            assert!(!inner.is_null());
            Naive { inner: inner }
        }
    }
}
