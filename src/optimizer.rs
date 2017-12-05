extern crate primitiv_sys as _primitiv;

use Parameter;
use Wrap;

pub trait Optimizer: Wrap<_primitiv::primitiv_Optimizer> {
    fn add_parameter(&mut self, param: &mut Parameter) {
        unsafe {
            _primitiv::primitiv_Optimizer_add_parameter(
                self.as_inner_mut_ptr(),
                param.as_inner_mut_ptr(),
            );
        }
    }
}
