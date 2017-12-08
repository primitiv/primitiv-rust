extern crate primitiv_sys as _primitiv;

use Tensor;
use Parameter;
use Wrap;

pub fn parameter(param: &mut Parameter) -> Tensor {
    unsafe {
        Tensor::from_inner_ptr(_primitiv::primitiv_tensor_func_parameter(
            param.as_inner_mut_ptr(),
        ))
    }
}

pub fn tanh(x: &Tensor) -> Tensor {
    unsafe { Tensor::from_inner_ptr(_primitiv::primitiv_tensor_func_tanh(x.as_inner_ptr())) }
}
