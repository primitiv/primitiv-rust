use primitiv_sys as _primitiv;
use Parameter;
use Status;
use Wrap;

pub trait Optimizer: Wrap<_primitiv::primitiv_Optimizer> {
    fn add_parameter(&mut self, param: &mut Parameter) {
        let mut status = Status::new();
        unsafe {
            _primitiv::safe_primitiv_Optimizer_add_parameter(
                self.as_inner_mut_ptr(),
                param.as_inner_mut_ptr(),
                status.as_inner_mut_ptr(),
            );
            status.into_result().unwrap();
        }
    }

    fn reset_gradients(&mut self) {
        let mut status = Status::new();
        unsafe {
            _primitiv::safe_primitiv_Optimizer_reset_gradients(
                self.as_inner_mut_ptr(),
                status.as_inner_mut_ptr(),
            );
            status.into_result().unwrap();
        }
    }

    fn update(&mut self) {
        let mut status = Status::new();
        unsafe {
            _primitiv::safe_primitiv_Optimizer_update(
                self.as_inner_mut_ptr(),
                status.as_inner_mut_ptr(),
            );
            status.into_result().unwrap();
        }
    }
}
