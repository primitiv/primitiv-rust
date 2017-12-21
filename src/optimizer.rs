use primitiv_sys as _primitiv;
use ApiResult;
use Parameter;
use Result;
use Wrap;

pub trait Optimizer: Wrap<_primitiv::primitiv_Optimizer> {
    fn add_parameter(&mut self, param: &mut Parameter) -> Result<()> {
        unsafe {
            Result::from_api_status(
                _primitiv::primitiv_Optimizer_add_parameter(
                    self.as_inner_mut_ptr(),
                    param.as_inner_mut_ptr(),
                ),
                (),
            )
        }
    }

    fn reset_gradients(&mut self) -> Result<()> {
        unsafe {
            Result::from_api_status(
                _primitiv::primitiv_Optimizer_reset_gradients(
                    self.as_inner_mut_ptr(),
                ),
                (),
            )
        }
    }

    fn update(&mut self) -> Result<()> {
        unsafe {
            Result::from_api_status(
                _primitiv::primitiv_Optimizer_update(
                    self.as_inner_mut_ptr(),
                ),
                (),
            )
        }
    }
}
