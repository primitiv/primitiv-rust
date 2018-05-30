use primitiv_sys as _primitiv;
use ApiResult;
use Tensor;
use Wrap;

/// `Initializer` trait
pub trait Initializer: Wrap<_primitiv::primitivInitializer_t> {
    /// Provides an initialized tensor.
    fn apply(&self, x: &mut Tensor) {
        unsafe {
            check_api_status!(_primitiv::primitivApplyInitializer(
                self.as_ptr(),
                x.as_mut_ptr(),
            ));
        }
    }
}

macro_rules! impl_initializer {
    ($name:ident) => {
        impl_wrap_owned!($name, primitivInitializer_t);
        impl_drop!($name, primitivDeleteInitializer);
        impl Initializer for $name {}
    }
}
