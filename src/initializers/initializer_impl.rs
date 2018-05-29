use primitiv_sys as _primitiv;
use std::ptr::{self, NonNull};
use ApiResult;
use Initializer;
use Wrap;

/// Initializer to generate a same-value tensor.
#[derive(Debug)]
pub struct Constant {
    inner: NonNull<_primitiv::primitivInitializer_t>,
}

impl_initializer!(Constant);

impl Constant {
    /// Crates a new Constant initializer.
    pub fn new(k: f32) -> Self {
        unsafe {
            let mut initializer_ptr: *mut _primitiv::primitivInitializer_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCreateConstantInitializer(
                k,
                &mut initializer_ptr,
            ));
            Constant::from_raw(initializer_ptr, true)
        }
    }
}

/// Initializer using a parameterized uniform distribution with the range (L,U].
#[derive(Debug)]
pub struct Uniform {
    inner: NonNull<_primitiv::primitivInitializer_t>,
}

impl_initializer!(Uniform);

impl Uniform {
    /// Crates a new Uniform initializer.
    pub fn new(lower: f32, upper: f32) -> Self {
        unsafe {
            let mut initializer_ptr: *mut _primitiv::primitivInitializer_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCreateUniformInitializer(
                lower,
                upper,
                &mut initializer_ptr,
            ));
            Uniform::from_raw(initializer_ptr, true)
        }
    }
}

/// Initializer using a parameterized normal distribution N(μ,σ).
#[derive(Debug)]
pub struct Normal {
    inner: NonNull<_primitiv::primitivInitializer_t>,
}

impl_initializer!(Normal);

impl Normal {
    /// Crates a new Normal initializer.
    pub fn new(mean: f32, sd: f32) -> Self {
        unsafe {
            let mut initializer_ptr: *mut _primitiv::primitivInitializer_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCreateNormalInitializer(
                mean,
                sd,
                &mut initializer_ptr,
            ));
            Normal::from_raw(initializer_ptr, true)
        }
    }
}

/// Identity matrix initializer.
#[derive(Debug)]
pub struct Identity {
    inner: NonNull<_primitiv::primitivInitializer_t>,
}

impl_initializer!(Identity);

impl Identity {
    /// Crates a new Identity initializer.
    pub fn new() -> Self {
        unsafe {
            let mut initializer_ptr: *mut _primitiv::primitivInitializer_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCreateIdentityInitializer(
                &mut initializer_ptr,
            ));
            Identity::from_raw(initializer_ptr, true)
        }
    }
}

/// The Xavier matrix initialization with the uniform distribution.
#[derive(Debug)]
pub struct XavierUniform {
    inner: NonNull<_primitiv::primitivInitializer_t>,
}

impl_initializer!(XavierUniform);

impl XavierUniform {
    /// Crates a new XavierUniform initializer.
    pub fn new(scale: f32) -> Self {
        unsafe {
            let mut initializer_ptr: *mut _primitiv::primitivInitializer_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCreateXavierUniformInitializer(
                scale,
                &mut initializer_ptr,
            ));
            XavierUniform::from_raw(initializer_ptr, true)
        }
    }
}

impl Default for XavierUniform {
    fn default() -> Self {
        Self::new(1.0)
    }
}

/// The Xavier matrix initialization with the normal distribution.
#[derive(Debug)]
pub struct XavierNormal {
    inner: NonNull<_primitiv::primitivInitializer_t>,
}

impl_initializer!(XavierNormal);

impl XavierNormal {
    /// Crates a new XavierNormal initializer.
    pub fn new(scale: f32) -> Self {
        unsafe {
            let mut initializer_ptr: *mut _primitiv::primitivInitializer_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCreateXavierNormalInitializer(
                scale,
                &mut initializer_ptr,
            ));
            XavierNormal::from_raw(initializer_ptr, true)
        }
    }
}

impl Default for XavierNormal {
    fn default() -> Self {
        Self::new(1.0)
    }
}

/// The Xavier initialization with the uniform distribution for conv2d filters.
#[derive(Debug)]
pub struct XavierUniformConv2D {
    inner: NonNull<_primitiv::primitivInitializer_t>,
}

impl_initializer!(XavierUniformConv2D);

impl XavierUniformConv2D {
    /// Crates a new XavierUniformConv2D initializer.
    pub fn new(scale: f32) -> Self {
        unsafe {
            let mut initializer_ptr: *mut _primitiv::primitivInitializer_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCreateXavierUniformConv2DInitializer(
                scale,
                &mut initializer_ptr,
            ));
            XavierUniformConv2D::from_raw(initializer_ptr, true)
        }
    }
}

impl Default for XavierUniformConv2D {
    fn default() -> Self {
        Self::new(1.0)
    }
}

/// The Xavier initialization with the normal distribution for conv2d filters.
#[derive(Debug)]
pub struct XavierNormalConv2D {
    inner: NonNull<_primitiv::primitivInitializer_t>,
}

impl_initializer!(XavierNormalConv2D);

impl XavierNormalConv2D {
    /// Crates a new XavierNormalConv2D initializer.
    pub fn new(scale: f32) -> Self {
        unsafe {
            let mut initializer_ptr: *mut _primitiv::primitivInitializer_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCreateXavierNormalConv2DInitializer(
                scale,
                &mut initializer_ptr,
            ));
            XavierNormalConv2D::from_raw(initializer_ptr, true)
        }
    }
}

impl Default for XavierNormalConv2D {
    fn default() -> Self {
        Self::new(1.0)
    }
}
