use primitiv_sys as _primitiv;
use std::ptr;
use ApiResult;
use Initializer;
use Wrap;

/// Initializer to generate a same-value tensor.
#[derive(Debug)]
pub struct Constant {
    inner: *mut _primitiv::primitivInitializer_t,
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
            assert!(!initializer_ptr.is_null());
            Constant { inner: initializer_ptr }
        }
    }
}

/// Initializer using a parameterized uniform distribution with the range (L,U].
#[derive(Debug)]
pub struct Uniform {
    inner: *mut _primitiv::primitivInitializer_t,
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
            assert!(!initializer_ptr.is_null());
            Uniform { inner: initializer_ptr }
        }
    }
}

/// Initializer using a parameterized normal distribution N(μ,σ).
#[derive(Debug)]
pub struct Normal {
    inner: *mut _primitiv::primitivInitializer_t,
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
            assert!(!initializer_ptr.is_null());
            Normal { inner: initializer_ptr }
        }
    }
}

/// Identity matrix initializer.
#[derive(Debug)]
pub struct Identity {
    inner: *mut _primitiv::primitivInitializer_t,
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
            assert!(!initializer_ptr.is_null());
            Identity { inner: initializer_ptr }
        }
    }
}

/// The Xavier matrix initialization with the uniform distribution.
#[derive(Debug)]
pub struct XavierUniform {
    inner: *mut _primitiv::primitivInitializer_t,
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
            assert!(!initializer_ptr.is_null());
            XavierUniform { inner: initializer_ptr }
        }
    }
}

/// The Xavier matrix initialization with the normal distribution.
#[derive(Debug)]
pub struct XavierNormal {
    inner: *mut _primitiv::primitivInitializer_t,
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
            assert!(!initializer_ptr.is_null());
            XavierNormal { inner: initializer_ptr }
        }
    }
}

/// The Xavier initialization with the uniform distribution for conv2d filters.
#[derive(Debug)]
pub struct XavierUniformConv2D {
    inner: *mut _primitiv::primitivInitializer_t,
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
            assert!(!initializer_ptr.is_null());
            XavierUniformConv2D { inner: initializer_ptr }
        }
    }
}

/// The Xavier initialization with the normal distribution for conv2d filters.
#[derive(Debug)]
pub struct XavierNormalConv2D {
    inner: *mut _primitiv::primitivInitializer_t,
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
            assert!(!initializer_ptr.is_null());
            XavierNormalConv2D { inner: initializer_ptr }
        }
    }
}
