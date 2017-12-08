use primitiv_sys as _primitiv;
use initializer;
use Status;
use Wrap;

#[derive(Debug)]
pub struct Constant {
    inner: *mut _primitiv::primitiv_Initializer,
}

impl_wrap!(Constant, primitiv_Initializer);
impl_drop!(Constant, safe_primitiv_Constant_delete);

impl initializer::Initializer for Constant {}

impl Constant {
    pub fn new(k: f32) -> Self {
        let mut status = Status::new();
        unsafe {
            let inner = _primitiv::safe_primitiv_Constant_new(k, status.as_inner_mut_ptr());
            status.into_result().unwrap();
            assert!(!inner.is_null());
            Constant { inner: inner }
        }
    }
}

#[derive(Debug)]
pub struct Identity {
    inner: *mut _primitiv::primitiv_Initializer,
}

impl_wrap!(Identity, primitiv_Initializer);
impl_new!(Identity, safe_primitiv_Identity_new);
impl_drop!(Identity, safe_primitiv_Identity_delete);

impl initializer::Initializer for Identity {}

#[derive(Debug)]
pub struct XavierUniform {
    inner: *mut _primitiv::primitiv_Initializer,
}

impl_wrap!(XavierUniform, primitiv_Initializer);
impl_drop!(XavierUniform, safe_primitiv_XavierUniform_delete);

impl initializer::Initializer for XavierUniform {}

impl XavierUniform {
    pub fn new(scale: f32) -> Self {
        let mut status = Status::new();
        unsafe {
            let inner =
                _primitiv::safe_primitiv_XavierUniform_new(scale, status.as_inner_mut_ptr());
            status.into_result().unwrap();
            assert!(!inner.is_null());
            XavierUniform { inner: inner }
        }
    }
}

#[derive(Debug)]
pub struct XavierNormal {
    inner: *mut _primitiv::primitiv_Initializer,
}

impl_wrap!(XavierNormal, primitiv_Initializer);
impl_drop!(XavierNormal, safe_primitiv_XavierNormal_delete);

impl initializer::Initializer for XavierNormal {}

impl XavierNormal {
    pub fn new(scale: f32) -> Self {
        let mut status = Status::new();
        unsafe {
            let inner = _primitiv::safe_primitiv_XavierNormal_new(scale, status.as_inner_mut_ptr());
            status.into_result().unwrap();
            assert!(!inner.is_null());
            XavierNormal { inner: inner }
        }
    }
}
