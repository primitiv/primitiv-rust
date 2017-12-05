extern crate primitiv_sys as _primitiv;

use Wrap;
use initializer;

#[derive(Debug)]
pub struct Constant {
    inner: *mut _primitiv::primitiv_Initializer,
}

impl_wrap!(Constant, primitiv_Initializer);
impl_drop!(Constant, primitiv_Constant_delete);

impl initializer::Initializer for Constant {}

impl Constant {
    pub fn new(k: f32) -> Self {
        unsafe {
            let inner = _primitiv::primitiv_Constant_new(k);
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
impl_new!(Identity, primitiv_Identity_new);
impl_drop!(Identity, primitiv_Identity_delete);

impl initializer::Initializer for Identity {}

#[derive(Debug)]
pub struct XavierUniform {
    inner: *mut _primitiv::primitiv_Initializer,
}

impl_wrap!(XavierUniform, primitiv_Initializer);
impl_drop!(XavierUniform, primitiv_XavierUniform_delete);

impl initializer::Initializer for XavierUniform {}

impl XavierUniform {
    pub fn new(scale: f32) -> Self {
        unsafe {
            let inner = _primitiv::primitiv_XavierUniform_new(scale);
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
impl_drop!(XavierNormal, primitiv_XavierNormal_delete);

impl initializer::Initializer for XavierNormal {}

impl XavierNormal {
    pub fn new(scale: f32) -> Self {
        unsafe {
            let inner = _primitiv::primitiv_XavierNormal_new(scale);
            assert!(!inner.is_null());
            XavierNormal { inner: inner }
        }
    }
}
