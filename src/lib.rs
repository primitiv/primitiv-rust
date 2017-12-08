extern crate primitiv_sys;
extern crate libc;

use std::fmt::{Debug, Display, Error, Formatter};

pub trait Wrap<T> {
    fn from_inner_ptr(inner: *mut T) -> Self;
    fn as_inner_ptr(&self) -> *const T;
    fn as_inner_mut_ptr(&mut self) -> *mut T;
}

macro_rules! impl_wrap {
    ($name:ident, $type:ident) => {
        impl Wrap<_primitiv::$type> for $name {
            fn from_inner_ptr(inner: *mut _primitiv::$type) -> Self {
                $name { inner: inner }
            }

            fn as_inner_ptr(&self) -> *const _primitiv::$type {
                self.inner
            }

            fn as_inner_mut_ptr(&mut self) -> *mut _primitiv::$type {
                self.inner
            }
        }
    }
}

macro_rules! impl_new {
    ($name:ident, $call:ident) => {
        impl $name {
            pub fn new() -> Self {
                unsafe {
                    let inner = _primitiv::$call();
                    assert!(!inner.is_null());
                    $name { inner: inner }
                }
            }
        }
    }
}

macro_rules! impl_drop {
    ($name:ident, $call:ident) => {
        impl Drop for $name {
            fn drop(&mut self) {
                unsafe { _primitiv::$call(self.inner); }
            }
        }
    }
}

pub trait DataType: Default + Clone + Display + Debug + 'static + Into<f32> {}

macro_rules! data_type {
    ($rust_type:ty) => {
        impl DataType for $rust_type {}
        // impl Into<f32> for $rust_type {
        //     fn into(self) -> f32 {
        //         self as f32
        //     }
        // }
        // impl Into<u32> for $rust_type {
        //     fn into(self) -> u32 {
        //         self as u32
        //     }
        // }
    }
}

data_type!(f32);
// data_type!(i8);
// data_type!(i16);
// data_type!(u8);
// data_type!(u16);

#[derive(Debug, Clone)]
pub struct Expr<T: DataType> {
    expr: T,
}

impl<T: DataType> Display for Expr<T> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        Display::fmt(&self.expr, f)
    }
}

impl<T: DataType> From<T> for Expr<T> {
    fn from(value: T) -> Self {
        Expr { expr: value }
    }
}

pub mod device;
mod graph;
pub use graph::*;
mod initializer;
pub use initializer::*;
mod parameter;
pub use parameter::*;
mod shape;
pub use shape::*;
mod status;
pub use status::*;
mod tensor;
pub use tensor::*;
mod optimizer;
pub use optimizer::*;
pub mod functions;
pub use functions::node_funcs as node_functions;
pub use functions::tensor_funcs as tensor_functions;

pub mod devices;
pub mod initializers;
pub mod optimizers;
