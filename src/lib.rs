extern crate primitiv_sys;
extern crate libc;

use std::fmt::{Debug, Display, Error, Formatter};

pub trait Wrap<T> {
    fn from_inner_ptr(inner: *mut T, owned: bool) -> Self;
    fn as_inner_ptr(&self) -> *const T;
    fn as_inner_mut_ptr(&mut self) -> *mut T;
    fn is_owned(&self) -> bool;
}

macro_rules! impl_wrap {
    ($name:ident, $type:ident) => {
        impl Wrap<_primitiv::$type> for $name {
            #[inline(always)]
            fn from_inner_ptr(inner: *mut _primitiv::$type, owned: bool) -> Self {
                $name { inner: inner, owned: owned }
            }

            #[inline(always)]
            fn as_inner_ptr(&self) -> *const _primitiv::$type {
                self.inner
            }

            #[inline(always)]
            fn as_inner_mut_ptr(&mut self) -> *mut _primitiv::$type {
                self.inner
            }

            #[inline(always)]
            fn is_owned(&self) -> bool {
                self.owned
            }
        }
    }
}

macro_rules! impl_wrap_owned {
    ($name:ident, $type:ident) => {
        impl Wrap<_primitiv::$type> for $name {
            #[inline(always)]
            fn from_inner_ptr(inner: *mut _primitiv::$type, owned: bool) -> Self {
                $name { inner: inner }
            }

            #[inline(always)]
            fn as_inner_ptr(&self) -> *const _primitiv::$type {
                self.inner
            }

            #[inline(always)]
            fn as_inner_mut_ptr(&mut self) -> *mut _primitiv::$type {
                self.inner
            }

            #[inline(always)]
            fn is_owned(&self) -> bool {
                true
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
                    $name { inner: inner, owned: true }
                }
            }
        }
    }
}

macro_rules! impl_drop {
    ($name:ident, $call:ident) => {
        impl Drop for $name {
            fn drop(&mut self) {
                if (self.is_owned()) {
                    unsafe {
                        _primitiv::$call(self.inner);
                    }
                }
            }
        }
    }
}

pub mod device;
mod graph;
pub use graph::*;
// mod initializer;
// pub use initializer::*;
// mod parameter;
// pub use parameter::*;
// mod shape;
// pub use shape::*;
mod status;
pub use status::*;
// mod tensor;
// pub use tensor::*;
// mod optimizer;
// pub use optimizer::*;
// pub mod functions;
// pub use functions::node_funcs as node_functions;
// pub use functions::tensor_funcs as tensor_functions;

// pub mod devices;
// pub mod initializers;
// pub mod optimizers;
