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

mod device;
pub use device::*;
mod graph;
pub use graph::*;
mod initializer;
pub use initializer::*;
mod parameter;
pub use parameter::*;
mod shape;
pub use shape::*;
mod optimizer;
pub use optimizer::*;

pub mod devices;
pub mod initializers;
pub mod optimizers;
