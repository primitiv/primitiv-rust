mod device;
pub use device::*;

pub mod devices;

pub trait Wrap<T> {
    fn as_inner_ptr(&self) -> *const T;
    fn as_inner_mut_ptr(&self) -> *mut T;
}
