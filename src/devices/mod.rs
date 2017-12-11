mod naive_device;
pub use self::naive_device::*;

#[cfg(feature = "cuda")]
mod cuda_device;
#[cfg(feature = "cuda")]
pub use self::cuda_device::*;
