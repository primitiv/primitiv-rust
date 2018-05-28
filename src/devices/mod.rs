pub use super::device::set_default;

mod any_device {
    use device::Device;
    use primitiv_sys as _primitiv;
    use ApiResult;
    use Wrap;

    #[derive(Debug)]
    pub struct AnyDevice {
        inner: *mut _primitiv::primitivDevice_t,
        owned: bool,
    }
    impl_device!(AnyDevice);
}
pub use self::any_device::AnyDevice;

mod naive_device;
pub use self::naive_device::Naive;

#[cfg(feature = "eigen")]
mod eigen_device;
#[cfg(feature = "eigen")]
pub use self::eigen_device::Eigen;

#[cfg(feature = "cuda")]
mod cuda_device;
#[cfg(feature = "cuda")]
pub use self::cuda_device::CUDA;

#[cfg(feature = "opencl")]
mod opencl_device;
#[cfg(feature = "opencl")]
pub use self::opencl_device::OpenCL;
