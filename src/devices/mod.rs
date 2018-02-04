mod any_device {
    use primitiv_sys as _primitiv;
    use device::Device;
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

#[cfg(feature = "cuda")]
mod cuda_device;
#[cfg(feature = "cuda")]
pub use self::cuda_device::CUDA;
