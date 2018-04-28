use primitiv_sys as _primitiv;
use std::ptr;
use ApiResult;
use devices::AnyDevice;
use Result;
use Shape;
use Wrap;

/// Value with any dimensions.
#[derive(Debug)]
pub struct Tensor {
    inner: *mut _primitiv::primitivTensor_t,
    owned: bool,
}

impl_wrap!(Tensor, primitivTensor_t);
impl_drop!(Tensor, primitivDeleteTensor);

impl Tensor {
    /// Creates an invalid Tensor object.
    pub fn new() -> Self {
        unsafe {
            let mut tensor_ptr: *mut _primitiv::primitivTensor_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCreateTensor(&mut tensor_ptr));
            assert!(!tensor_ptr.is_null());
            Tensor {
                inner: tensor_ptr,
                owned: true,
            }
        }
    }

    /// Check whether the object is valid or not.
    pub fn valid(&self) -> bool {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(_primitiv::primitivIsValidTensor(
                self.as_ptr(),
                &mut retval as *mut _,
            ));
            retval == 1
        }
    }

    /// Returns shape of the Tensor.
    pub fn shape(&self) -> Shape {
        unsafe {
            let mut shape_ptr: *mut _primitiv::primitivShape_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivGetTensorShape(
                self.as_ptr(),
                &mut shape_ptr,
            ));
            assert!(!shape_ptr.is_null());
            Shape::from_raw(shape_ptr, true)
        }
    }

    /// Returns the Device object related to the internal memory.
    pub fn device(&self) -> AnyDevice {
        unsafe {
            let mut device_ptr: *mut _primitiv::primitivDevice_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivGetDeviceFromTensor(
                self.as_ptr(),
                &mut device_ptr,
            ));
            assert!(!device_ptr.is_null());
            AnyDevice::from_raw(device_ptr, false)
        }
    }

    /// Retrieves one internal value in the tensor.
    ///
    /// Remark: This function can be used only when the tensor is a scalar and non-minibatched
    /// (i.e., shape() == Shape()).
    pub fn to_float(&self) -> f32 {
        unsafe {
            let mut retval: f32 = 0.0;
            check_api_status!(_primitiv::primitivEvaluateTensorAsFloat(
                self.as_ptr(),
                &mut retval as *mut _,
            ));
            retval
        }
    }

    /// Retrieves internal values in the tensor as a vector.
    /// Remark: Each resulting values a re ordered by the column-major order, and the batch size is
    /// assumed as the last dimension of the tensor.
    pub fn to_vector(&self) -> Vec<f32> {
        unsafe {
            // Use a vector as a C-style array because it must be a contiguous array actually.
            // See: https://doc.rust-lang.org/book/first-edition/vectors.html
            let mut size: usize = 0;
            check_api_status!(_primitiv::primitivEvaluateTensorAsArray(
                self.as_ptr(),
                ptr::null_mut(),
                &mut size as *mut _,
            ));
            let mut retval = vec![0f32; size];
            check_api_status!(_primitiv::primitivEvaluateTensorAsArray(
                self.as_ptr(),
                retval.as_mut_ptr(),
                &mut size as *mut _,
            ));
            retval
        }
    }

    /// Retrieves argmax indices along an axis.
    pub fn argmax(&self, dim: u32) -> Vec<u32> {
        unsafe {
            let mut size: usize = 0;
            check_api_status!(_primitiv::primitivGetTensorArgmax(
                self.as_ptr(),
                dim,
                ptr::null_mut(),
                &mut size as *mut _,
            ));
            let mut retval = vec![0u32; size];
            check_api_status!(_primitiv::primitivGetTensorArgmax(
                self.as_ptr(),
                dim,
                retval.as_mut_ptr(),
                &mut size as *mut _,
            ));
            retval
        }
    }

    /// Retrieves argmin indices along an axis.
    pub fn argmin(&self, dim: u32) -> Vec<u32> {
        unsafe {
            let mut size: usize = 0;
            check_api_status!(_primitiv::primitivGetTensorArgmin(
                self.as_ptr(),
                dim,
                ptr::null_mut(),
                &mut size as *mut _,
            ));
            let mut retval = vec![0u32; size];
            check_api_status!(_primitiv::primitivGetTensorArgmin(
                self.as_ptr(),
                dim,
                retval.as_mut_ptr(),
                &mut size as *mut _,
            ));
            retval
        }
    }

    /// Reset internal values using a constant.
    pub fn reset(&mut self, k: f32) {
        unsafe {
            check_api_status!(_primitiv::primitivResetTensor(self.as_mut_ptr(), k));
        }
    }

    /// Reset internal values using a slice.
    pub fn reset_by_slice(&mut self, values: &[f32]) {
        unsafe {
            check_api_status!(_primitiv::primitivResetTensorByArray(
                self.as_mut_ptr(),
                values.as_ptr() as *const _,
            ));
        }
    }

    /// Returns a tensor which have the same values and different shape.
    pub fn reshape(&self, new_shape: &Shape) -> Self {
        unsafe {
            let mut tensor_ptr: *mut _primitiv::primitivTensor_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivReshapeTensor(
                self.as_ptr(),
                new_shape.as_ptr(),
                &mut tensor_ptr,
            ));
            assert!(!tensor_ptr.is_null());
            Tensor {
                inner: tensor_ptr,
                owned: true,
            }
        }
    }

    /// Returns a flattened tensor.
    pub fn flatten(&self) -> Self {
        unsafe {
            let mut tensor_ptr: *mut _primitiv::primitivTensor_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivFlattenTensor(
                self.as_ptr(),
                &mut tensor_ptr,
            ));
            assert!(!tensor_ptr.is_null());
            Tensor {
                inner: tensor_ptr,
                owned: true,
            }
        }
    }

    /// Directly multiplies a constant.
    pub fn inplace_multiply_const(&mut self, k: f32) -> &mut Self {
        unsafe {
            check_api_status!(_primitiv::primitivMultiplyTensorByConstantInplace(
                self.as_mut_ptr(),
                k,
            ));
            self
        }
    }

    /// Directly adds a value.
    pub fn inplace_add(&mut self, x: &Tensor) -> &mut Self {
        unsafe {
            check_api_status!(_primitiv::primitivAddTensorInplace(
                self.as_mut_ptr(),
                x.as_ptr(),
            ));
            self
        }
    }

    /// Directly subtracts a value.
    pub fn inplace_subtract(&mut self, x: &Tensor) -> &mut Self {
        unsafe {
            check_api_status!(_primitiv::primitivSubtractTensorInplace(
                self.as_mut_ptr(),
                x.as_ptr(),
            ));
            self
        }
    }
}

impl Clone for Tensor {
    #[inline]
    fn clone(&self) -> Self {
        unsafe {
            let mut tensor_ptr: *mut _primitiv::primitivTensor_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCloneTensor(
                self.as_ptr(),
                &mut tensor_ptr,
            ));
            Tensor::from_raw(tensor_ptr, true)
        }
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        unsafe {
            check_api_status!(_primitiv::primitivDeleteTensor(self.inner));
            let mut tensor_ptr: *mut _primitiv::primitivTensor_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCloneTensor(
                source.as_ptr(),
                &mut tensor_ptr,
            ));
            self.inner = tensor_ptr;
        }
    }
}

impl AsRef<Tensor> for Tensor {
    #[inline]
    fn as_ref(&self) -> &Tensor {
        self
    }
}

impl Default for Tensor {
    fn default() -> Tensor {
        Tensor::new()
    }
}
