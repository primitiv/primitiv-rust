use primitiv_sys as _primitiv;
use std::cmp::{Eq, PartialEq};
use std::ffi::CString;
use std::fmt;
use std::ptr;
use ApiResult;
use Result;
use Wrap;

/// Data structure to represent the shape of the node.
#[derive(Debug)]
pub struct Shape {
    inner: *mut _primitiv::primitivShape_t,
}

impl_wrap_owned!(Shape, primitivShape_t);
impl_drop!(Shape, primitivDeleteShape);

impl Shape {
    /// Creates a new scalar Shape object.
    pub fn new() -> Self {
        unsafe {
            let mut shape_ptr: *mut _primitiv::primitivShape_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCreateShape(&mut shape_ptr));
            assert!(!shape_ptr.is_null());
            Shape { inner: shape_ptr }
        }
    }

    /// Creates a new Shape object.
    pub fn from_dims(dims: &[u32], batch: u32) -> Self {
        unsafe {
            let mut shape_ptr: *mut _primitiv::primitivShape_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCreateShapeWithDims(
                dims.as_ptr(),
                dims.len(),
                batch,
                &mut shape_ptr,
            ));
            Shape { inner: shape_ptr }
        }
    }

    /// Returns the size of the i-th dimension.
    pub fn at(&self, i: u32) -> u32 {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(_primitiv::primitivGetShapeDimSize(
                self.as_ptr(),
                i,
                &mut retval as *mut _,
            ));
            retval
        }
    }

    /// Returns the dimension array.
    pub fn dims(&self) -> Vec<u32> {
        unsafe {
            let mut size: usize = 0;
            check_api_status!(_primitiv::primitivGetShapeDims(
                self.as_ptr(),
                ptr::null_mut(),
                &mut size as *mut _,
            ));
            let mut retval = vec![0u32; size];
            check_api_status!(_primitiv::primitivGetShapeDims(
                self.as_ptr(),
                retval.as_mut_ptr(),
                &mut size as *mut _,
            ));
            retval
        }
    }

    /// Returns the depth (length of non-1 dimensions) of the shape.
    pub fn depth(&self) -> u32 {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(_primitiv::primitivGetShapeDepth(
                self.as_ptr(),
                &mut retval as *mut _,
            ));
            retval
        }
    }

    /// Returns the batch size.
    pub fn batch(&self) -> u32 {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(_primitiv::primitivGetShapeBatchSize(
                self.as_ptr(),
                &mut retval as *mut _,
            ));
            retval
        }
    }

    /// Returns the number of elements in each sample.
    /// This value is equal to the product of all dimensions.
    pub fn volume(&self) -> u32 {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(_primitiv::primitivGetShapeVolume(
                self.as_ptr(),
                &mut retval as *mut _,
            ));
            retval
        }
    }

    /// Returns the number of elements in 1 to specified dim.
    pub fn lower_volume(&self, dim: u32) -> u32 {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(_primitiv::primitivGetShapeLowerVolume(
                self.as_ptr(),
                dim,
                &mut retval as *mut _,
            ));
            retval
        }
    }

    /// Returns the number of elements in all samples of the mini-batch.
    /// This value is equal to batch() * volume().
    pub fn size(&self) -> usize {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(_primitiv::primitivGetShapeSize(
                self.as_ptr(),
                &mut retval as *mut _,
            ));
            retval as usize
        }
    }

    /// Checks whether the shape has minibatch or not.
    pub fn has_batch(&self) -> bool {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(_primitiv::primitivHasShapeBatch(
                self.as_ptr(),
                &mut retval as *mut _,
            ));
            retval == 1
        }
    }

    /// Checks whether two batch size is compatible (broadcastable) or not.
    pub fn has_compatible_batch(&self, other: &Shape) -> bool {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(_primitiv::primitivHasShapeCompatibleBatch(
                self.as_ptr(),
                other.as_ptr(),
                &mut retval as *mut _,
            ));
            retval == 1
        }
    }

    /// Checks whether the shape is a scalar or not.
    pub fn is_scalar(&self) -> bool {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(_primitiv::primitivIsShapeScalar(
                self.as_ptr(),
                &mut retval as *mut _,
            ));
            retval == 1
        }
    }

    /// Checks whether the shape is a column vector or not.
    pub fn is_column_vector(&self) -> bool {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(_primitiv::primitivIsShapeColumnVector(
                self.as_ptr(),
                &mut retval as *mut _,
            ));
            retval == 1
        }
    }

    /// Checks whether the shape is a vector or a matrix, or not.
    pub fn is_matrix(&self) -> bool {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(_primitiv::primitivIsShapeMatrix(
                self.as_ptr(),
                &mut retval as *mut _,
            ));
            retval == 1
        }
    }

    /// Checks whether two shapes have completely same dimensions.
    pub fn has_same_dims(&self, other: &Shape) -> bool {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(_primitiv::primitivHasShapeSameDims(
                self.as_ptr(),
                other.as_ptr(),
                &mut retval as *mut _,
            ));
            retval == 1
        }
    }

    /// Checks whether two shapes have same dimensions without an axis. (LOO: leave one out)
    pub fn has_same_loo_dims(&self, other: &Shape, dim: u32) -> bool {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(_primitiv::primitivHasShapeSameLooDims(
                self.as_ptr(),
                other.as_ptr(),
                dim,
                &mut retval as *mut _,
            ));
            retval == 1
        }
    }

    /// Creates a new shape which have one different dimension.
    pub fn resize_dim(&self, dim: u32, m: u32) -> Shape {
        unsafe {
            let mut shape_ptr: *mut _primitiv::primitivShape_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivResizeShapeDim(
                self.as_ptr(),
                dim,
                m,
                &mut shape_ptr,
            ));
            Shape { inner: shape_ptr }
        }
    }

    /// Creates a new shape which have specified batch size.
    pub fn resize_batch(&self, batch: u32) -> Shape {
        unsafe {
            let mut shape_ptr: *mut _primitiv::primitivShape_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivResizeShapeBatch(
                self.as_ptr(),
                batch,
                &mut shape_ptr,
            ));
            Shape { inner: shape_ptr }
        }
    }

    /// Directly updates a specified dimension.
    pub fn update_dim(&mut self, dim: u32, m: u32) {
        unsafe {
            check_api_status!(_primitiv::primitivUpdateShapeDim(self.as_mut_ptr(), dim, m));
        }
    }

    /// Directly updates the batch size.
    pub fn update_batch(&mut self, batch: u32) {
        unsafe {
            check_api_status!(_primitiv::primitivUpdateShapeBatchSize(
                self.as_mut_ptr(),
                batch,
            ));
        }
    }
}

impl Clone for Shape {
    #[inline]
    fn clone(&self) -> Self {
        unsafe {
            let mut shape_ptr: *mut _primitiv::primitivShape_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCloneShape(self.as_ptr(), &mut shape_ptr));
            Shape::from_raw(shape_ptr, true)
        }
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        unsafe {
            check_api_status!(_primitiv::primitivDeleteShape(self.inner));
            let mut shape_ptr: *mut _primitiv::primitivShape_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCloneShape(
                source.as_ptr(),
                &mut shape_ptr,
            ));
            self.inner = shape_ptr;
        }
    }
}

impl PartialEq for Shape {
    fn eq(&self, other: &Shape) -> bool {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(_primitiv::primitivIsShapeEqualTo(
                self.as_ptr(),
                other.as_ptr(),
                &mut retval as *mut _,
            ));
            retval == 1
        }
    }
}

impl Eq for Shape {}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        unsafe {
            let mut size: usize = 0;
            check_api_status!(_primitiv::primitivRepresentShapeAsString(
                self.as_ptr(),
                ptr::null_mut(),
                &mut size as *mut _,
            ));
            let buffer = CString::new(Vec::with_capacity(size)).unwrap().into_raw();
            check_api_status!(_primitiv::primitivRepresentShapeAsString(
                self.as_ptr(),
                buffer,
                &mut size as *mut _,
            ));
            f.write_str(CString::from_raw(buffer).to_str().unwrap())
        }
    }
}

macro_rules! impl_array_into_shape {
    ($num:expr) => {
        impl Into<Shape> for [u32; $num] {
            fn into(self) -> Shape {
                Shape::from_dims(&self, 1)
            }
        }
    }
}
impl_array_into_shape!(0);
impl_array_into_shape!(1);
impl_array_into_shape!(2);
impl_array_into_shape!(3);
impl_array_into_shape!(4);
impl_array_into_shape!(5);
impl_array_into_shape!(6);
impl_array_into_shape!(7);
impl_array_into_shape!(8);

macro_rules! impl_tuple_into_shape {
    ($num:expr) => {
        impl Into<Shape> for ([u32; $num], u32) {
            fn into(self) -> Shape {
                Shape::from_dims(&self.0, self.1)
            }
        }
    }
}
impl_tuple_into_shape!(0);
impl_tuple_into_shape!(1);
impl_tuple_into_shape!(2);
impl_tuple_into_shape!(3);
impl_tuple_into_shape!(4);
impl_tuple_into_shape!(5);
impl_tuple_into_shape!(6);
impl_tuple_into_shape!(7);
impl_tuple_into_shape!(8);
