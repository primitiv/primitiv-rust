use primitiv_sys as _primitiv;
use std::cmp::{Eq, PartialEq};
use std::ffi::CString;
use std::fmt;
use std::ptr::{self, NonNull};
use ApiResult;
use Result;
use Wrap;

/// Data structure to represent the shape of the node.
#[derive(Debug)]
pub struct Shape {
    inner: NonNull<_primitiv::primitivShape_t>,
}

impl_wrap_owned!(Shape, primitivShape_t);
impl_drop!(Shape, primitivDeleteShape);

impl Shape {
    /// Creates a new scalar Shape object.
    pub fn new() -> Self {
        unsafe {
            let mut shape_ptr: *mut _primitiv::primitivShape_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCreateShape(&mut shape_ptr));
            Shape::from_raw(shape_ptr, true)
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
            Shape::from_raw(shape_ptr, true)
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
            Shape::from_raw(shape_ptr, true)
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
            Shape::from_raw(shape_ptr, true)
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
            check_api_status!(_primitiv::primitivDeleteShape(self.as_mut_ptr()));
            let mut shape_ptr: *mut _primitiv::primitivShape_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCloneShape(
                source.as_ptr(),
                &mut shape_ptr,
            ));
            self.inner = NonNull::new(shape_ptr).expect("pointer must not be null");
        }
    }
}

impl Default for Shape {
    fn default() -> Shape {
        Shape::new()
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
            let buffer = CString::new(vec![b'0'; size]).unwrap().into_raw();
            check_api_status!(_primitiv::primitivRepresentShapeAsString(
                self.as_ptr(),
                buffer,
                &mut size as *mut _,
            ));
            f.write_str(CString::from_raw(buffer).to_str().unwrap())
        }
    }
}

macro_rules! impl_shape_from_array {
    ($num:expr) => {
        impl From<[u32; $num]> for Shape {
            fn from(dims: [u32; $num]) -> Shape {
                Shape::from_dims(&dims, 1)
            }
        }
    };
}
impl_shape_from_array!(0);
impl_shape_from_array!(1);
impl_shape_from_array!(2);
impl_shape_from_array!(3);
impl_shape_from_array!(4);
impl_shape_from_array!(5);
impl_shape_from_array!(6);
impl_shape_from_array!(7);
impl_shape_from_array!(8);

macro_rules! impl_shape_from_tuple {
    ($num:expr) => {
        impl From<([u32; $num], u32)> for Shape {
            fn from(dims_with_batch: ([u32; $num], u32)) -> Shape {
                Shape::from_dims(&dims_with_batch.0, dims_with_batch.1)
            }
        }
    };
}
impl_shape_from_tuple!(0);
impl_shape_from_tuple!(1);
impl_shape_from_tuple!(2);
impl_shape_from_tuple!(3);
impl_shape_from_tuple!(4);
impl_shape_from_tuple!(5);
impl_shape_from_tuple!(6);
impl_shape_from_tuple!(7);
impl_shape_from_tuple!(8);
