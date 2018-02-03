use primitiv_sys as _primitiv;
use std::ffi::CString;
use std::ptr;
use AnyDevice;
use ApiResult;
// use Device;
use Result;
use Shape;
use Status;
use Tensor;
// use Code;
use Wrap;

#[derive(Clone, Debug)]
pub struct Node {
    inner: *mut _primitiv::primitivNode_t,
}

impl_wrap_owned!(Node, primitivNode_t);
impl_drop!(Node, primitivDeleteNode);

impl Node {
    /// Creates a new Node object.
    pub fn new() -> Self {
        unsafe {
            let mut node_ptr: *mut _primitiv::primitivNode_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCreateNode(&mut node_ptr));
            assert!(!node_ptr.is_null());
            Node { inner: node_ptr }
        }
    }

    /// Returns whether the node is valid or not.
    pub fn valid(&self) -> bool {
        unsafe {
            let retval: u32 = 0;
            check_api_status!(_primitiv::primitivIsValidNode(self.as_ptr(), retval as *mut _));
            retval == 1
        }
    }

    /// Returns the operator ID.
    pub fn operator_id(&self) -> u32 {
        unsafe {
            let id: u32 = 0;
            check_api_status!(_primitiv::primitivGetNodeOperatorId(self.as_ptr(), id as *mut _));
            id
        }
    }

    /// Returns the value ID of the operator.
    pub fn value_id(&self) -> u32 {
        unsafe {
            let id: u32 = 0;
            check_api_status!(_primitiv::primitivGetNodeValueId(self.as_ptr(), id as *mut _));
            id
        }
    }

    /*
    pub fn shape(&self) -> Shape {
        unsafe {
            Shape::from_ptr(_primitiv::primitiv_Node_shape(self.as_ptr()) as
                *mut _)
        }
    }

    pub fn to_float(&self) -> f32 {
        let mut status = Status::new();
        unsafe {
            let value = _primitiv::safe_primitiv_Node_to_float(
                self.as_ptr(),
                status.as_mut_ptr(),
            );
            status.into_result().unwrap();
            value
        }
    }

    pub fn to_vector(&self) -> Vec<f32> {
        let mut status = Status::new();
        unsafe {
            // Use a vector as a C-style array because it must be a contiguous array actually.
            // See: https://doc.rust-lang.org/book/first-edition/vectors.html
            let mut v = vec![0f32; self.shape().size()];
            _primitiv::safe_primitiv_Node_to_array(
                self.as_ptr(),
                v.as_mut_ptr(),
                status.as_mut_ptr(),
            );
            status.into_result().unwrap();
            v
        }
    }

    pub fn backward(&self) {
        let mut status = Status::new();
        unsafe {
            _primitiv::safe_primitiv_Node_backward(self.as_ptr(), status.as_mut_ptr());
            status.into_result().unwrap();
        }
    }

    */
}

/*
impl From<Node> for Node {
    fn from(Node: node) -> Self {
        let mut device_ptr: *mut _primitiv::primitiv_Device = ptr::null_mut();
        let status = _primitiv::primitiv_Device_get_default(&mut device_ptr);
    }
}
*/

impl AsRef<Node> for Node {
    #[inline]
    fn as_ref(&self) -> &Node {
        self
    }
}

/// Computation graph.
#[derive(Debug)]
pub struct Graph {
    inner: *mut _primitiv::primitivGraph_t,
    owned: bool,
}

impl_wrap!(Graph, primitivGraph_t);
impl_drop!(Graph, primitivDeleteGraph);

impl Graph {

    /// Creates a new Graph object.
    pub fn new() -> Self {
        unsafe {
            let mut inner: *mut _primitiv::primitivGraph_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCreateGraph(&mut inner));
            assert!(!inner.is_null());
            Graph { inner: inner, owned: true }
        }
    }

    /// Specifies a new default graph.
    pub fn set_default(graph: &mut Self) {
        unsafe {
            check_api_status!(_primitiv::primitivSetDefaultGraph(graph.as_mut_ptr()));
        }
    }

    /// Clear all operators in the graph.
    ///
    /// Remark: After calling this method, all Node objects supplied by the graph itself is invalidated.
    pub fn clear(&mut self) {
        unsafe {
            check_api_status!(_primitiv::primitivClearGraph(self.as_mut_ptr()));
        }
    }

    /// Calculates the value of given node.
    pub fn forward(&mut self, node: &Node) -> Tensor {
        unsafe {
            let mut tensor_ptr: *const _primitiv::primitivTensor_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivExecuteGraphForward(self.as_mut_ptr(), node.as_ptr(), &mut tensor_ptr));
            Tensor::from_raw(tensor_ptr as *mut _, false)
        }
    }

    /// Calculates the backpropagation.
    pub fn backward(&mut self, node: &Node) {
        unsafe {
            check_api_status!(_primitiv::primitivExecuteGraphBackward(self.as_mut_ptr(), node.as_ptr()));
        }
    }

    /// Retrieves the shape of the node.
    pub fn get_shape(&mut self, node: &Node) -> Shape {
        unsafe {
            let mut shape_ptr: *mut _primitiv::primitivShape_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivGetGraphShape(self.as_mut_ptr(), node.as_ptr(), &mut shape_ptr));
            Shape::from_raw(shape_ptr as *mut _, true)
        }
    }

    /// Retrieves the device of the node.
    pub fn get_device(&mut self, node: &Node) -> AnyDevice {
        unsafe {
            let mut device_ptr: *mut _primitiv::primitivDevice_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivGetDeviceFromGraph(self.as_mut_ptr(), node.as_ptr(), &mut device_ptr));
            AnyDevice::from_raw(device_ptr as *mut _, false)
        }
    }

    /// Dump internal graph structure.
    ///
    /// Available options:
    ///
    /// * “dot” … Graphviz’s dot format.
    ///
    pub fn dump(&self, format: &str) -> String {
        unsafe {
            let format_ptr = CString::new(format).unwrap().as_ptr();
            let size: u32 = 0;
            check_api_status!(_primitiv::primitivDumpGraph(self.as_ptr(), format_ptr, ptr::null_mut(), size as *mut _));
            let buffer = CString::new(Vec::with_capacity(size as usize)).unwrap().into_raw();
            check_api_status!(_primitiv::primitivDumpGraph(self.as_ptr(), format_ptr, buffer, size as *mut _));
            CString::from_raw(buffer).into_string().unwrap()
        }
    }

    /// Returns the number of operators in the computation graph.
    pub fn num_operators(&self, node: &Node) -> u32 {
        unsafe {
            let retval: u32 = 0;
            check_api_status!(_primitiv::primitivGetGraphNumOperators(self.as_ptr(), retval as *mut _));
            retval
        }
    }
}
