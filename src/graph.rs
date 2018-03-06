use primitiv_sys as _primitiv;
use std::ffi::CString;
use std::ptr;
use devices::AnyDevice;
use ApiResult;
use Shape;
use Tensor;
use Wrap;

/// Pointer of a node in the computation graph.
#[derive(Debug)]
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

    /// Returns corresponding Graph object.
    pub fn graph(&self) -> Graph {
        unsafe {
            let mut graph_ptr: *mut _primitiv::primitivGraph_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivGetGraphFromNode(
                self.as_ptr(),
                &mut graph_ptr,
            ));
            assert!(!graph_ptr.is_null());
            Graph::from_raw(graph_ptr, false)
        }
    }

    /// Returns whether the node is valid or not.
    pub fn valid(&self) -> bool {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(_primitiv::primitivIsValidNode(
                self.as_ptr(),
                &mut retval as *mut _,
            ));
            retval == 1
        }
    }

    /// Returns the operator ID.
    pub fn operator_id(&self) -> u32 {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(_primitiv::primitivGetNodeOperatorId(
                self.as_ptr(),
                &mut retval as *mut _,
            ));
            retval
        }
    }

    /// Returns the value ID of the operator.
    pub fn value_id(&self) -> u32 {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(_primitiv::primitivGetNodeValueId(
                self.as_ptr(),
                &mut retval as *mut _,
            ));
            retval
        }
    }

    /// Returns shape of the node.
    pub fn shape(&self) -> Shape {
        unsafe {
            let mut shape_ptr: *mut _primitiv::primitivShape_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivGetNodeShape(
                self.as_ptr(),
                &mut shape_ptr,
            ));
            assert!(!shape_ptr.is_null());
            Shape::from_raw(shape_ptr, true)
        }
    }

    /// Returns device of the node.
    pub fn device(&self) -> AnyDevice {
        unsafe {
            let mut device_ptr: *mut _primitiv::primitivDevice_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivGetDeviceFromNode(
                self.as_ptr(),
                &mut device_ptr,
            ));
            assert!(!device_ptr.is_null());
            AnyDevice::from_raw(device_ptr, false)
        }
    }

    /// Calculates the value of this node and returns a float.
    ///
    /// Remark: This function calls Graph::forward() internally.
    /// This function can be used only when the Node has a scalar and non-minibatched shape
    /// (i.e., shape() == Shape())
    pub fn to_float(&self) -> f32 {
        unsafe {
            let mut retval: f32 = 0.0;
            check_api_status!(_primitiv::primitivEvaluateNodeAsFloat(
                self.as_ptr(),
                &mut retval as *mut _,
            ));
            retval
        }
    }

    /// Calculates the value of this node and returns a list of float.
    pub fn to_vector(&self) -> Vec<f32> {
        unsafe {
            // Use a vector as a C-style array because it must be a contiguous array actually.
            // See: https://doc.rust-lang.org/book/first-edition/vectors.html
            let mut size: usize = 0;
            check_api_status!(_primitiv::primitivEvaluateNodeAsArray(
                self.as_ptr(),
                ptr::null_mut(),
                &mut size as *mut _,
            ));
            let mut retval = vec![0f32; size];
            check_api_status!(_primitiv::primitivEvaluateNodeAsArray(
                self.as_ptr(),
                retval.as_mut_ptr(),
                &mut size as *mut _,
            ));
            retval
        }
    }

    /// Returns argmax indices along an axis of this node.
    pub fn argmax(&self, dim: u32) -> Vec<u32> {
        unsafe {
            let mut size: usize = 0;
            check_api_status!(_primitiv::primitivGetNodeArgmax(
                self.as_ptr(),
                dim,
                ptr::null_mut(),
                &mut size as *mut _,
            ));
            let mut retval = vec![0u32; size];
            check_api_status!(_primitiv::primitivGetNodeArgmax(
                self.as_ptr(),
                dim,
                retval.as_mut_ptr(),
                &mut size as *mut _,
            ));
            retval
        }
    }

    /// Returns argmin indices along an axis of this node.
    pub fn argmin(&self, dim: u32) -> Vec<u32> {
        unsafe {
            let mut size: usize = 0;
            check_api_status!(_primitiv::primitivGetNodeArgmin(
                self.as_ptr(),
                dim,
                ptr::null_mut(),
                &mut size as *mut _,
            ));
            let mut retval = vec![0u32; size];
            check_api_status!(_primitiv::primitivGetNodeArgmin(
                self.as_ptr(),
                dim,
                retval.as_mut_ptr(),
                &mut size as *mut _,
            ));
            retval
        }
    }

    /// Executes the backward operation from this node.
    pub fn backward(&self) {
        unsafe {
            check_api_status!(_primitiv::primitivExecuteNodeBackward(self.as_ptr()));
        }
    }
}

impl Clone for Node {
    #[inline]
    fn clone(&self) -> Self {
        unsafe {
            let mut node_ptr: *mut _primitiv::primitivNode_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCloneNode(self.as_ptr(), &mut node_ptr));
            Node::from_raw(node_ptr, true)
        }
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        unsafe {
            check_api_status!(_primitiv::primitivDeleteNode(self.inner));
            let mut node_ptr: *mut _primitiv::primitivNode_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCloneNode(source.as_ptr(), &mut node_ptr));
            self.inner = node_ptr;
        }
    }
}

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
            Graph {
                inner: inner,
                owned: true,
            }
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
    /// Remark: After calling this method, all Node objects supplied by the graph itself is
    /// invalidated.
    pub fn clear(&mut self) {
        unsafe {
            check_api_status!(_primitiv::primitivClearGraph(self.as_mut_ptr()));
        }
    }

    /// Calculates the value of given node.
    pub fn forward(&mut self, node: &Node) -> Tensor {
        unsafe {
            let mut tensor_ptr: *const _primitiv::primitivTensor_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivExecuteGraphForward(
                self.as_mut_ptr(),
                node.as_ptr(),
                &mut tensor_ptr,
            ));
            assert!(!tensor_ptr.is_null());
            Tensor::from_raw(tensor_ptr as *mut _, false)
        }
    }

    /// Calculates the backpropagation.
    pub fn backward(&mut self, node: &Node) {
        unsafe {
            check_api_status!(_primitiv::primitivExecuteGraphBackward(
                self.as_mut_ptr(),
                node.as_ptr(),
            ));
        }
    }

    /// Retrieves the shape of the node.
    pub fn get_shape(&self, node: &Node) -> Shape {
        unsafe {
            let mut shape_ptr: *mut _primitiv::primitivShape_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivGetGraphShape(
                self.as_ptr(),
                node.as_ptr(),
                &mut shape_ptr,
            ));
            assert!(!shape_ptr.is_null());
            Shape::from_raw(shape_ptr, true)
        }
    }

    /// Retrieves the device of the node.
    pub fn get_device(&self, node: &Node) -> AnyDevice {
        unsafe {
            let mut device_ptr: *mut _primitiv::primitivDevice_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivGetDeviceFromGraph(
                self.as_ptr(),
                node.as_ptr(),
                &mut device_ptr,
            ));
            assert!(!device_ptr.is_null());
            AnyDevice::from_raw(device_ptr, false)
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
            let format_c = CString::new(format).unwrap();
            let format_ptr = format_c.as_ptr();
            let mut size: usize = 0;
            check_api_status!(_primitiv::primitivDumpGraph(
                self.as_ptr(),
                format_ptr,
                ptr::null_mut(),
                &mut size as *mut _,
            ));
            let buffer = CString::new(Vec::with_capacity(size)).unwrap().into_raw();
            check_api_status!(_primitiv::primitivDumpGraph(
                self.as_ptr(),
                format_ptr,
                buffer,
                &mut size as *mut _,
            ));
            CString::from_raw(buffer).into_string().unwrap()
        }
    }

    /// Returns the number of operators in the computation graph.
    pub fn num_operators(&self) -> u32 {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(_primitiv::primitivGetGraphNumOperators(
                self.as_ptr(),
                &mut retval as *mut _,
            ));
            retval
        }
    }
}
