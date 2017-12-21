use primitiv_sys as _primitiv;
// use Shape;
use Status;
use Result;
use ApiResult;
// use Code;
use Wrap;

#[derive(Clone, Debug)]
pub struct Node {
    inner: *mut _primitiv::primitiv_Node,
}

impl_wrap_owned!(Node, primitiv_Node);
impl_drop!(Node, primitiv_Node_delete);

impl Node {
    /// Creates a new Node object.
    pub fn new() -> Self {
        unsafe { Node { inner: _primitiv::primitiv_Node_new() } }
    }

    /// Returns whether the node is valid or not.
    pub fn valid(&self) -> bool {
        unsafe { _primitiv::primitiv_Node_valid(self.as_inner_ptr()) == 1 }
    }

    /// Returns the operator ID.
    pub fn operator_id(&self) -> Result<u32> {
        unsafe {
            let id: u32 = 0;
            Result::from_api_status(
                _primitiv::primitiv_Node_operator_id(self.as_inner_ptr(), id as *mut _),
                id,
            )
        }
    }

    /// Returns the value ID of the operator.
    pub fn value_id(&self) -> Result<u32> {
        unsafe {
            let id: u32 = 0;
            Result::from_api_status(
                _primitiv::primitiv_Node_value_id(self.as_inner_ptr(), id as *mut _),
                id,
            )
        }
    }

    /*
    pub fn shape(&self) -> Shape {
        unsafe {
            Shape::from_inner_ptr(_primitiv::primitiv_Node_shape(self.as_inner_ptr()) as
                *mut _)
        }
    }

    pub fn to_float(&self) -> f32 {
        let mut status = Status::new();
        unsafe {
            let value = _primitiv::safe_primitiv_Node_to_float(
                self.as_inner_ptr(),
                status.as_inner_mut_ptr(),
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
                self.as_inner_ptr(),
                v.as_mut_ptr(),
                status.as_inner_mut_ptr(),
            );
            status.into_result().unwrap();
            v
        }
    }

    pub fn backward(&self) {
        let mut status = Status::new();
        unsafe {
            _primitiv::safe_primitiv_Node_backward(self.as_inner_ptr(), status.as_inner_mut_ptr());
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

#[derive(Debug)]
pub struct Graph {
    inner: *mut _primitiv::primitiv_Graph,
    owned: bool,
}

impl_wrap!(Graph, primitiv_Graph);
impl_new!(Graph, primitiv_Graph_new);
impl_drop!(Graph, primitiv_Graph_delete);

impl Graph {
    pub fn clear(&mut self) {
        unsafe {
            _primitiv::primitiv_Graph_clear(self.as_inner_mut_ptr());
        }
    }

    pub fn set_default(graph: &mut Self) {
        unsafe {
            _primitiv::primitiv_Graph_set_default(graph.as_inner_mut_ptr());
        }
    }
}
