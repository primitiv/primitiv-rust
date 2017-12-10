use primitiv_sys as _primitiv;
use Shape;
use Status;
use Wrap;

#[derive(Clone, Debug)]
pub struct Node {
    inner: *mut _primitiv::primitiv_Node,
}

impl_wrap!(Node, primitiv_Node);
impl_new!(Node, safe_primitiv_Node_new);
// impl_drop!(Node, safe_primitiv_Node_delete);

impl Node {
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
}

impl AsRef<Node> for Node {
    #[inline]
    fn as_ref(&self) -> &Node {
        self
    }
}

#[derive(Debug)]
pub struct Graph {
    inner: *mut _primitiv::primitiv_Graph,
}

impl_wrap!(Graph, primitiv_Graph);
impl_new!(Graph, safe_primitiv_Graph_new);
impl_drop!(Graph, safe_primitiv_Graph_delete);

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
