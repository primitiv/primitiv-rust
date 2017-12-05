extern crate primitiv_sys as _primitiv;

use Wrap;

#[derive(Debug)]
pub struct Node {
    inner: *mut _primitiv::primitiv_Node,
}

impl_wrap!(Node, primitiv_Node);
impl_new!(Node, primitiv_Node_new);
impl_drop!(Node, primitiv_Node_delete);

#[derive(Debug)]
pub struct Graph {
    inner: *mut _primitiv::primitiv_Graph,
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
