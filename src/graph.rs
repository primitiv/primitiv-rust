extern crate primitiv_sys as _primitiv;

use Wrap;

#[derive(Debug)]
pub struct Node {
    node: _primitiv::primitiv_Node,
    // inner: *mut _primitiv::primitiv_Graph,
}

impl Drop for Node {
    fn drop(&mut self) {}
}

#[derive(Debug)]
pub struct Graph {
    inner: *mut _primitiv::primitiv_Graph,
}

impl_wrap!(Graph, primitiv_Graph);
impl_new!(Graph, primitiv_Graph_new);
impl_drop!(Graph, primitiv_Graph_delete);

impl Graph {
    pub fn set_default(graph: &Self) {
        unsafe {
            _primitiv::primitiv_Graph_set_default(graph.inner);
        }
    }
}
