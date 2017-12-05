extern crate primitiv_sys as _primitiv;

use Node;
use Parameter;
use Wrap;

pub fn parameter(param: &mut Parameter) -> Node {
    unsafe {
        Node::from_inner_ptr(_primitiv::primitiv_node_op_parameter(
            param.as_inner_mut_ptr(),
        ))
    }
}

pub fn tanh(x: &Node) -> Node {
    unsafe { Node::from_inner_ptr(_primitiv::primitiv_node_op_tanh(x.as_inner_ptr())) }
}
