extern crate primitiv_sys as _primitiv;

use std::ops;
use std::ptr;
use device::{AnyDevice, Device};
use DataType;
use Expr;
use Node;
use Parameter;
use Shape;
use Wrap;

macro_rules! impl_bin_node_op {
    ($name:ident,
     $op_fn:ident,
     $api_nc_fn:ident,
     $api_cn_fn:ident,
     $api_nn_fn:ident) => {
        impl<T: DataType> ops::$name<Expr<T>> for Node {
            type Output = Node;

            fn $op_fn(self, rhs: Expr<T>) -> Node {
                unsafe {
                    Node::from_inner_ptr(_primitiv::$api_nc_fn(
                        self.as_inner_ptr(),
                        rhs.expr.into(),
                    ))
                }
            }
        }

        impl<T: DataType> ops::$name<Node> for Expr<T> {
            type Output = Node;

            fn $op_fn(self, rhs: Node) -> Node {
                unsafe {
                    Node::from_inner_ptr(_primitiv::$api_cn_fn(
                        self.expr.into(),
                        rhs.as_inner_ptr(),
                    ))
                }
            }
        }

        impl ops::$name<Node> for Node {
            type Output = Node;

            fn $op_fn(self, rhs: Node) -> Node {
                unsafe {
                    Node::from_inner_ptr(_primitiv::$api_nn_fn(
                        self.as_inner_ptr(),
                        rhs.as_inner_ptr(),
                    ))
                }
            }
        }
    }
}

impl_bin_node_op!(
    Add,
    add,
    primitiv_op_node_add_const,
    primitiv_op_const_add_node,
    primitiv_op_node_add_node
);
impl_bin_node_op!(
    Sub,
    sub,
    primitiv_op_node_sub_const,
    primitiv_op_const_sub_node,
    primitiv_op_node_sub_node
);
impl_bin_node_op!(
    Mul,
    mul,
    primitiv_op_node_mul_const,
    primitiv_op_const_mul_node,
    primitiv_op_node_mul_node
);
impl_bin_node_op!(
    Div,
    div,
    primitiv_op_node_div_const,
    primitiv_op_const_div_node,
    primitiv_op_node_div_node
);

pub fn input(shape: &Shape, data: &[f32]) -> Node {
    input_with_device::<AnyDevice>(shape, data, None)
}

pub fn input_with_device<D: Device>(shape: &Shape, data: &[f32], device: Option<&mut D>) -> Node {
    unsafe {
        Node::from_inner_ptr(_primitiv::primitiv_node_op_input(
            shape.as_inner_ptr(),
            data.as_ptr() as *const _,
            data.len(),
            device.map(|d| d.as_inner_mut_ptr()).unwrap_or(
                ptr::null_mut(),
            ),
        ))
    }
}

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

pub fn matmul(a: &Node, b: &Node) -> Node {
    unsafe {
        Node::from_inner_ptr(_primitiv::primitiv_node_op_matmul(
            a.as_inner_ptr(),
            b.as_inner_ptr(),
        ))
    }
}
