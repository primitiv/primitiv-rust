use primitiv_sys as _primitiv;
use std::ops;
use std::ptr;
use device::{AnyDevice, Device};
use DataType;
use Expr;
use Node;
use Parameter;
use Shape;
use Status;
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
                let mut status = Status::new();
                unsafe {
                    let node = Node::from_inner_ptr(_primitiv::$api_nc_fn(
                        self.as_inner_ptr(),
                        rhs.expr.into(),
                        status.as_inner_mut_ptr(),
                    ));
                    status.into_result().unwrap();
                    node
                }
            }
        }

        impl<T: DataType> ops::$name<Node> for Expr<T> {
            type Output = Node;

            fn $op_fn(self, rhs: Node) -> Node {
                let mut status = Status::new();
                unsafe {
                    let node = Node::from_inner_ptr(_primitiv::$api_cn_fn(
                        self.expr.into(),
                        rhs.as_inner_ptr(),
                        status.as_inner_mut_ptr(),
                    ));
                    status.into_result().unwrap();
                    node
                }
            }
        }

        impl ops::$name<Node> for Node {
            type Output = Node;

            fn $op_fn(self, rhs: Node) -> Node {
                let mut status = Status::new();
                unsafe {
                    let node = Node::from_inner_ptr(_primitiv::$api_nn_fn(
                        self.as_inner_ptr(),
                        rhs.as_inner_ptr(),
                        status.as_inner_mut_ptr(),
                    ));
                    status.into_result().unwrap();
                    node
                }
            }
        }
    }
}

impl_bin_node_op!(
    Add,
    add,
    safe_primitiv_node_func_add_node_const,
    safe_primitiv_node_func_add_const_node,
    safe_primitiv_node_func_add_node_node
);
impl_bin_node_op!(
    Sub,
    sub,
    safe_primitiv_node_func_subtract_node_const,
    safe_primitiv_node_func_subtract_const_node,
    safe_primitiv_node_func_subtract_node_node
);
impl_bin_node_op!(
    Mul,
    mul,
    safe_primitiv_node_func_multiply_node_const,
    safe_primitiv_node_func_multiply_const_node,
    safe_primitiv_node_func_multiply_node_node
);
impl_bin_node_op!(
    Div,
    div,
    safe_primitiv_node_func_divide_node_const,
    safe_primitiv_node_func_divide_const_node,
    safe_primitiv_node_func_divide_node_node
);

pub fn input(shape: &Shape, data: &[f32]) -> Node {
    input_with_device::<AnyDevice>(shape, data, None)
}

pub fn input_with_device<D: Device>(shape: &Shape, data: &[f32], device: Option<&mut D>) -> Node {
    let mut status = Status::new();
    unsafe {
        let node = Node::from_inner_ptr(_primitiv::safe_primitiv_node_func_input(
            shape.as_inner_ptr(),
            data.as_ptr() as *const _,
            data.len(),
            device.map(|d| d.as_inner_mut_ptr()).unwrap_or(
                ptr::null_mut(),
            ),
            ptr::null_mut(),
            status.as_inner_mut_ptr(),
        ));
        status.into_result().unwrap();
        node
    }
}

pub fn parameter(param: &mut Parameter) -> Node {
    let mut status = Status::new();
    unsafe {
        let node = Node::from_inner_ptr(_primitiv::safe_primitiv_node_func_parameter(
            param.as_inner_mut_ptr(),
            ptr::null_mut(),
            status.as_inner_mut_ptr(),
        ));
        status.into_result().unwrap();
        node
    }
}

pub fn tanh(x: &Node) -> Node {
    let mut status = Status::new();
    unsafe {
        let node = Node::from_inner_ptr(_primitiv::safe_primitiv_node_func_tanh(
            x.as_inner_ptr(),
            status.as_inner_mut_ptr(),
        ));
        status.into_result().unwrap();
        node
    }
}

pub fn matmul(a: &Node, b: &Node) -> Node {
    let mut status = Status::new();
    unsafe {
        let node = Node::from_inner_ptr(_primitiv::safe_primitiv_node_func_matmul(
            a.as_inner_ptr(),
            b.as_inner_ptr(),
            status.as_inner_mut_ptr(),
        ));
        status.into_result().unwrap();
        node
    }
}
