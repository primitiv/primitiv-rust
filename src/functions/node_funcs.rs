use primitiv_sys as _primitiv;
use std::ops;
use std::ptr;
use ApiResult;
use Device;
use devices::AnyDevice;
use Graph;
use Node;
use Parameter;
use Shape;
use Wrap;

macro_rules! node_func_body {
    ($api:ident, $($arg:expr),* ) => {
        unsafe {
            let mut node_ptr: *mut _primitiv::primitivNode_t = ptr::null_mut();
            check_api_status!(_primitiv::$api(
                $($arg),*,
                &mut node_ptr,
            ));
            assert!(!node_ptr.is_null());
            Node::from_raw(node_ptr, true)
        }
    }
}

macro_rules! impl_node_unary_func {
    ($name:ident, $api:ident) => {
        pub fn $name<N: AsRef<Node>>(x: N) -> Node {
            unsafe {
                let mut node_ptr: *mut _primitiv::primitivNode_t = ptr::null_mut();
                check_api_status!(_primitiv::$api(x.as_ref().as_ptr(), &mut node_ptr));
                assert!(!node_ptr.is_null());
                Node::from_raw(node_ptr, true)
            }
        }
    }
}

macro_rules! impl_node_scalar_op {
    ($scalar:ty,
     $name:ident,
     $op_fn:ident,
     $api_nc_fn:ident,
     $api_cn_fn:ident) => {
        impl ops::$name<$scalar> for Node {
            type Output = Node;

            fn $op_fn(self, rhs: $scalar) -> Node {
                unsafe {
                    let mut node_ptr: *mut _primitiv::primitivNode_t = ptr::null_mut();
                    check_api_status!(_primitiv::primitivApplyNodeAddXC(
                        self.as_ptr(),
                        rhs as f32,
                        &mut node_ptr,
                    ));
                    assert!(!node_ptr.is_null());
                    Node::from_raw(node_ptr, true)
                }
            }
        }

        impl ops::$name<Node> for $scalar {
            type Output = Node;

            fn $op_fn(self, rhs: Node) -> Node {
                unsafe {
                    let mut node_ptr: *mut _primitiv::primitivNode_t = ptr::null_mut();
                    check_api_status!(_primitiv::primitivApplyNodeAddCX(
                        self as f32,
                        rhs.as_ptr(),
                        &mut node_ptr,
                    ));
                    assert!(!node_ptr.is_null());
                    Node::from_raw(node_ptr, true)
                }
            }
        }
    }
}

macro_rules! impl_bin_node_op {
    ($name:ident,
     $op_fn:ident,
     $api_nc_fn:ident,
     $api_cn_fn:ident,
     $api_nn_fn:ident) => {
        impl_node_scalar_op!(i8, $name, $op_fn, $api_nc_fn, $api_cn_fn);
        impl_node_scalar_op!(u8, $name, $op_fn, $api_nc_fn, $api_cn_fn);
        impl_node_scalar_op!(i16, $name, $op_fn, $api_nc_fn, $api_cn_fn);
        impl_node_scalar_op!(u16, $name, $op_fn, $api_nc_fn, $api_cn_fn);
        impl_node_scalar_op!(i32, $name, $op_fn, $api_nc_fn, $api_cn_fn);
        impl_node_scalar_op!(u32, $name, $op_fn, $api_nc_fn, $api_cn_fn);
        impl_node_scalar_op!(i64, $name, $op_fn, $api_nc_fn, $api_cn_fn);
        impl_node_scalar_op!(u64, $name, $op_fn, $api_nc_fn, $api_cn_fn);
        impl_node_scalar_op!(f32, $name, $op_fn, $api_nc_fn, $api_cn_fn);
        impl_node_scalar_op!(f64, $name, $op_fn, $api_nc_fn, $api_cn_fn);

        impl ops::$name<Node> for Node {
            type Output = Node;

            fn $op_fn(self, rhs: Node) -> Node {
                unsafe {
                    let mut node_ptr: *mut _primitiv::primitivNode_t = ptr::null_mut();
                    check_api_status!(_primitiv::$api_nn_fn(
                        self.as_ptr(),
                        rhs.as_ptr(),
                        &mut node_ptr,
                    ));
                    assert!(!node_ptr.is_null());
                    Node::from_raw(node_ptr, true)
                }
            }
        }

        impl<'a> ops::$name<Node> for &'a Node {
            type Output = Node;

            fn $op_fn(self, rhs: Node) -> Node {
                unsafe {
                    let mut node_ptr: *mut _primitiv::primitivNode_t = ptr::null_mut();
                    check_api_status!(_primitiv::$api_nn_fn(
                        self.as_ptr(),
                        rhs.as_ptr(),
                        &mut node_ptr,
                    ));
                    assert!(!node_ptr.is_null());
                    Node::from_raw(node_ptr, true)
                }
            }
        }

        impl<'a> ops::$name<&'a Node> for Node {
            type Output = Node;

            fn $op_fn(self, rhs: &'a Node) -> Node {
                unsafe {
                    let mut node_ptr: *mut _primitiv::primitivNode_t = ptr::null_mut();
                    check_api_status!(_primitiv::$api_nn_fn(
                        self.as_ptr(),
                        rhs.as_ptr(),
                        &mut node_ptr,
                    ));
                    assert!(!node_ptr.is_null());
                    Node::from_raw(node_ptr, true)
                }
            }
        }

        impl<'a, 'b> ops::$name<&'a Node> for &'b Node {
            type Output = Node;

            fn $op_fn(self, rhs: &'a Node) -> Node {
                unsafe {
                    let mut node_ptr: *mut _primitiv::primitivNode_t = ptr::null_mut();
                    check_api_status!(_primitiv::$api_nn_fn(
                        self.as_ptr(),
                        rhs.as_ptr(),
                        &mut node_ptr,
                    ));
                    assert!(!node_ptr.is_null());
                    Node::from_raw(node_ptr, true)
                }
            }
        }

    }
}

impl_node_unary_func!(positive, primitivApplyNodePositive);
impl_node_unary_func!(negative, primitivApplyNodeNegative);

impl_bin_node_op!(
    Add,
    add,
    primitivApplyNodeAddXC,
    primitivApplyNodeAddCX,
    primitivApplyNodeAdd
);
impl_bin_node_op!(
    Sub,
    sub,
    primitivApplyNodeSubtractXC,
    primitivApplyNodeSubtractCX,
    primitivApplyNodeSubtract
);
impl_bin_node_op!(
    Mul,
    mul,
    primitivApplyNodeMultiplyXC,
    primitivApplyNodeMultiplyCX,
    primitivApplyNodeMultiply
);
impl_bin_node_op!(
    Div,
    div,
    primitivApplyNodeDivideXC,
    primitivApplyNodeDivideCX,
    primitivApplyNodeDivide
);

pub fn input<S: Into<Shape>>(shape: S, data: &[f32]) -> Node {
    input_into::<S, AnyDevice>(shape, data, None, None)
}

pub fn input_on<S: Into<Shape>, D: Device>(shape: S, data: &[f32], dev: Option<&mut D>) -> Node {
    input_into::<S, D>(shape, data, dev, None)
}

pub fn input_into<S: Into<Shape>, D: Device>(
    shape: S,
    data: &[f32],
    dev: Option<&mut D>,
    g: Option<&mut Graph>,
) -> Node {
    unsafe {
        let mut node_ptr: *mut _primitiv::primitivNode_t = ptr::null_mut();
        check_api_status!(_primitiv::primitivApplyNodeInput(
            shape.into().as_ptr(),
            data.as_ptr(),
            data.len(),
            dev.map(|d| d.as_mut_ptr()).unwrap_or(ptr::null_mut()),
            g.map(|_g| _g.as_mut_ptr()).unwrap_or(ptr::null_mut()),
            &mut node_ptr,
        ));
        assert!(!node_ptr.is_null());
        Node::from_raw(node_ptr, true)
    }
}

pub fn parameter(param: &mut Parameter) -> Node {
    parameter_into(param, None)
}

pub fn parameter_into(param: &mut Parameter, g: Option<&mut Graph>) -> Node {
    unsafe {
        let mut node_ptr: *mut _primitiv::primitivNode_t = ptr::null_mut();
        check_api_status!(_primitiv::primitivApplyNodeParameter(
            param.as_mut_ptr(),
            g.map(|_g| _g.as_mut_ptr()).unwrap_or(ptr::null_mut()),
            &mut node_ptr,
        ));
        assert!(!node_ptr.is_null());
        Node::from_raw(node_ptr, true)
    }
}

pub fn copy<N: AsRef<Node>>(x: N) -> Node {
    copy_on::<N, AnyDevice>(x, None)
}

pub fn copy_on<N: AsRef<Node>, D: Device>(x: N, dev: Option<&mut D>) -> Node {
    unsafe {
        let mut node_ptr: *mut _primitiv::primitivNode_t = ptr::null_mut();
        check_api_status!(_primitiv::primitivApplyNodeCopy(
            x.as_ref().as_ptr(),
            dev.map(|d| d.as_mut_ptr()).unwrap_or(ptr::null_mut()),
            &mut node_ptr,
        ));
        assert!(!node_ptr.is_null());
        Node::from_raw(node_ptr, true)
    }
}

pub fn pick<N: AsRef<Node>>(x: N, ids: &[u32], dim: u32) -> Node {
    unsafe {
        let mut node_ptr: *mut _primitiv::primitivNode_t = ptr::null_mut();
        check_api_status!(_primitiv::primitivApplyNodePick(
            x.as_ref().as_ptr(),
            ids.as_ptr(),
            ids.len(),
            dim,
            &mut node_ptr,
        ));
        assert!(!node_ptr.is_null());
        Node::from_raw(node_ptr, true)
    }
}

pub fn slice<N: AsRef<Node>>(x: N, dim: u32, lower: u32, upper: u32) -> Node {
    unsafe {
        let mut node_ptr: *mut _primitiv::primitivNode_t = ptr::null_mut();
        check_api_status!(_primitiv::primitivApplyNodeSlice(
            x.as_ref().as_ptr(),
            dim,
            lower,
            upper,
            &mut node_ptr,
        ));
        assert!(!node_ptr.is_null());
        Node::from_raw(node_ptr, true)
    }
}

pub fn concat<N: AsRef<Node>>(xs: &[N], dim: u32) -> Node {
    unsafe {
        let mut node_ptr: *mut _primitiv::primitivNode_t = ptr::null_mut();
        let x_ptrs = xs.iter().map(|x| x.as_ref().as_ptr()).collect::<Vec<_>>();
        check_api_status!(_primitiv::primitivApplyNodeConcat(
            x_ptrs.as_ptr(),
            x_ptrs.len(),
            dim,
            &mut node_ptr,
        ));
        assert!(!node_ptr.is_null());
        Node::from_raw(node_ptr, true)
    }
}

pub fn reshape<N: AsRef<Node>, S: Into<Shape>>(x: N, new_shape: S) -> Node {
    unsafe {
        let mut node_ptr: *mut _primitiv::primitivNode_t = ptr::null_mut();
        check_api_status!(_primitiv::primitivApplyNodeReshape(
            x.as_ref().as_ptr(),
            new_shape.into().as_ptr(),
            &mut node_ptr,
        ));
        assert!(!node_ptr.is_null());
        Node::from_raw(node_ptr, true)
    }
}

impl_node_unary_func!(flatten, primitivApplyNodeFlatten);
impl_node_unary_func!(transpose, primitivApplyNodeTranspose);

pub fn matmul<N: AsRef<Node>>(a: N, b: N) -> Node {
    unsafe {
        let mut node_ptr: *mut _primitiv::primitivNode_t = ptr::null_mut();
        check_api_status!(_primitiv::primitivApplyNodeMatmul(
            a.as_ref().as_ptr(),
            b.as_ref().as_ptr(),
            &mut node_ptr,
        ));
        assert!(!node_ptr.is_null());
        Node::from_raw(node_ptr, true)
    }
}

impl_node_unary_func!(sqrt, primitivApplyNodeSqrt);
impl_node_unary_func!(exp, primitivApplyNodeExp);
impl_node_unary_func!(log, primitivApplyNodeLog);
impl_node_unary_func!(tanh, primitivApplyNodeTanh);
impl_node_unary_func!(sigmoid, primitivApplyNodeSigmoid);
impl_node_unary_func!(softplus, primitivApplyNodeSoftplus);
impl_node_unary_func!(sin, primitivApplyNodeSin);
impl_node_unary_func!(cos, primitivApplyNodeCos);
impl_node_unary_func!(tan, primitivApplyNodeTan);
impl_node_unary_func!(relu, primitivApplyNodeRelu);
impl_node_unary_func!(lrelu, primitivApplyNodeLrelu);

pub fn prelu<N: AsRef<Node>>(x: N, a: f32) -> Node {
    node_func_body!(primitivApplyNodePrelu, x.as_ref().as_ptr(), a)
}

pub fn elu<N: AsRef<Node>>(x: N, a: f32) -> Node {
    node_func_body!(primitivApplyNodeElu, x.as_ref().as_ptr(), a)
}

pub fn sum<N: AsRef<Node>>(x: N, dim: u32) -> Node {
    node_func_body!(primitivApplyNodeSum, x.as_ref().as_ptr(), dim)
}

pub fn broadcast<N: AsRef<Node>>(x: N, dim: u32, size: u32) -> Node {
    node_func_body!(primitivApplyNodeBroadcast, x.as_ref().as_ptr(), dim, size)
}

pub fn logsumexp<N: AsRef<Node>>(x: N, dim: u32) -> Node {
    node_func_body!(primitivApplyNodeLogsumexp, x.as_ref().as_ptr(), dim)
}

pub fn log_softmax<N: AsRef<Node>>(x: N, dim: u32) -> Node {
    node_func_body!(primitivApplyNodeLogSoftmax, x.as_ref().as_ptr(), dim)
}

pub fn softmax<N: AsRef<Node>>(x: N, dim: u32) -> Node {
    node_func_body!(primitivApplyNodeSoftmax, x.as_ref().as_ptr(), dim)
}

pub fn softmax_cross_entropy<N: AsRef<Node>>(x: N, t: N, dim: u32) -> Node {
    node_func_body!(
        primitivApplyNodeSoftmaxCrossEntropy,
        x.as_ref().as_ptr(),
        t.as_ref().as_ptr(),
        dim
    )
}

pub fn softmax_cross_entropy_with_ids<N: AsRef<Node>>(x: N, ids: &[u32], dim: u32) -> Node {
    node_func_body!(
        primitivApplyNodeSoftmaxCrossEntropyWithArray,
        x.as_ref().as_ptr(),
        ids.as_ptr(),
        ids.len(),
        dim
    )
}

impl_node_unary_func!(stop_gradient, primitivApplyNodeStopGradient);

pub fn conv2d<N: AsRef<Node>>(
    x: N,
    w: N,
    padding0: u32,
    padding1: u32,
    stride0: u32,
    stride1: u32,
    dilation0: u32,
    dilation1: u32,
) -> Node {
    node_func_body!(
        primitivApplyNodeConv2d,
        x.as_ref().as_ptr(),
        w.as_ref().as_ptr(),
        padding0,
        padding1,
        stride0,
        stride1,
        dilation0,
        dilation1
    )
}

pub fn max_pool2d<N: AsRef<Node>>(
    x: N,
    window0: u32,
    window1: u32,
    padding0: u32,
    padding1: u32,
    stride0: u32,
    stride1: u32,
) -> Node {
    node_func_body!(
        primitivApplyNodeMaxPool2d,
        x.as_ref().as_ptr(),
        window0,
        window1,
        padding0,
        padding1,
        stride0,
        stride1
    )
}

pub fn constant<S: Into<Shape>>(shape: S, k: f32) -> Node {
    constant_into::<S, AnyDevice>(shape, k, None, None)
}

pub fn constant_on<S: Into<Shape>, D: Device>(shape: S, k: f32, dev: Option<&mut D>) -> Node {
    constant_into::<S, D>(shape, k, dev, None)
}

pub fn constant_into<S: Into<Shape>, D: Device>(
    shape: S,
    k: f32,
    dev: Option<&mut D>,
    g: Option<&mut Graph>,
) -> Node {
    node_func_body!(
        primitivApplyNodeConstant,
        shape.into().as_ptr(),
        k,
        dev.map(|d| d.as_mut_ptr()).unwrap_or(ptr::null_mut()),
        g.map(|_g| _g.as_mut_ptr()).unwrap_or(ptr::null_mut())
    )
}

pub fn identity(size: u32) -> Node {
    identity_into::<AnyDevice>(size, None, None)
}

pub fn identity_on<D: Device>(size: u32, dev: Option<&mut D>) -> Node {
    identity_into::<D>(size, dev, None)
}

pub fn identity_into<D: Device>(size: u32, dev: Option<&mut D>, g: Option<&mut Graph>) -> Node {
    node_func_body!(
        primitivApplyNodeIdentity,
        size,
        dev.map(|d| d.as_mut_ptr()).unwrap_or(ptr::null_mut()),
        g.map(|_g| _g.as_mut_ptr()).unwrap_or(ptr::null_mut())
    )
}

pub mod random {
    use primitiv_sys as _primitiv;
    use std::ptr;
    use ApiResult;
    use Device;
    use devices::AnyDevice;
    use Graph;
    use Node;
    use Shape;
    use Wrap;

    pub fn bernoulli<S: Into<Shape>>(shape: S, p: f32) -> Node {
        bernoulli_into::<S, AnyDevice>(shape, p, None, None)
    }

    pub fn bernoulli_on<S: Into<Shape>, D: Device>(shape: S, p: f32, dev: Option<&mut D>) -> Node {
        bernoulli_into::<S, D>(shape, p, dev, None)
    }

    pub fn bernoulli_into<S: Into<Shape>, D: Device>(
        shape: S,
        p: f32,
        dev: Option<&mut D>,
        g: Option<&mut Graph>,
    ) -> Node {
        node_func_body!(
            primitivApplyNodeRandomBernoulli,
            shape.into().as_ptr(),
            p,
            dev.map(|d| d.as_mut_ptr()).unwrap_or(ptr::null_mut()),
            g.map(|_g| _g.as_mut_ptr()).unwrap_or(ptr::null_mut())
        )
    }

    pub fn uniform<S: Into<Shape>>(shape: S, lower: f32, upper: f32) -> Node {
        uniform_into::<S, AnyDevice>(shape, lower, upper, None, None)
    }

    pub fn uniform_on<S: Into<Shape>, D: Device>(
        shape: S,
        lower: f32,
        upper: f32,
        dev: Option<&mut D>,
    ) -> Node {
        uniform_into::<S, D>(shape, lower, upper, dev, None)
    }

    pub fn uniform_into<S: Into<Shape>, D: Device>(
        shape: S,
        lower: f32,
        upper: f32,
        dev: Option<&mut D>,
        g: Option<&mut Graph>,
    ) -> Node {
        node_func_body!(
            primitivApplyNodeRandomUniform,
            shape.into().as_ptr(),
            lower,
            upper,
            dev.map(|d| d.as_mut_ptr()).unwrap_or(ptr::null_mut()),
            g.map(|_g| _g.as_mut_ptr()).unwrap_or(ptr::null_mut())
        )
    }

    pub fn normal<S: Into<Shape>>(shape: S, mean: f32, sd: f32) -> Node {
        normal_into::<S, AnyDevice>(shape, mean, sd, None, None)
    }

    pub fn normal_on<S: Into<Shape>, D: Device>(
        shape: S,
        mean: f32,
        sd: f32,
        dev: Option<&mut D>,
    ) -> Node {
        normal_into::<S, D>(shape, mean, sd, dev, None)
    }

    pub fn normal_into<S: Into<Shape>, D: Device>(
        shape: S,
        mean: f32,
        sd: f32,
        dev: Option<&mut D>,
        g: Option<&mut Graph>,
    ) -> Node {
        node_func_body!(
            primitivApplyNodeRandomNormal,
            shape.into().as_ptr(),
            mean,
            sd,
            dev.map(|d| d.as_mut_ptr()).unwrap_or(ptr::null_mut()),
            g.map(|_g| _g.as_mut_ptr()).unwrap_or(ptr::null_mut())
        )
    }

    pub fn log_normal<S: Into<Shape>>(shape: S, mean: f32, sd: f32) -> Node {
        log_normal_into::<S, AnyDevice>(shape, mean, sd, None, None)
    }

    pub fn log_normal_on<S: Into<Shape>, D: Device>(
        shape: S,
        mean: f32,
        sd: f32,
        dev: Option<&mut D>,
    ) -> Node {
        log_normal_into::<S, D>(shape, mean, sd, dev, None)
    }

    pub fn log_normal_into<S: Into<Shape>, D: Device>(
        shape: S,
        mean: f32,
        sd: f32,
        dev: Option<&mut D>,
        g: Option<&mut Graph>,
    ) -> Node {
        node_func_body!(
            primitivApplyNodeRandomLogNormal,
            shape.into().as_ptr(),
            mean,
            sd,
            dev.map(|d| d.as_mut_ptr()).unwrap_or(ptr::null_mut()),
            g.map(|_g| _g.as_mut_ptr()).unwrap_or(ptr::null_mut())
        )
    }

    pub fn gumbel<S: Into<Shape>>(shape: S, mu: f32, beta: f32) -> Node {
        gumbel_into::<S, AnyDevice>(shape, mu, beta, None, None)
    }

    pub fn gumbel_on<S: Into<Shape>, D: Device>(
        shape: S,
        mu: f32,
        beta: f32,
        dev: Option<&mut D>,
    ) -> Node {
        gumbel_into::<S, D>(shape, mu, beta, dev, None)
    }

    pub fn gumbel_into<S: Into<Shape>, D: Device>(
        shape: S,
        mu: f32,
        beta: f32,
        dev: Option<&mut D>,
        g: Option<&mut Graph>,
    ) -> Node {
        node_func_body!(
            primitivApplyNodeRandomNormal,
            shape.into().as_ptr(),
            mu,
            beta,
            dev.map(|d| d.as_mut_ptr()).unwrap_or(ptr::null_mut()),
            g.map(|_g| _g.as_mut_ptr()).unwrap_or(ptr::null_mut())
        )
    }
}

/*
pub mod batch {
    use primitiv_sys as _primitiv;
    // use std::ptr;
    use ApiResult;
    // use Device;
    // use devices::AnyDevice;
    // use Graph;
    use Node;
    // use Parameter;
    // use Shape;
    use Wrap;
    // use Node;
    // use Status;
    // use Wrap;

    pub fn mean<N: AsRef<Node>>(x: N) -> Node {
        let mut status = Status::new();
        unsafe {
            let node = Node::from_inner_ptr(_primitiv::safe_primitiv_node_func_batch_mean(
                x.as_ref().as_inner_ptr(),
                status.as_inner_mut_ptr(),
            ));
            status.into_result().unwrap();
            node
        }
    }
}

pub fn softmax_cross_entropy<N: AsRef<Node>>(x: N, ids: &[u32], dim: u32) -> Node {
    let mut status = Status::new();
    unsafe {
        let node = Node::from_inner_ptr(
            _primitiv::safe_primitiv_node_func_softmax_cross_entropy_with_array(
                x.as_ref().as_inner_ptr(),
                ids.as_ptr() as *const _,
                ids.len(),
                dim,
                status.as_inner_mut_ptr(),
            ),
        );
        status.into_result().unwrap();
        node
    }
}

pub fn dropout<N: AsRef<Node>>(x: N, rate: f32, enabled: bool) -> Node {
    let mut status = Status::new();
    unsafe {
        let node = Node::from_inner_ptr(_primitiv::safe_primitiv_node_func_dropout(
            x.as_ref().as_inner_ptr(),
            rate,
            enabled as u8,
            status.as_inner_mut_ptr(),
        ));
        status.into_result().unwrap();
        node
    }
}

*/
