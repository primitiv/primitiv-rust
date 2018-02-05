use primitiv_sys as _primitiv;
use std::ops;
use std::ptr;
use ApiResult;
use Device;
use devices::AnyDevice;
use Parameter;
use Shape;
use Tensor;
use Wrap;

macro_rules! tensor_func_body {
    ($api_fn:ident, $($arg:expr),* ) => {
        unsafe {
            let mut tensor_ptr: *mut _primitiv::primitivTensor_t = ptr::null_mut();
            check_api_status!(_primitiv::$api_fn(
                $($arg),*,
                &mut tensor_ptr,
            ));
            assert!(!tensor_ptr.is_null());
            Tensor::from_raw(tensor_ptr, true)
        }
    }
}

macro_rules! impl_tensor_unary_func {
    ($name:ident, $api_fn:ident) => {
        pub fn $name<T: AsRef<Tensor>>(x: T) -> Tensor {
            tensor_func_body!($api_fn, x.as_ref().as_ptr())
        }
    }
}

macro_rules! impl_tensor_binary_func {
    ($name:ident,
     $api_fn:ident,
     $name_xc:ident,
     $api_fn_xc:ident,
     $name_cx:ident,
     $api_fn_cx:ident) => {
        pub fn $name<T: AsRef<Tensor>>(a: T, b: T) -> Tensor {
            tensor_func_body!($api_fn, a.as_ref().as_ptr(), b.as_ref().as_ptr())
        }

        pub fn $name_xc<T: AsRef<Tensor>>(x: T, k: f32) -> Tensor {
            tensor_func_body!($api_fn_xc, x.as_ref().as_ptr(), k)
        }

        pub fn $name_cx<T: AsRef<Tensor>>(k: f32, x: T) -> Tensor {
            tensor_func_body!($api_fn_cx, k, x.as_ref().as_ptr())
        }
    }
}

macro_rules! impl_tensor_unary_op {
    ($name:ident,
     $op_fn:ident,
     $api_fn:ident) => {
        impl ops::$name for Tensor {
            type Output = Tensor;

            fn $op_fn(self) -> Tensor {
                tensor_func_body!($api_fn, self.as_ptr())
            }
        }
    }
}

macro_rules! impl_tensor_binary_with_constant_op {
    ($scalar:ty,
     $name:ident,
     $op_fn:ident,
     $api_fn_xc:ident,
     $api_fn_cx:ident) => {
        impl ops::$name<$scalar> for Tensor {
            type Output = Tensor;

            fn $op_fn(self, rhs: $scalar) -> Tensor {
                tensor_func_body!($api_fn_xc, self.as_ptr(), rhs as f32)
            }
        }

        impl ops::$name<Tensor> for $scalar {
            type Output = Tensor;

            fn $op_fn(self, rhs: Tensor) -> Tensor {
                tensor_func_body!($api_fn_cx, self as f32, rhs.as_ptr())
            }
        }
    }
}

macro_rules! impl_tensor_binary_op {
    ($name:ident,
     $op_fn:ident,
     $api_fn:ident,
     $api_fn_xc:ident,
     $api_fn_cx:ident) => {
        impl_tensor_binary_with_constant_op!(i8, $name, $op_fn, $api_fn_xc, $api_fn_cx);
        impl_tensor_binary_with_constant_op!(u8, $name, $op_fn, $api_fn_xc, $api_fn_cx);
        impl_tensor_binary_with_constant_op!(i16, $name, $op_fn, $api_fn_xc, $api_fn_cx);
        impl_tensor_binary_with_constant_op!(u16, $name, $op_fn, $api_fn_xc, $api_fn_cx);
        impl_tensor_binary_with_constant_op!(i32, $name, $op_fn, $api_fn_xc, $api_fn_cx);
        impl_tensor_binary_with_constant_op!(u32, $name, $op_fn, $api_fn_xc, $api_fn_cx);
        impl_tensor_binary_with_constant_op!(i64, $name, $op_fn, $api_fn_xc, $api_fn_cx);
        impl_tensor_binary_with_constant_op!(u64, $name, $op_fn, $api_fn_xc, $api_fn_cx);
        impl_tensor_binary_with_constant_op!(f32, $name, $op_fn, $api_fn_xc, $api_fn_cx);
        impl_tensor_binary_with_constant_op!(f64, $name, $op_fn, $api_fn_xc, $api_fn_cx);

        impl ops::$name for Tensor {
            type Output = Tensor;

            fn $op_fn(self, rhs: Tensor) -> Tensor {
                tensor_func_body!($api_fn, self.as_ptr(), rhs.as_ptr())
            }
        }

        impl<'a> ops::$name<Tensor> for &'a Tensor {
            type Output = Tensor;

            fn $op_fn(self, rhs: Tensor) -> Tensor {
                tensor_func_body!($api_fn, self.as_ptr(), rhs.as_ptr())
            }
        }

        impl<'a> ops::$name<&'a Tensor> for Tensor {
            type Output = Tensor;

            fn $op_fn(self, rhs: &'a Tensor) -> Tensor {
                tensor_func_body!($api_fn, self.as_ptr(), rhs.as_ptr())
            }
        }

        impl<'a, 'b> ops::$name<&'a Tensor> for &'b Tensor {
            type Output = Tensor;

            fn $op_fn(self, rhs: &'a Tensor) -> Tensor {
                tensor_func_body!($api_fn, self.as_ptr(), rhs.as_ptr())
            }
        }
    }
}

impl_tensor_unary_func!(positive, primitivApplyTensorPositive);
impl_tensor_unary_func!(negative, primitivApplyTensorNegative);
impl_tensor_unary_op!(Neg, neg, primitivApplyTensorNegative);
impl_tensor_binary_func!(
    add,
    primitivApplyTensorAdd,
    add_const,
    primitivApplyTensorAddXC,
    add_tensor,
    primitivApplyTensorAddCX
);
impl_tensor_binary_op!(
    Add,
    add,
    primitivApplyTensorAdd,
    primitivApplyTensorAddXC,
    primitivApplyTensorAddCX
);
impl_tensor_binary_func!(
    subtract,
    primitivApplyTensorSubtract,
    subtract_const,
    primitivApplyTensorSubtractXC,
    subtract_tensor,
    primitivApplyTensorSubtractCX
);
impl_tensor_binary_op!(
    Sub,
    sub,
    primitivApplyTensorSubtract,
    primitivApplyTensorSubtractXC,
    primitivApplyTensorSubtractCX
);
impl_tensor_binary_func!(
    multiply,
    primitivApplyTensorMultiply,
    multiply_const,
    primitivApplyTensorMultiplyXC,
    multiply_tensor,
    primitivApplyTensorMultiplyCX
);
impl_tensor_binary_op!(
    Mul,
    mul,
    primitivApplyTensorMultiply,
    primitivApplyTensorMultiplyXC,
    primitivApplyTensorMultiplyCX
);
impl_tensor_binary_func!(
    divide,
    primitivApplyTensorDivide,
    divide_const,
    primitivApplyTensorDivideXC,
    divide_tensor,
    primitivApplyTensorDivideCX
);
impl_tensor_binary_op!(
    Div,
    div,
    primitivApplyTensorDivide,
    primitivApplyTensorDivideXC,
    primitivApplyTensorDivideCX
);
impl_tensor_binary_func!(
    pow,
    primitivApplyTensorPow,
    pow_const,
    primitivApplyTensorPowXC,
    pow_tensor,
    primitivApplyTensorPowCX
);

pub fn pown<T: AsRef<Tensor>>(x: T, k: i32) -> Tensor {
    tensor_func_body!(primitivApplyTensorPowN, x.as_ref().as_ptr(), k)
}

pub fn input<S: Into<Shape>>(shape: S, data: &[f32]) -> Tensor {
    input_on::<S, AnyDevice>(shape, data, None)
}

pub fn input_on<S: Into<Shape>, D: Device>(shape: S, data: &[f32], dev: Option<&mut D>) -> Tensor {
    tensor_func_body!(
        primitivApplyTensorInput,
        shape.into().as_ptr(),
        data.as_ptr(),
        data.len(),
        dev.map(|d| d.as_mut_ptr()).unwrap_or(ptr::null_mut())
    )
}

pub fn parameter(param: &mut Parameter) -> Tensor {
    tensor_func_body!(primitivApplyTensorParameter, param.as_mut_ptr())
}

pub fn copy<T: AsRef<Tensor>>(x: T) -> Tensor {
    copy_on::<T, AnyDevice>(x, None)
}

pub fn copy_on<T: AsRef<Tensor>, D: Device>(x: T, dev: Option<&mut D>) -> Tensor {
    tensor_func_body!(
        primitivApplyTensorCopy,
        x.as_ref().as_ptr(),
        dev.map(|d| d.as_mut_ptr()).unwrap_or(ptr::null_mut())
    )
}

pub fn pick<T: AsRef<Tensor>>(x: T, ids: &[u32], dim: u32) -> Tensor {
    tensor_func_body!(
        primitivApplyTensorPick,
        x.as_ref().as_ptr(),
        ids.as_ptr(),
        ids.len(),
        dim
    )
}

pub fn slice<T: AsRef<Tensor>>(x: T, dim: u32, lower: u32, upper: u32) -> Tensor {
    tensor_func_body!(
        primitivApplyTensorSlice,
        x.as_ref().as_ptr(),
        dim,
        lower,
        upper
    )
}

pub fn concat<T: AsRef<Tensor>>(xs: &[T], dim: u32) -> Tensor {
    let x_ptrs = xs.iter().map(|x| x.as_ref().as_ptr()).collect::<Vec<_>>();
    tensor_func_body!(
        primitivApplyTensorConcat,
        x_ptrs.as_ptr(),
        x_ptrs.len(),
        dim
    )
}

pub fn reshape<T: AsRef<Tensor>, S: Into<Shape>>(x: T, new_shape: S) -> Tensor {
    tensor_func_body!(
        primitivApplyTensorReshape,
        x.as_ref().as_ptr(),
        new_shape.into().as_ptr()
    )
}

impl_tensor_unary_func!(flatten, primitivApplyTensorFlatten);
impl_tensor_unary_func!(transpose, primitivApplyTensorTranspose);

pub fn matmul<T: AsRef<Tensor>>(a: T, b: T) -> Tensor {
    tensor_func_body!(
        primitivApplyTensorMatmul,
        a.as_ref().as_ptr(),
        b.as_ref().as_ptr()
    )
}

impl_tensor_unary_func!(sqrt, primitivApplyTensorSqrt);
impl_tensor_unary_func!(exp, primitivApplyTensorExp);
impl_tensor_unary_func!(log, primitivApplyTensorLog);
impl_tensor_unary_func!(tanh, primitivApplyTensorTanh);
impl_tensor_unary_func!(sigmoid, primitivApplyTensorSigmoid);
impl_tensor_unary_func!(softplus, primitivApplyTensorSoftplus);
impl_tensor_unary_func!(sin, primitivApplyTensorSin);
impl_tensor_unary_func!(cos, primitivApplyTensorCos);
impl_tensor_unary_func!(tan, primitivApplyTensorTan);
impl_tensor_unary_func!(relu, primitivApplyTensorRelu);
impl_tensor_unary_func!(lrelu, primitivApplyTensorLrelu);

pub fn prelu<T: AsRef<Tensor>>(x: T, a: f32) -> Tensor {
    tensor_func_body!(primitivApplyTensorPrelu, x.as_ref().as_ptr(), a)
}

pub fn elu<T: AsRef<Tensor>>(x: T, a: f32) -> Tensor {
    tensor_func_body!(primitivApplyTensorElu, x.as_ref().as_ptr(), a)
}

impl_tensor_unary_func!(selu, primitivApplyTensorSelu);

pub fn sum<T: AsRef<Tensor>>(x: T, dim: u32) -> Tensor {
    tensor_func_body!(primitivApplyTensorSum, x.as_ref().as_ptr(), dim)
}

pub fn sum_tensors<T: AsRef<Tensor>>(xs: &[T]) -> Tensor {
    let x_ptrs = xs.iter().map(|x| x.as_ref().as_ptr()).collect::<Vec<_>>();
    tensor_func_body!(primitivApplyTensorSumTensors, x_ptrs.as_ptr(), x_ptrs.len())
}

pub fn mean<T: AsRef<Tensor>>(x: T, dim: u32) -> Tensor {
    tensor_func_body!(primitivApplyTensorMean, x.as_ref().as_ptr(), dim)
}

pub fn mean_tensors<T: AsRef<Tensor>>(xs: &[T]) -> Tensor {
    let x_ptrs = xs.iter().map(|x| x.as_ref().as_ptr()).collect::<Vec<_>>();
    tensor_func_body!(
        primitivApplyTensorMeanTensors,
        x_ptrs.as_ptr(),
        x_ptrs.len()
    )
}

pub fn broadcast<T: AsRef<Tensor>>(x: T, dim: u32, size: u32) -> Tensor {
    tensor_func_body!(primitivApplyTensorBroadcast, x.as_ref().as_ptr(), dim, size)
}

pub fn logsumexp<T: AsRef<Tensor>>(x: T, dim: u32) -> Tensor {
    tensor_func_body!(primitivApplyTensorLogsumexp, x.as_ref().as_ptr(), dim)
}

pub fn log_softmax<T: AsRef<Tensor>>(x: T, dim: u32) -> Tensor {
    tensor_func_body!(primitivApplyTensorLogSoftmax, x.as_ref().as_ptr(), dim)
}

pub fn softmax<T: AsRef<Tensor>>(x: T, dim: u32) -> Tensor {
    tensor_func_body!(primitivApplyTensorSoftmax, x.as_ref().as_ptr(), dim)
}

pub fn softmax_cross_entropy<T: AsRef<Tensor>>(x: T, t: T, dim: u32) -> Tensor {
    tensor_func_body!(
        primitivApplyTensorSoftmaxCrossEntropy,
        x.as_ref().as_ptr(),
        t.as_ref().as_ptr(),
        dim
    )
}

pub fn softmax_cross_entropy_with_ids<T: AsRef<Tensor>>(x: T, ids: &[u32], dim: u32) -> Tensor {
    tensor_func_body!(
        primitivApplyTensorSoftmaxCrossEntropyWithArray,
        x.as_ref().as_ptr(),
        ids.as_ptr(),
        ids.len(),
        dim
    )
}

impl_tensor_unary_func!(stop_gradient, primitivApplyTensorStopGradient);

pub fn conv2d<T: AsRef<Tensor>>(
    x: T,
    w: T,
    padding0: u32,
    padding1: u32,
    stride0: u32,
    stride1: u32,
    dilation0: u32,
    dilation1: u32,
) -> Tensor {
    tensor_func_body!(
        primitivApplyTensorConv2d,
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

pub fn max_pool2d<T: AsRef<Tensor>>(
    x: T,
    window0: u32,
    window1: u32,
    padding0: u32,
    padding1: u32,
    stride0: u32,
    stride1: u32,
) -> Tensor {
    tensor_func_body!(
        primitivApplyTensorMaxPool2d,
        x.as_ref().as_ptr(),
        window0,
        window1,
        padding0,
        padding1,
        stride0,
        stride1
    )
}

pub fn constant<S: Into<Shape>>(shape: S, k: f32) -> Tensor {
    constant_on::<S, AnyDevice>(shape, k, None)
}

pub fn constant_on<S: Into<Shape>, D: Device>(shape: S, k: f32, dev: Option<&mut D>) -> Tensor {
    tensor_func_body!(
        primitivApplyTensorConstant,
        shape.into().as_ptr(),
        k,
        dev.map(|d| d.as_mut_ptr()).unwrap_or(ptr::null_mut())
    )
}

pub fn identity(size: u32) -> Tensor {
    identity_on::<AnyDevice>(size, None)
}

pub fn identity_on<D: Device>(size: u32, dev: Option<&mut D>) -> Tensor {
    tensor_func_body!(
        primitivApplyTensorIdentity,
        size,
        dev.map(|d| d.as_mut_ptr()).unwrap_or(ptr::null_mut())
    )
}

pub fn zeros<S: Into<Shape>>(shape: S) -> Tensor {
    zeros_on::<S, AnyDevice>(shape, None)
}

pub fn zeros_on<S: Into<Shape>, D: Device>(shape: S, dev: Option<&mut D>) -> Tensor {
    tensor_func_body!(
        primitivApplyTensorZeros,
        shape.into().as_ptr(),
        dev.map(|d| d.as_mut_ptr()).unwrap_or(ptr::null_mut())
    )
}

pub fn ones<S: Into<Shape>>(shape: S) -> Tensor {
    ones_on::<S, AnyDevice>(shape, None)
}

pub fn ones_on<S: Into<Shape>, D: Device>(shape: S, dev: Option<&mut D>) -> Tensor {
    tensor_func_body!(
        primitivApplyTensorOnes,
        shape.into().as_ptr(),
        dev.map(|d| d.as_mut_ptr()).unwrap_or(ptr::null_mut())
    )
}

pub fn dropout<T: AsRef<Tensor>>(x: T, rate: f32, enabled: bool) -> Tensor {
    tensor_func_body!(
        primitivApplyTensorDropout,
        x.as_ref().as_ptr(),
        rate,
        enabled as u32
    )
}

pub mod random {
    use primitiv_sys as _primitiv;
    use std::ptr;
    use ApiResult;
    use Device;
    use devices::AnyDevice;
    use Shape;
    use Tensor;
    use Wrap;

    pub fn bernoulli<S: Into<Shape>>(shape: S, p: f32) -> Tensor {
        bernoulli_on::<S, AnyDevice>(shape, p, None)
    }

    pub fn bernoulli_on<S: Into<Shape>, D: Device>(
        shape: S,
        p: f32,
        dev: Option<&mut D>,
    ) -> Tensor {
        tensor_func_body!(
            primitivApplyTensorRandomBernoulli,
            shape.into().as_ptr(),
            p,
            dev.map(|d| d.as_mut_ptr()).unwrap_or(ptr::null_mut())
        )
    }

    pub fn uniform<S: Into<Shape>>(shape: S, lower: f32, upper: f32) -> Tensor {
        uniform_on::<S, AnyDevice>(shape, lower, upper, None)
    }

    pub fn uniform_on<S: Into<Shape>, D: Device>(
        shape: S,
        lower: f32,
        upper: f32,
        dev: Option<&mut D>,
    ) -> Tensor {
        tensor_func_body!(
            primitivApplyTensorRandomUniform,
            shape.into().as_ptr(),
            lower,
            upper,
            dev.map(|d| d.as_mut_ptr()).unwrap_or(ptr::null_mut())
        )
    }

    pub fn normal<S: Into<Shape>>(shape: S, mean: f32, sd: f32) -> Tensor {
        normal_on::<S, AnyDevice>(shape, mean, sd, None)
    }

    pub fn normal_on<S: Into<Shape>, D: Device>(
        shape: S,
        mean: f32,
        sd: f32,
        dev: Option<&mut D>,
    ) -> Tensor {
        tensor_func_body!(
            primitivApplyTensorRandomNormal,
            shape.into().as_ptr(),
            mean,
            sd,
            dev.map(|d| d.as_mut_ptr()).unwrap_or(ptr::null_mut())
        )
    }

    pub fn log_normal<S: Into<Shape>>(shape: S, mean: f32, sd: f32) -> Tensor {
        log_normal_on::<S, AnyDevice>(shape, mean, sd, None)
    }

    pub fn log_normal_on<S: Into<Shape>, D: Device>(
        shape: S,
        mean: f32,
        sd: f32,
        dev: Option<&mut D>,
    ) -> Tensor {
        tensor_func_body!(
            primitivApplyTensorRandomLogNormal,
            shape.into().as_ptr(),
            mean,
            sd,
            dev.map(|d| d.as_mut_ptr()).unwrap_or(ptr::null_mut())
        )
    }

    pub fn gumbel<S: Into<Shape>>(shape: S, mu: f32, beta: f32) -> Tensor {
        gumbel_on::<S, AnyDevice>(shape, mu, beta, None)
    }

    pub fn gumbel_on<S: Into<Shape>, D: Device>(
        shape: S,
        mu: f32,
        beta: f32,
        dev: Option<&mut D>,
    ) -> Tensor {
        tensor_func_body!(
            primitivApplyTensorRandomNormal,
            shape.into().as_ptr(),
            mu,
            beta,
            dev.map(|d| d.as_mut_ptr()).unwrap_or(ptr::null_mut())
        )
    }
}

pub mod batch {
    use primitiv_sys as _primitiv;
    use std::ptr;
    use ApiResult;
    use Tensor;
    use Wrap;

    impl_tensor_unary_func!(sum, primitivApplyTensorBatchSum);
    impl_tensor_unary_func!(mean, primitivApplyTensorBatchMean);
    impl_tensor_unary_func!(normalize, primitivApplyTensorBatchNormalize);
}
