use std::marker::PhantomData;
use Device;
use Node;
use Parameter;
use Shape;
use Tensor;

use super::node_funcs;
use super::tensor_funcs;

pub trait Variable: AsRef<Self> + Default + Sized {
    type F: Functions<Self>;

    fn new() -> Self {
        Self::default()
    }
}

impl Variable for Node {
    type F = FuncImpls<Node>;
}

impl Variable for Tensor {
    type F = FuncImpls<Tensor>;
}

pub fn positive<T: AsRef<V>, V: Variable>(x: T) -> V {
    <V as Variable>::F::negative(x)
}

pub fn negative<T: AsRef<V>, V: Variable>(x: T) -> V {
    <V as Variable>::F::negative(x)
}

pub fn add<T1: AsRef<V>, T2: AsRef<V>, V: Variable>(a: T1, b: T2) -> V {
    <V as Variable>::F::add(a, b)
}

pub fn add_const<T: AsRef<V>, V: Variable>(x: T, k: f32) -> V {
    <V as Variable>::F::add_const(x, k)
}

pub fn add_var<T: AsRef<V>, V: Variable>(k: f32, x: T) -> V {
    <V as Variable>::F::add_var(k, x)
}

pub fn subtract<T1: AsRef<V>, T2: AsRef<V>, V: Variable>(a: T1, b: T2) -> V {
    <V as Variable>::F::subtract(a, b)
}

pub fn subtract_const<T: AsRef<V>, V: Variable>(x: T, k: f32) -> V {
    <V as Variable>::F::subtract_const(x, k)
}

pub fn subtract_var<T: AsRef<V>, V: Variable>(k: f32, x: T) -> V {
    <V as Variable>::F::subtract_var(k, x)
}

pub fn multiply<T1: AsRef<V>, T2: AsRef<V>, V: Variable>(a: T1, b: T2) -> V {
    <V as Variable>::F::multiply(a, b)
}

pub fn multiply_const<T: AsRef<V>, V: Variable>(x: T, k: f32) -> V {
    <V as Variable>::F::multiply_const(x, k)
}

pub fn multiply_var<T: AsRef<V>, V: Variable>(k: f32, x: T) -> V {
    <V as Variable>::F::multiply_var(k, x)
}

pub fn divide<T1: AsRef<V>, T2: AsRef<V>, V: Variable>(a: T1, b: T2) -> V {
    <V as Variable>::F::divide(a, b)
}

pub fn divide_const<T: AsRef<V>, V: Variable>(x: T, k: f32) -> V {
    <V as Variable>::F::divide_const(x, k)
}

pub fn divide_var<T: AsRef<V>, V: Variable>(k: f32, x: T) -> V {
    <V as Variable>::F::divide_var(k, x)
}

pub fn pow<T1: AsRef<V>, T2: AsRef<V>, V: Variable>(a: T1, b: T2) -> V {
    <V as Variable>::F::pow(a, b)
}

pub fn pow_const<T: AsRef<V>, V: Variable>(x: T, k: f32) -> V {
    <V as Variable>::F::pow_const(x, k)
}

pub fn pow_var<T: AsRef<V>, V: Variable>(k: f32, x: T) -> V {
    <V as Variable>::F::pow_var(k, x)
}

pub fn pown<T: AsRef<V>, V: Variable>(x: T, k: i32) -> V {
    <V as Variable>::F::pown(x, k)
}

pub fn input<S: Into<Shape>, V: Variable>(shape: S, data: &[f32]) -> V {
    <V as Variable>::F::input(shape, data)
}

pub fn input_on<S: Into<Shape>, D: Device, V: Variable>(
    shape: S,
    data: &[f32],
    dev: Option<&mut D>,
) -> V {
    <V as Variable>::F::input_on(shape, data, dev)
}

pub fn parameter<V: Variable>(param: &mut Parameter) -> V {
    <V as Variable>::F::parameter(param)
}

pub fn copy<T: AsRef<V>, V: Variable>(x: T) -> V {
    <V as Variable>::F::copy(x)
}

pub fn copy_on<T: AsRef<V>, D: Device, V: Variable>(x: T, dev: Option<&mut D>) -> V {
    <V as Variable>::F::copy_on(x, dev)
}

pub fn pick<T: AsRef<V>, V: Variable>(x: T, ids: &[u32], dim: u32) -> V {
    <V as Variable>::F::pick(x, ids, dim)
}

pub fn slice<T: AsRef<V>, V: Variable>(x: T, dim: u32, lower: u32, upper: u32) -> V {
    <V as Variable>::F::slice(x, dim, lower, upper)
}

pub fn split<T: AsRef<V>, V: Variable>(x: T, dim: u32, n: u32) -> Vec<V> {
    <V as Variable>::F::split(x, dim, n)
}

pub fn concat<TS: AsRef<[T]>, T: AsRef<V>, V: Variable>(xs: TS, dim: u32) -> V {
    <V as Variable>::F::concat(xs, dim)
}

pub fn reshape<T: AsRef<V>, S: Into<Shape>, V: Variable>(x: T, new_shape: S) -> V {
    <V as Variable>::F::reshape(x, new_shape)
}

pub fn flatten<T: AsRef<V>, V: Variable>(x: T) -> V {
    <V as Variable>::F::flatten(x)
}

pub fn transpose<T: AsRef<V>, V: Variable>(x: T) -> V {
    <V as Variable>::F::transpose(x)
}

pub fn matmul<T1: AsRef<V>, T2: AsRef<V>, V: Variable>(a: T1, b: T2) -> V {
    <V as Variable>::F::matmul(a, b)
}

pub fn abs<T: AsRef<V>, V: Variable>(x: T) -> V {
    <V as Variable>::F::abs(x)
}

pub fn sqrt<T: AsRef<V>, V: Variable>(x: T) -> V {
    <V as Variable>::F::sqrt(x)
}

pub fn exp<T: AsRef<V>, V: Variable>(x: T) -> V {
    <V as Variable>::F::exp(x)
}

pub fn log<T: AsRef<V>, V: Variable>(x: T) -> V {
    <V as Variable>::F::log(x)
}

pub fn tanh<T: AsRef<V>, V: Variable>(x: T) -> V {
    <V as Variable>::F::tanh(x)
}

pub fn sigmoid<T: AsRef<V>, V: Variable>(x: T) -> V {
    <V as Variable>::F::sigmoid(x)
}

pub fn softplus<T: AsRef<V>, V: Variable>(x: T) -> V {
    <V as Variable>::F::softplus(x)
}

pub fn sin<T: AsRef<V>, V: Variable>(x: T) -> V {
    <V as Variable>::F::sin(x)
}

pub fn cos<T: AsRef<V>, V: Variable>(x: T) -> V {
    <V as Variable>::F::cos(x)
}

pub fn tan<T: AsRef<V>, V: Variable>(x: T) -> V {
    <V as Variable>::F::tan(x)
}

pub fn relu<T: AsRef<V>, V: Variable>(x: T) -> V {
    <V as Variable>::F::relu(x)
}

pub fn lrelu<T: AsRef<V>, V: Variable>(x: T) -> V {
    <V as Variable>::F::lrelu(x)
}

pub fn prelu<T: AsRef<V>, V: Variable>(x: T, a: f32) -> V {
    <V as Variable>::F::prelu(x, a)
}

pub fn elu<T: AsRef<V>, V: Variable>(x: T, a: f32) -> V {
    <V as Variable>::F::elu(x, a)
}

pub fn selu<T: AsRef<V>, V: Variable>(x: T) -> V {
    <V as Variable>::F::selu(x)
}

pub fn max<T: AsRef<V>, V: Variable>(x: T, dim: u32) -> V {
    <V as Variable>::F::max(x, dim)
}

pub fn min<T: AsRef<V>, V: Variable>(x: T, dim: u32) -> V {
    <V as Variable>::F::min(x, dim)
}

pub fn sum<T: AsRef<V>, V: Variable>(x: T, dim: u32) -> V {
    <V as Variable>::F::sum(x, dim)
}

pub fn sum_vars<TS: AsRef<[T]>, T: AsRef<V>, V: Variable>(xs: TS) -> V {
    <V as Variable>::F::sum_vars(xs)
}

pub fn mean<T: AsRef<V>, V: Variable>(x: T, dim: u32) -> V {
    <V as Variable>::F::mean(x, dim)
}

pub fn mean_vars<TS: AsRef<[T]>, T: AsRef<V>, V: Variable>(xs: TS) -> V {
    <V as Variable>::F::mean_vars(xs)
}

pub fn broadcast<T: AsRef<V>, V: Variable>(x: T, dim: u32, size: u32) -> V {
    <V as Variable>::F::broadcast(x, dim, size)
}

pub fn logsumexp<T: AsRef<V>, V: Variable>(x: T, dim: u32) -> V {
    <V as Variable>::F::logsumexp(x, dim)
}

pub fn log_softmax<T: AsRef<V>, V: Variable>(x: T, dim: u32) -> V {
    <V as Variable>::F::log_softmax(x, dim)
}

pub fn softmax<T: AsRef<V>, V: Variable>(x: T, dim: u32) -> V {
    <V as Variable>::F::softmax(x, dim)
}

pub fn softmax_cross_entropy<T1: AsRef<V>, T2: AsRef<V>, V: Variable>(x: T1, t: T2, dim: u32) -> V {
    <V as Variable>::F::softmax_cross_entropy(x, t, dim)
}

pub fn softmax_cross_entropy_with_ids<T: AsRef<V>, V: Variable>(x: T, ids: &[u32], dim: u32) -> V {
    <V as Variable>::F::softmax_cross_entropy_with_ids(x, ids, dim)
}

pub fn stop_gradient<T: AsRef<V>, V: Variable>(x: T) -> V {
    <V as Variable>::F::stop_gradient(x)
}

pub fn conv2d<T1: AsRef<V>, T2: AsRef<V>, V: Variable>(
    x: T1,
    w: T2,
    padding0: u32,
    padding1: u32,
    stride0: u32,
    stride1: u32,
    dilation0: u32,
    dilation1: u32,
) -> V {
    <V as Variable>::F::conv2d(
        x,
        w,
        padding0,
        padding1,
        stride0,
        stride1,
        dilation0,
        dilation1,
    )
}

pub fn max_pool2d<T: AsRef<V>, V: Variable>(
    x: T,
    window0: u32,
    window1: u32,
    padding0: u32,
    padding1: u32,
    stride0: u32,
    stride1: u32,
) -> V {
    <V as Variable>::F::max_pool2d(x, window0, window1, padding0, padding1, stride0, stride1)
}

pub fn constant<S: Into<Shape>, V: Variable>(shape: S, k: f32) -> V {
    <V as Variable>::F::constant(shape, k)
}

pub fn constant_on<S: Into<Shape>, D: Device, V: Variable>(
    shape: S,
    k: f32,
    dev: Option<&mut D>,
) -> V {
    <V as Variable>::F::constant_on(shape, k, dev)
}

pub fn identity<V: Variable>(size: u32) -> V {
    <V as Variable>::F::identity(size)
}

pub fn identity_on<D: Device, V: Variable>(size: u32, dev: Option<&mut D>) -> V {
    <V as Variable>::F::identity_on(size, dev)
}

pub fn zeros<S: Into<Shape>, V: Variable>(shape: S) -> V {
    <V as Variable>::F::zeros(shape)
}

pub fn zeros_on<S: Into<Shape>, D: Device, V: Variable>(shape: S, dev: Option<&mut D>) -> V {
    <V as Variable>::F::zeros_on(shape, dev)
}

pub fn ones<S: Into<Shape>, V: Variable>(shape: S) -> V {
    <V as Variable>::F::ones(shape)
}

pub fn ones_on<S: Into<Shape>, D: Device, V: Variable>(shape: S, dev: Option<&mut D>) -> V {
    <V as Variable>::F::ones_on(shape, dev)
}

pub fn dropout<T: AsRef<V>, V: Variable>(x: T, rate: f32, enabled: bool) -> V {
    <V as Variable>::F::dropout(x, rate, enabled)
}

pub mod random {
    use super::Variable;
    use super::Functions;
    use Device;
    use Shape;

    pub fn bernoulli<S: Into<Shape>, V: Variable>(shape: S, p: f32) -> V {
        <V as Variable>::F::random_bernoulli(shape, p)
    }

    pub fn bernoulli_on<S: Into<Shape>, D: Device, V: Variable>(
        shape: S,
        p: f32,
        dev: Option<&mut D>,
    ) -> V {
        <V as Variable>::F::random_bernoulli_on(shape, p, dev)
    }

    pub fn uniform<S: Into<Shape>, V: Variable>(shape: S, lower: f32, upper: f32) -> V {
        <V as Variable>::F::random_uniform(shape, lower, upper)
    }

    pub fn uniform_on<S: Into<Shape>, D: Device, V: Variable>(
        shape: S,
        lower: f32,
        upper: f32,
        dev: Option<&mut D>,
    ) -> V {
        <V as Variable>::F::random_uniform_on(shape, lower, upper, dev)
    }

    pub fn normal<S: Into<Shape>, V: Variable>(shape: S, mean: f32, sd: f32) -> V {
        <V as Variable>::F::random_normal(shape, mean, sd)
    }

    pub fn normal_on<S: Into<Shape>, D: Device, V: Variable>(
        shape: S,
        mean: f32,
        sd: f32,
        dev: Option<&mut D>,
    ) -> V {
        <V as Variable>::F::random_normal_on(shape, mean, sd, dev)
    }

    pub fn log_normal<S: Into<Shape>, V: Variable>(shape: S, mean: f32, sd: f32) -> V {
        <V as Variable>::F::random_log_normal(shape, mean, sd)
    }

    pub fn log_normal_on<S: Into<Shape>, D: Device, V: Variable>(
        shape: S,
        mean: f32,
        sd: f32,
        dev: Option<&mut D>,
    ) -> V {
        <V as Variable>::F::random_log_normal_on(shape, mean, sd, dev)
    }

    pub fn gumbel<S: Into<Shape>, V: Variable>(shape: S, mu: f32, beta: f32) -> V {
        <V as Variable>::F::random_gumbel(shape, mu, beta)
    }

    pub fn gumbel_on<S: Into<Shape>, D: Device, V: Variable>(
        shape: S,
        mu: f32,
        beta: f32,
        dev: Option<&mut D>,
    ) -> V {
        <V as Variable>::F::random_gumbel_on(shape, mu, beta, dev)
    }
}

pub mod batch {
    use super::Variable;
    use super::Functions;

    pub fn pick<T: AsRef<V>, V: Variable>(x: T, ids: &[u32]) -> V {
        <V as Variable>::F::batch_pick(x, ids)
    }

    pub fn slice<T: AsRef<V>, V: Variable>(x: T, lower: u32, upper: u32) -> V {
        <V as Variable>::F::batch_slice(x, lower, upper)
    }

    pub fn split<T: AsRef<V>, V: Variable>(x: T, n: u32) -> Vec<V> {
        <V as Variable>::F::batch_split(x, n)
    }

    pub fn concat<TS: AsRef<[T]>, T: AsRef<V>, V: Variable>(xs: TS) -> V {
        <V as Variable>::F::batch_concat(xs)
    }

    pub fn sum<T: AsRef<V>, V: Variable>(x: T) -> V {
        <V as Variable>::F::batch_sum(x)
    }

    pub fn mean<T: AsRef<V>, V: Variable>(x: T) -> V {
        <V as Variable>::F::batch_mean(x)
    }

    pub fn normalize<T: AsRef<V>, V: Variable>(x: T) -> V {
        <V as Variable>::F::batch_normalize(x)
    }
}

pub trait Functions<Var> {
    fn positive<T: AsRef<Var>>(x: T) -> Var;
    fn negative<T: AsRef<Var>>(x: T) -> Var;
    fn add<T1: AsRef<Var>, T2: AsRef<Var>>(a: T1, b: T2) -> Var;
    fn add_const<T: AsRef<Var>>(x: T, k: f32) -> Var;
    fn add_var<T: AsRef<Var>>(k: f32, x: T) -> Var;
    fn subtract<T1: AsRef<Var>, T2: AsRef<Var>>(a: T1, b: T2) -> Var;
    fn subtract_const<T: AsRef<Var>>(x: T, k: f32) -> Var;
    fn subtract_var<T: AsRef<Var>>(k: f32, x: T) -> Var;
    fn multiply<T1: AsRef<Var>, T2: AsRef<Var>>(a: T1, b: T2) -> Var;
    fn multiply_const<T: AsRef<Var>>(x: T, k: f32) -> Var;
    fn multiply_var<T: AsRef<Var>>(k: f32, x: T) -> Var;
    fn divide<T1: AsRef<Var>, T2: AsRef<Var>>(a: T1, b: T2) -> Var;
    fn divide_const<T: AsRef<Var>>(x: T, k: f32) -> Var;
    fn divide_var<T: AsRef<Var>>(k: f32, x: T) -> Var;
    fn pow<T1: AsRef<Var>, T2: AsRef<Var>>(a: T1, b: T2) -> Var;
    fn pow_const<T: AsRef<Var>>(x: T, k: f32) -> Var;
    fn pow_var<T: AsRef<Var>>(k: f32, x: T) -> Var;
    fn pown<T: AsRef<Var>>(x: T, k: i32) -> Var;
    fn input<S: Into<Shape>>(shape: S, data: &[f32]) -> Var;
    fn input_on<S: Into<Shape>, D: Device>(shape: S, data: &[f32], dev: Option<&mut D>) -> Var;
    fn parameter(param: &mut Parameter) -> Var;
    fn copy<T: AsRef<Var>>(x: T) -> Var;
    fn copy_on<T: AsRef<Var>, D: Device>(x: T, dev: Option<&mut D>) -> Var;
    fn pick<T: AsRef<Var>>(x: T, ids: &[u32], dim: u32) -> Var;
    fn slice<T: AsRef<Var>>(x: T, dim: u32, lower: u32, upper: u32) -> Var;
    fn split<T: AsRef<Var>>(x: T, dim: u32, n: u32) -> Vec<Var>;
    fn concat<TS: AsRef<[T]>, T: AsRef<Var>>(xs: TS, dim: u32) -> Var;
    fn reshape<T: AsRef<Var>, S: Into<Shape>>(x: T, new_shape: S) -> Var;
    fn flatten<T: AsRef<Var>>(x: T) -> Var;
    fn transpose<T: AsRef<Var>>(x: T) -> Var;
    fn matmul<T1: AsRef<Var>, T2: AsRef<Var>>(a: T1, b: T2) -> Var;
    fn abs<T: AsRef<Var>>(x: T) -> Var;
    fn sqrt<T: AsRef<Var>>(x: T) -> Var;
    fn exp<T: AsRef<Var>>(x: T) -> Var;
    fn log<T: AsRef<Var>>(x: T) -> Var;
    fn tanh<T: AsRef<Var>>(x: T) -> Var;
    fn sigmoid<T: AsRef<Var>>(x: T) -> Var;
    fn softplus<T: AsRef<Var>>(x: T) -> Var;
    fn sin<T: AsRef<Var>>(x: T) -> Var;
    fn cos<T: AsRef<Var>>(x: T) -> Var;
    fn tan<T: AsRef<Var>>(x: T) -> Var;
    fn relu<T: AsRef<Var>>(x: T) -> Var;
    fn lrelu<T: AsRef<Var>>(x: T) -> Var;
    fn prelu<T: AsRef<Var>>(x: T, a: f32) -> Var;
    fn elu<T: AsRef<Var>>(x: T, a: f32) -> Var;
    fn selu<T: AsRef<Var>>(x: T) -> Var;
    fn max<T: AsRef<Var>>(x: T, dim: u32) -> Var;
    fn min<T: AsRef<Var>>(x: T, dim: u32) -> Var;
    fn sum<T: AsRef<Var>>(x: T, dim: u32) -> Var;
    fn sum_vars<TS: AsRef<[T]>, T: AsRef<Var>>(xs: TS) -> Var;
    fn mean<T: AsRef<Var>>(x: T, dim: u32) -> Var;
    fn mean_vars<TS: AsRef<[T]>, T: AsRef<Var>>(xs: TS) -> Var;
    fn broadcast<T: AsRef<Var>>(x: T, dim: u32, size: u32) -> Var;
    fn logsumexp<T: AsRef<Var>>(x: T, dim: u32) -> Var;
    fn log_softmax<T: AsRef<Var>>(x: T, dim: u32) -> Var;
    fn softmax<T: AsRef<Var>>(x: T, dim: u32) -> Var;
    fn softmax_cross_entropy<T1: AsRef<Var>, T2: AsRef<Var>>(x: T1, t: T2, dim: u32) -> Var;
    fn softmax_cross_entropy_with_ids<T: AsRef<Var>>(x: T, ids: &[u32], dim: u32) -> Var;
    fn stop_gradient<T: AsRef<Var>>(x: T) -> Var;
    fn conv2d<T1: AsRef<Var>, T2: AsRef<Var>>(
        x: T1,
        w: T2,
        padding0: u32,
        padding1: u32,
        stride0: u32,
        stride1: u32,
        dilation0: u32,
        dilation1: u32,
    ) -> Var;
    fn max_pool2d<T: AsRef<Var>>(
        x: T,
        window0: u32,
        window1: u32,
        padding0: u32,
        padding1: u32,
        stride0: u32,
        stride1: u32,
    ) -> Var;
    fn constant<S: Into<Shape>>(shape: S, k: f32) -> Var;
    fn constant_on<S: Into<Shape>, D: Device>(shape: S, k: f32, dev: Option<&mut D>) -> Var;
    fn identity(size: u32) -> Var;
    fn identity_on<D: Device>(size: u32, dev: Option<&mut D>) -> Var;
    fn zeros<S: Into<Shape>>(shape: S) -> Var;
    fn zeros_on<S: Into<Shape>, D: Device>(shape: S, dev: Option<&mut D>) -> Var;
    fn ones<S: Into<Shape>>(shape: S) -> Var;
    fn ones_on<S: Into<Shape>, D: Device>(shape: S, dev: Option<&mut D>) -> Var;
    fn dropout<T: AsRef<Var>>(x: T, rate: f32, enabled: bool) -> Var;
    fn random_bernoulli<S: Into<Shape>>(shape: S, p: f32) -> Var;
    fn random_bernoulli_on<S: Into<Shape>, D: Device>(shape: S, p: f32, dev: Option<&mut D>)
        -> Var;
    fn random_uniform<S: Into<Shape>>(shape: S, lower: f32, upper: f32) -> Var;
    fn random_uniform_on<S: Into<Shape>, D: Device>(
        shape: S,
        lower: f32,
        upper: f32,
        dev: Option<&mut D>,
    ) -> Var;
    fn random_normal<S: Into<Shape>>(shape: S, mean: f32, sd: f32) -> Var;
    fn random_normal_on<S: Into<Shape>, D: Device>(
        shape: S,
        mean: f32,
        sd: f32,
        dev: Option<&mut D>,
    ) -> Var;
    fn random_log_normal<S: Into<Shape>>(shape: S, mean: f32, sd: f32) -> Var;
    fn random_log_normal_on<S: Into<Shape>, D: Device>(
        shape: S,
        mean: f32,
        sd: f32,
        dev: Option<&mut D>,
    ) -> Var;
    fn random_gumbel<S: Into<Shape>>(shape: S, mu: f32, beta: f32) -> Var;
    fn random_gumbel_on<S: Into<Shape>, D: Device>(
        shape: S,
        mu: f32,
        beta: f32,
        dev: Option<&mut D>,
    ) -> Var;
    fn batch_pick<T: AsRef<Var>>(x: T, ids: &[u32]) -> Var;
    fn batch_slice<T: AsRef<Var>>(x: T, lower: u32, upper: u32) -> Var;
    fn batch_split<T: AsRef<Var>>(x: T, n: u32) -> Vec<Var>;
    fn batch_concat<TS: AsRef<[T]>, T: AsRef<Var>>(xs: TS) -> Var;
    fn batch_sum<T: AsRef<Var>>(x: T) -> Var;
    fn batch_mean<T: AsRef<Var>>(x: T) -> Var;
    fn batch_normalize<T: AsRef<Var>>(x: T) -> Var;
}

pub struct FuncImpls<Type> {
    _phantom: PhantomData<Type>,
}

impl Functions<Node> for FuncImpls<Node> {
    #[inline]
    fn positive<T: AsRef<Node>>(x: T) -> Node {
        node_funcs::positive(x)
    }

    #[inline]
    fn negative<T: AsRef<Node>>(x: T) -> Node {
        node_funcs::negative(x)
    }

    #[inline]
    fn add<T1: AsRef<Node>, T2: AsRef<Node>>(a: T1, b: T2) -> Node {
        node_funcs::add(a, b)
    }

    #[inline]
    fn add_const<T: AsRef<Node>>(x: T, k: f32) -> Node {
        node_funcs::add_const(x, k)
    }

    #[inline]
    fn add_var<T: AsRef<Node>>(k: f32, x: T) -> Node {
        node_funcs::add_node(k, x)
    }

    #[inline]
    fn subtract<T1: AsRef<Node>, T2: AsRef<Node>>(a: T1, b: T2) -> Node {
        node_funcs::subtract(a, b)
    }

    #[inline]
    fn subtract_const<T: AsRef<Node>>(x: T, k: f32) -> Node {
        node_funcs::subtract_const(x, k)
    }

    #[inline]
    fn subtract_var<T: AsRef<Node>>(k: f32, x: T) -> Node {
        node_funcs::subtract_node(k, x)
    }

    #[inline]
    fn multiply<T1: AsRef<Node>, T2: AsRef<Node>>(a: T1, b: T2) -> Node {
        node_funcs::multiply(a, b)
    }

    #[inline]
    fn multiply_const<T: AsRef<Node>>(x: T, k: f32) -> Node {
        node_funcs::multiply_const(x, k)
    }

    #[inline]
    fn multiply_var<T: AsRef<Node>>(k: f32, x: T) -> Node {
        node_funcs::multiply_node(k, x)
    }

    #[inline]
    fn divide<T1: AsRef<Node>, T2: AsRef<Node>>(a: T1, b: T2) -> Node {
        node_funcs::divide(a, b)
    }

    #[inline]
    fn divide_const<T: AsRef<Node>>(x: T, k: f32) -> Node {
        node_funcs::divide_const(x, k)
    }

    #[inline]
    fn divide_var<T: AsRef<Node>>(k: f32, x: T) -> Node {
        node_funcs::divide_node(k, x)
    }

    #[inline]
    fn pow<T1: AsRef<Node>, T2: AsRef<Node>>(a: T1, b: T2) -> Node {
        node_funcs::pow(a, b)
    }

    #[inline]
    fn pow_const<T: AsRef<Node>>(x: T, k: f32) -> Node {
        node_funcs::pow_const(x, k)
    }

    #[inline]
    fn pow_var<T: AsRef<Node>>(k: f32, x: T) -> Node {
        node_funcs::pow_node(k, x)
    }

    #[inline]
    fn pown<T: AsRef<Node>>(x: T, k: i32) -> Node {
        node_funcs::pown(x, k)
    }

    #[inline]
    fn input<S: Into<Shape>>(shape: S, data: &[f32]) -> Node {
        node_funcs::input(shape, data)
    }

    #[inline]
    fn input_on<S: Into<Shape>, D: Device>(shape: S, data: &[f32], dev: Option<&mut D>) -> Node {
        node_funcs::input_on(shape, data, dev)
    }

    #[inline]
    fn copy<T: AsRef<Node>>(x: T) -> Node {
        node_funcs::copy(x)
    }

    #[inline]
    fn copy_on<T: AsRef<Node>, D: Device>(x: T, dev: Option<&mut D>) -> Node {
        node_funcs::copy_on(x, dev)
    }

    #[inline]
    fn parameter(param: &mut Parameter) -> Node {
        node_funcs::parameter(param)
    }

    #[inline]
    fn pick<T: AsRef<Node>>(x: T, ids: &[u32], dim: u32) -> Node {
        node_funcs::pick(x, ids, dim)
    }

    #[inline]
    fn slice<T: AsRef<Node>>(x: T, dim: u32, lower: u32, upper: u32) -> Node {
        node_funcs::slice(x, dim, lower, upper)
    }

    #[inline]
    fn split<T: AsRef<Node>>(x: T, dim: u32, n: u32) -> Vec<Node> {
        node_funcs::split(x, dim, n)
    }

    #[inline]
    fn concat<TS: AsRef<[T]>, T: AsRef<Node>>(xs: TS, dim: u32) -> Node {
        node_funcs::concat(xs, dim)
    }

    #[inline]
    fn reshape<T: AsRef<Node>, S: Into<Shape>>(x: T, new_shape: S) -> Node {
        node_funcs::reshape(x, new_shape)
    }

    #[inline]
    fn flatten<T: AsRef<Node>>(x: T) -> Node {
        node_funcs::flatten(x)
    }

    #[inline]
    fn transpose<T: AsRef<Node>>(x: T) -> Node {
        node_funcs::transpose(x)
    }

    #[inline]
    fn matmul<T1: AsRef<Node>, T2: AsRef<Node>>(a: T1, b: T2) -> Node {
        node_funcs::matmul(a, b)
    }

    #[inline]
    fn abs<T: AsRef<Node>>(x: T) -> Node {
        node_funcs::abs(x)
    }

    #[inline]
    fn sqrt<T: AsRef<Node>>(x: T) -> Node {
        node_funcs::sqrt(x)
    }

    #[inline]
    fn exp<T: AsRef<Node>>(x: T) -> Node {
        node_funcs::exp(x)
    }

    #[inline]
    fn log<T: AsRef<Node>>(x: T) -> Node {
        node_funcs::log(x)
    }

    #[inline]
    fn tanh<T: AsRef<Node>>(x: T) -> Node {
        node_funcs::tanh(x)
    }

    #[inline]
    fn sigmoid<T: AsRef<Node>>(x: T) -> Node {
        node_funcs::sigmoid(x)
    }

    #[inline]
    fn softplus<T: AsRef<Node>>(x: T) -> Node {
        node_funcs::softplus(x)
    }

    #[inline]
    fn sin<T: AsRef<Node>>(x: T) -> Node {
        node_funcs::sin(x)
    }

    #[inline]
    fn cos<T: AsRef<Node>>(x: T) -> Node {
        node_funcs::cos(x)
    }

    #[inline]
    fn tan<T: AsRef<Node>>(x: T) -> Node {
        node_funcs::tan(x)
    }

    #[inline]
    fn relu<T: AsRef<Node>>(x: T) -> Node {
        node_funcs::relu(x)
    }

    #[inline]
    fn lrelu<T: AsRef<Node>>(x: T) -> Node {
        node_funcs::lrelu(x)
    }

    #[inline]
    fn prelu<T: AsRef<Node>>(x: T, a: f32) -> Node {
        node_funcs::prelu(x, a)
    }

    #[inline]
    fn elu<T: AsRef<Node>>(x: T, a: f32) -> Node {
        node_funcs::elu(x, a)
    }

    #[inline]
    fn selu<T: AsRef<Node>>(x: T) -> Node {
        node_funcs::selu(x)
    }

    #[inline]
    fn max<T: AsRef<Node>>(x: T, dim: u32) -> Node {
        node_funcs::max(x, dim)
    }

    #[inline]
    fn min<T: AsRef<Node>>(x: T, dim: u32) -> Node {
        node_funcs::min(x, dim)
    }

    #[inline]
    fn sum<T: AsRef<Node>>(x: T, dim: u32) -> Node {
        node_funcs::sum(x, dim)
    }

    #[inline]
    fn sum_vars<TS: AsRef<[T]>, T: AsRef<Node>>(xs: TS) -> Node {
        node_funcs::sum_nodes(xs)
    }

    #[inline]
    fn mean<T: AsRef<Node>>(x: T, dim: u32) -> Node {
        node_funcs::mean(x, dim)
    }

    #[inline]
    fn mean_vars<TS: AsRef<[T]>, T: AsRef<Node>>(xs: TS) -> Node {
        node_funcs::mean_nodes(xs)
    }

    #[inline]
    fn broadcast<T: AsRef<Node>>(x: T, dim: u32, size: u32) -> Node {
        node_funcs::broadcast(x, dim, size)
    }

    #[inline]
    fn logsumexp<T: AsRef<Node>>(x: T, dim: u32) -> Node {
        node_funcs::logsumexp(x, dim)
    }

    #[inline]
    fn log_softmax<T: AsRef<Node>>(x: T, dim: u32) -> Node {
        node_funcs::log_softmax(x, dim)
    }

    #[inline]
    fn softmax<T: AsRef<Node>>(x: T, dim: u32) -> Node {
        node_funcs::softmax(x, dim)
    }

    #[inline]
    fn softmax_cross_entropy<T1: AsRef<Node>, T2: AsRef<Node>>(x: T1, t: T2, dim: u32) -> Node {
        node_funcs::softmax_cross_entropy(x, t, dim)
    }

    #[inline]
    fn softmax_cross_entropy_with_ids<T: AsRef<Node>>(x: T, ids: &[u32], dim: u32) -> Node {
        node_funcs::softmax_cross_entropy_with_ids(x, ids, dim)
    }

    #[inline]
    fn stop_gradient<T: AsRef<Node>>(x: T) -> Node {
        node_funcs::stop_gradient(x)
    }

    #[inline]
    fn conv2d<T1: AsRef<Node>, T2: AsRef<Node>>(
        x: T1,
        w: T2,
        padding0: u32,
        padding1: u32,
        stride0: u32,
        stride1: u32,
        dilation0: u32,
        dilation1: u32,
    ) -> Node {
        node_funcs::conv2d(
            x,
            w,
            padding0,
            padding1,
            stride0,
            stride1,
            dilation0,
            dilation1,
        )
    }

    #[inline]
    fn max_pool2d<T: AsRef<Node>>(
        x: T,
        window0: u32,
        window1: u32,
        padding0: u32,
        padding1: u32,
        stride0: u32,
        stride1: u32,
    ) -> Node {
        node_funcs::max_pool2d(x, window0, window1, padding0, padding1, stride0, stride1)
    }

    #[inline]
    fn constant<S: Into<Shape>>(shape: S, k: f32) -> Node {
        node_funcs::constant(shape, k)
    }

    #[inline]
    fn constant_on<S: Into<Shape>, D: Device>(shape: S, k: f32, dev: Option<&mut D>) -> Node {
        node_funcs::constant_on(shape, k, dev)
    }

    #[inline]
    fn identity(size: u32) -> Node {
        node_funcs::identity(size)
    }

    #[inline]
    fn identity_on<D: Device>(size: u32, dev: Option<&mut D>) -> Node {
        node_funcs::identity_on(size, dev)
    }

    #[inline]
    fn zeros<S: Into<Shape>>(shape: S) -> Node {
        node_funcs::zeros(shape)
    }

    #[inline]
    fn zeros_on<S: Into<Shape>, D: Device>(shape: S, dev: Option<&mut D>) -> Node {
        node_funcs::zeros_on(shape, dev)
    }

    #[inline]
    fn ones<S: Into<Shape>>(shape: S) -> Node {
        node_funcs::ones(shape)
    }

    #[inline]
    fn ones_on<S: Into<Shape>, D: Device>(shape: S, dev: Option<&mut D>) -> Node {
        node_funcs::ones_on(shape, dev)
    }

    #[inline]
    fn dropout<T: AsRef<Node>>(x: T, rate: f32, enabled: bool) -> Node {
        node_funcs::dropout(x, rate, enabled)
    }

    #[inline]
    fn random_bernoulli<S: Into<Shape>>(shape: S, p: f32) -> Node {
        node_funcs::random::bernoulli(shape, p)
    }

    #[inline]
    fn random_bernoulli_on<S: Into<Shape>, D: Device>(
        shape: S,
        p: f32,
        dev: Option<&mut D>,
    ) -> Node {
        node_funcs::random::bernoulli_on(shape, p, dev)
    }

    #[inline]
    fn random_uniform<S: Into<Shape>>(shape: S, lower: f32, upper: f32) -> Node {
        node_funcs::random::uniform(shape, lower, upper)
    }

    #[inline]
    fn random_uniform_on<S: Into<Shape>, D: Device>(
        shape: S,
        lower: f32,
        upper: f32,
        dev: Option<&mut D>,
    ) -> Node {
        node_funcs::random::uniform_on(shape, lower, upper, dev)
    }

    #[inline]
    fn random_normal<S: Into<Shape>>(shape: S, mean: f32, sd: f32) -> Node {
        node_funcs::random::normal(shape, mean, sd)
    }

    #[inline]
    fn random_normal_on<S: Into<Shape>, D: Device>(
        shape: S,
        mean: f32,
        sd: f32,
        dev: Option<&mut D>,
    ) -> Node {
        node_funcs::random::normal_on(shape, mean, sd, dev)
    }

    #[inline]
    fn random_log_normal<S: Into<Shape>>(shape: S, mean: f32, sd: f32) -> Node {
        node_funcs::random::log_normal(shape, mean, sd)
    }

    #[inline]
    fn random_log_normal_on<S: Into<Shape>, D: Device>(
        shape: S,
        mean: f32,
        sd: f32,
        dev: Option<&mut D>,
    ) -> Node {
        node_funcs::random::log_normal_on(shape, mean, sd, dev)
    }

    #[inline]
    fn random_gumbel<S: Into<Shape>>(shape: S, mu: f32, beta: f32) -> Node {
        node_funcs::random::gumbel(shape, mu, beta)
    }

    #[inline]
    fn random_gumbel_on<S: Into<Shape>, D: Device>(
        shape: S,
        mu: f32,
        beta: f32,
        dev: Option<&mut D>,
    ) -> Node {
        node_funcs::random::gumbel_on(shape, mu, beta, dev)
    }

    #[inline]
    fn batch_pick<T: AsRef<Node>>(x: T, ids: &[u32]) -> Node {
        node_funcs::batch::pick(x, ids)
    }

    #[inline]
    fn batch_slice<T: AsRef<Node>>(x: T, lower: u32, upper: u32) -> Node {
        node_funcs::batch::slice(x, lower, upper)
    }

    #[inline]
    fn batch_split<T: AsRef<Node>>(x: T, n: u32) -> Vec<Node> {
        node_funcs::batch::split(x, n)
    }

    #[inline]
    fn batch_concat<TS: AsRef<[T]>, T: AsRef<Node>>(xs: TS) -> Node {
        node_funcs::batch::concat(xs)
    }

    #[inline]
    fn batch_sum<T: AsRef<Node>>(x: T) -> Node {
        node_funcs::batch::sum(x)
    }

    #[inline]
    fn batch_mean<T: AsRef<Node>>(x: T) -> Node {
        node_funcs::batch::mean(x)
    }

    #[inline]
    fn batch_normalize<T: AsRef<Node>>(x: T) -> Node {
        node_funcs::batch::normalize(x)
    }
}

impl Functions<Tensor> for FuncImpls<Tensor> {
    #[inline]
    fn positive<T: AsRef<Tensor>>(x: T) -> Tensor {
        tensor_funcs::positive(x)
    }

    #[inline]
    fn negative<T: AsRef<Tensor>>(x: T) -> Tensor {
        tensor_funcs::negative(x)
    }

    #[inline]
    fn add<T1: AsRef<Tensor>, T2: AsRef<Tensor>>(a: T1, b: T2) -> Tensor {
        tensor_funcs::add(a, b)
    }

    #[inline]
    fn add_const<T: AsRef<Tensor>>(x: T, k: f32) -> Tensor {
        tensor_funcs::add_const(x, k)
    }

    #[inline]
    fn add_var<T: AsRef<Tensor>>(k: f32, x: T) -> Tensor {
        tensor_funcs::add_tensor(k, x)
    }

    #[inline]
    fn subtract<T1: AsRef<Tensor>, T2: AsRef<Tensor>>(a: T1, b: T2) -> Tensor {
        tensor_funcs::subtract(a, b)
    }

    #[inline]
    fn subtract_const<T: AsRef<Tensor>>(x: T, k: f32) -> Tensor {
        tensor_funcs::subtract_const(x, k)
    }

    #[inline]
    fn subtract_var<T: AsRef<Tensor>>(k: f32, x: T) -> Tensor {
        tensor_funcs::subtract_tensor(k, x)
    }

    #[inline]
    fn multiply<T1: AsRef<Tensor>, T2: AsRef<Tensor>>(a: T1, b: T2) -> Tensor {
        tensor_funcs::multiply(a, b)
    }

    #[inline]
    fn multiply_const<T: AsRef<Tensor>>(x: T, k: f32) -> Tensor {
        tensor_funcs::multiply_const(x, k)
    }

    #[inline]
    fn multiply_var<T: AsRef<Tensor>>(k: f32, x: T) -> Tensor {
        tensor_funcs::multiply_tensor(k, x)
    }

    #[inline]
    fn divide<T1: AsRef<Tensor>, T2: AsRef<Tensor>>(a: T1, b: T2) -> Tensor {
        tensor_funcs::divide(a, b)
    }

    #[inline]
    fn divide_const<T: AsRef<Tensor>>(x: T, k: f32) -> Tensor {
        tensor_funcs::divide_const(x, k)
    }

    #[inline]
    fn divide_var<T: AsRef<Tensor>>(k: f32, x: T) -> Tensor {
        tensor_funcs::divide_tensor(k, x)
    }

    #[inline]
    fn pow<T1: AsRef<Tensor>, T2: AsRef<Tensor>>(a: T1, b: T2) -> Tensor {
        tensor_funcs::pow(a, b)
    }

    #[inline]
    fn pow_const<T: AsRef<Tensor>>(x: T, k: f32) -> Tensor {
        tensor_funcs::pow_const(x, k)
    }

    #[inline]
    fn pow_var<T: AsRef<Tensor>>(k: f32, x: T) -> Tensor {
        tensor_funcs::pow_tensor(k, x)
    }

    #[inline]
    fn pown<T: AsRef<Tensor>>(x: T, k: i32) -> Tensor {
        tensor_funcs::pown(x, k)
    }

    #[inline]
    fn input<S: Into<Shape>>(shape: S, data: &[f32]) -> Tensor {
        tensor_funcs::input(shape, data)
    }

    #[inline]
    fn input_on<S: Into<Shape>, D: Device>(shape: S, data: &[f32], dev: Option<&mut D>) -> Tensor {
        tensor_funcs::input_on(shape, data, dev)
    }

    #[inline]
    fn copy<T: AsRef<Tensor>>(x: T) -> Tensor {
        tensor_funcs::copy(x)
    }

    #[inline]
    fn copy_on<T: AsRef<Tensor>, D: Device>(x: T, dev: Option<&mut D>) -> Tensor {
        tensor_funcs::copy_on(x, dev)
    }

    #[inline]
    fn parameter(param: &mut Parameter) -> Tensor {
        tensor_funcs::parameter(param)
    }

    #[inline]
    fn pick<T: AsRef<Tensor>>(x: T, ids: &[u32], dim: u32) -> Tensor {
        tensor_funcs::pick(x, ids, dim)
    }

    #[inline]
    fn slice<T: AsRef<Tensor>>(x: T, dim: u32, lower: u32, upper: u32) -> Tensor {
        tensor_funcs::slice(x, dim, lower, upper)
    }

    #[inline]
    fn split<T: AsRef<Tensor>>(x: T, dim: u32, n: u32) -> Vec<Tensor> {
        tensor_funcs::split(x, dim, n)
    }

    #[inline]
    fn concat<TS: AsRef<[T]>, T: AsRef<Tensor>>(xs: TS, dim: u32) -> Tensor {
        tensor_funcs::concat(xs, dim)
    }

    #[inline]
    fn reshape<T: AsRef<Tensor>, S: Into<Shape>>(x: T, new_shape: S) -> Tensor {
        tensor_funcs::reshape(x, new_shape)
    }

    #[inline]
    fn flatten<T: AsRef<Tensor>>(x: T) -> Tensor {
        tensor_funcs::flatten(x)
    }

    #[inline]
    fn transpose<T: AsRef<Tensor>>(x: T) -> Tensor {
        tensor_funcs::transpose(x)
    }

    #[inline]
    fn matmul<T1: AsRef<Tensor>, T2: AsRef<Tensor>>(a: T1, b: T2) -> Tensor {
        tensor_funcs::matmul(a, b)
    }

    #[inline]
    fn abs<T: AsRef<Tensor>>(x: T) -> Tensor {
        tensor_funcs::abs(x)
    }

    #[inline]
    fn sqrt<T: AsRef<Tensor>>(x: T) -> Tensor {
        tensor_funcs::sqrt(x)
    }

    #[inline]
    fn exp<T: AsRef<Tensor>>(x: T) -> Tensor {
        tensor_funcs::exp(x)
    }

    #[inline]
    fn log<T: AsRef<Tensor>>(x: T) -> Tensor {
        tensor_funcs::log(x)
    }

    #[inline]
    fn tanh<T: AsRef<Tensor>>(x: T) -> Tensor {
        tensor_funcs::tanh(x)
    }

    #[inline]
    fn sigmoid<T: AsRef<Tensor>>(x: T) -> Tensor {
        tensor_funcs::sigmoid(x)
    }

    #[inline]
    fn softplus<T: AsRef<Tensor>>(x: T) -> Tensor {
        tensor_funcs::softplus(x)
    }

    #[inline]
    fn sin<T: AsRef<Tensor>>(x: T) -> Tensor {
        tensor_funcs::sin(x)
    }

    #[inline]
    fn cos<T: AsRef<Tensor>>(x: T) -> Tensor {
        tensor_funcs::cos(x)
    }

    #[inline]
    fn tan<T: AsRef<Tensor>>(x: T) -> Tensor {
        tensor_funcs::tan(x)
    }

    #[inline]
    fn relu<T: AsRef<Tensor>>(x: T) -> Tensor {
        tensor_funcs::relu(x)
    }

    #[inline]
    fn lrelu<T: AsRef<Tensor>>(x: T) -> Tensor {
        tensor_funcs::lrelu(x)
    }

    #[inline]
    fn prelu<T: AsRef<Tensor>>(x: T, a: f32) -> Tensor {
        tensor_funcs::prelu(x, a)
    }

    #[inline]
    fn elu<T: AsRef<Tensor>>(x: T, a: f32) -> Tensor {
        tensor_funcs::elu(x, a)
    }

    #[inline]
    fn selu<T: AsRef<Tensor>>(x: T) -> Tensor {
        tensor_funcs::selu(x)
    }

    #[inline]
    fn max<T: AsRef<Tensor>>(x: T, dim: u32) -> Tensor {
        tensor_funcs::max(x, dim)
    }

    #[inline]
    fn min<T: AsRef<Tensor>>(x: T, dim: u32) -> Tensor {
        tensor_funcs::min(x, dim)
    }

    #[inline]
    fn sum<T: AsRef<Tensor>>(x: T, dim: u32) -> Tensor {
        tensor_funcs::sum(x, dim)
    }

    #[inline]
    fn sum_vars<TS: AsRef<[T]>, T: AsRef<Tensor>>(xs: TS) -> Tensor {
        tensor_funcs::sum_tensors(xs)
    }

    #[inline]
    fn mean<T: AsRef<Tensor>>(x: T, dim: u32) -> Tensor {
        tensor_funcs::mean(x, dim)
    }

    #[inline]
    fn mean_vars<TS: AsRef<[T]>, T: AsRef<Tensor>>(xs: TS) -> Tensor {
        tensor_funcs::mean_tensors(xs)
    }

    #[inline]
    fn broadcast<T: AsRef<Tensor>>(x: T, dim: u32, size: u32) -> Tensor {
        tensor_funcs::broadcast(x, dim, size)
    }

    #[inline]
    fn logsumexp<T: AsRef<Tensor>>(x: T, dim: u32) -> Tensor {
        tensor_funcs::logsumexp(x, dim)
    }

    #[inline]
    fn log_softmax<T: AsRef<Tensor>>(x: T, dim: u32) -> Tensor {
        tensor_funcs::log_softmax(x, dim)
    }

    #[inline]
    fn softmax<T: AsRef<Tensor>>(x: T, dim: u32) -> Tensor {
        tensor_funcs::softmax(x, dim)
    }

    #[inline]
    fn softmax_cross_entropy<T1: AsRef<Tensor>, T2: AsRef<Tensor>>(
        x: T1,
        t: T2,
        dim: u32,
    ) -> Tensor {
        tensor_funcs::softmax_cross_entropy(x, t, dim)
    }

    #[inline]
    fn softmax_cross_entropy_with_ids<T: AsRef<Tensor>>(x: T, ids: &[u32], dim: u32) -> Tensor {
        tensor_funcs::softmax_cross_entropy_with_ids(x, ids, dim)
    }

    #[inline]
    fn stop_gradient<T: AsRef<Tensor>>(x: T) -> Tensor {
        tensor_funcs::stop_gradient(x)
    }

    #[inline]
    fn conv2d<T1: AsRef<Tensor>, T2: AsRef<Tensor>>(
        x: T1,
        w: T2,
        padding0: u32,
        padding1: u32,
        stride0: u32,
        stride1: u32,
        dilation0: u32,
        dilation1: u32,
    ) -> Tensor {
        tensor_funcs::conv2d(
            x,
            w,
            padding0,
            padding1,
            stride0,
            stride1,
            dilation0,
            dilation1,
        )
    }

    #[inline]
    fn max_pool2d<T: AsRef<Tensor>>(
        x: T,
        window0: u32,
        window1: u32,
        padding0: u32,
        padding1: u32,
        stride0: u32,
        stride1: u32,
    ) -> Tensor {
        tensor_funcs::max_pool2d(x, window0, window1, padding0, padding1, stride0, stride1)
    }

    #[inline]
    fn constant<S: Into<Shape>>(shape: S, k: f32) -> Tensor {
        tensor_funcs::constant(shape, k)
    }

    #[inline]
    fn constant_on<S: Into<Shape>, D: Device>(shape: S, k: f32, dev: Option<&mut D>) -> Tensor {
        tensor_funcs::constant_on(shape, k, dev)
    }

    #[inline]
    fn identity(size: u32) -> Tensor {
        tensor_funcs::identity(size)
    }

    #[inline]
    fn identity_on<D: Device>(size: u32, dev: Option<&mut D>) -> Tensor {
        tensor_funcs::identity_on(size, dev)
    }

    #[inline]
    fn zeros<S: Into<Shape>>(shape: S) -> Tensor {
        tensor_funcs::zeros(shape)
    }

    #[inline]
    fn zeros_on<S: Into<Shape>, D: Device>(shape: S, dev: Option<&mut D>) -> Tensor {
        tensor_funcs::zeros_on(shape, dev)
    }

    #[inline]
    fn ones<S: Into<Shape>>(shape: S) -> Tensor {
        tensor_funcs::ones(shape)
    }

    #[inline]
    fn ones_on<S: Into<Shape>, D: Device>(shape: S, dev: Option<&mut D>) -> Tensor {
        tensor_funcs::ones_on(shape, dev)
    }

    #[inline]
    fn dropout<T: AsRef<Tensor>>(x: T, rate: f32, enabled: bool) -> Tensor {
        tensor_funcs::dropout(x, rate, enabled)
    }

    #[inline]
    fn random_bernoulli<S: Into<Shape>>(shape: S, p: f32) -> Tensor {
        tensor_funcs::random::bernoulli(shape, p)
    }

    #[inline]
    fn random_bernoulli_on<S: Into<Shape>, D: Device>(
        shape: S,
        p: f32,
        dev: Option<&mut D>,
    ) -> Tensor {
        tensor_funcs::random::bernoulli_on(shape, p, dev)
    }

    #[inline]
    fn random_uniform<S: Into<Shape>>(shape: S, lower: f32, upper: f32) -> Tensor {
        tensor_funcs::random::uniform(shape, lower, upper)
    }

    #[inline]
    fn random_uniform_on<S: Into<Shape>, D: Device>(
        shape: S,
        lower: f32,
        upper: f32,
        dev: Option<&mut D>,
    ) -> Tensor {
        tensor_funcs::random::uniform_on(shape, lower, upper, dev)
    }

    #[inline]
    fn random_normal<S: Into<Shape>>(shape: S, mean: f32, sd: f32) -> Tensor {
        tensor_funcs::random::normal(shape, mean, sd)
    }

    #[inline]
    fn random_normal_on<S: Into<Shape>, D: Device>(
        shape: S,
        mean: f32,
        sd: f32,
        dev: Option<&mut D>,
    ) -> Tensor {
        tensor_funcs::random::normal_on(shape, mean, sd, dev)
    }

    #[inline]
    fn random_log_normal<S: Into<Shape>>(shape: S, mean: f32, sd: f32) -> Tensor {
        tensor_funcs::random::log_normal(shape, mean, sd)
    }

    #[inline]
    fn random_log_normal_on<S: Into<Shape>, D: Device>(
        shape: S,
        mean: f32,
        sd: f32,
        dev: Option<&mut D>,
    ) -> Tensor {
        tensor_funcs::random::log_normal_on(shape, mean, sd, dev)
    }

    #[inline]
    fn random_gumbel<S: Into<Shape>>(shape: S, mu: f32, beta: f32) -> Tensor {
        tensor_funcs::random::gumbel(shape, mu, beta)
    }

    #[inline]
    fn random_gumbel_on<S: Into<Shape>, D: Device>(
        shape: S,
        mu: f32,
        beta: f32,
        dev: Option<&mut D>,
    ) -> Tensor {
        tensor_funcs::random::gumbel_on(shape, mu, beta, dev)
    }

    #[inline]
    fn batch_pick<T: AsRef<Tensor>>(x: T, ids: &[u32]) -> Tensor {
        tensor_funcs::batch::pick(x, ids)
    }

    #[inline]
    fn batch_slice<T: AsRef<Tensor>>(x: T, lower: u32, upper: u32) -> Tensor {
        tensor_funcs::batch::slice(x, lower, upper)
    }

    #[inline]
    fn batch_split<T: AsRef<Tensor>>(x: T, n: u32) -> Vec<Tensor> {
        tensor_funcs::batch::split(x, n)
    }

    #[inline]
    fn batch_concat<TS: AsRef<[T]>, T: AsRef<Tensor>>(xs: TS) -> Tensor {
        tensor_funcs::batch::concat(xs)
    }

    #[inline]
    fn batch_sum<T: AsRef<Tensor>>(x: T) -> Tensor {
        tensor_funcs::batch::sum(x)
    }

    #[inline]
    fn batch_mean<T: AsRef<Tensor>>(x: T) -> Tensor {
        tensor_funcs::batch::mean(x)
    }

    #[inline]
    fn batch_normalize<T: AsRef<Tensor>>(x: T) -> Tensor {
        tensor_funcs::batch::normalize(x)
    }
}
