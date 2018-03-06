use std::marker::PhantomData;
use Device;
use Node;
use Parameter;
use Shape;
use Tensor;

use super::node_funcs;
use super::tensor_funcs;

pub trait Functions<Var> {
    fn pown<T: AsRef<Var>>(x: T, k: i32) -> Var;
    fn input<S: Into<Shape>>(shape: S, data: &[f32]) -> Var;
    fn input_on<S: Into<Shape>, D: Device>(shape: S, data: &[f32], dev: Option<&mut D>) -> Var;
    fn parameter(param: &mut Parameter) -> Var;
    fn copy<T: AsRef<Var>>(x: T) -> Var;
    fn copy_on<T: AsRef<Var>, D: Device>(x: T, dev: Option<&mut D>) -> Var;
    fn pick<T: AsRef<Var>>(x: T, ids: &[u32], dim: u32) -> Var;
    fn slice<T: AsRef<Var>>(x: T, dim: u32, lower: u32, upper: u32) -> Var;
    fn split<T: AsRef<Var>>(x: T, dim: u32, n: u32) -> Vec<Var>;
    fn concat<T: AsRef<Var>>(xs: &[T], dim: u32) -> Var;
    fn reshape<T: AsRef<Var>, S: Into<Shape>>(x: T, new_shape: S) -> Var;
    fn flatten<T: AsRef<Var>>(x: T) -> Var;
    fn transpose<T: AsRef<Var>>(x: T) -> Var;
    fn matmul<T1: AsRef<Var>, T2: AsRef<Var>>(a: T1, b: T2) -> Var;
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
    fn sum<T: AsRef<Var>>(x: T, dim: u32) -> Var;
    fn sum_vars<T: AsRef<Var>>(xs: &[T]) -> Var;
    fn mean<T: AsRef<Var>>(x: T, dim: u32) -> Var;
    fn mean_vars<T: AsRef<Var>>(xs: &[T]) -> Var;
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
    fn batch_sum<T: AsRef<Var>>(x: T) -> Var;
    fn batch_mean<T: AsRef<Var>>(x: T) -> Var;
    fn batch_normalize<T: AsRef<Var>>(x: T) -> Var;
}

pub struct FuncImpls<Type> {
    _phantom: PhantomData<Type>,
}

pub fn pown<T: AsRef<Var>, Var>(x: T, k: i32) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::pown(x, k)
}

pub fn input<S: Into<Shape>, Var>(shape: S, data: &[f32]) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::input(shape, data)
}

pub fn input_on<S: Into<Shape>, D: Device, Var>(shape: S, data: &[f32], dev: Option<&mut D>) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::input_on(shape, data, dev)
}

pub fn parameter<Var>(param: &mut Parameter) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::parameter(param)
}

pub fn copy<T: AsRef<Var>, Var>(x: T) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::copy(x)
}

pub fn copy_on<T: AsRef<Var>, D: Device, Var>(x: T, dev: Option<&mut D>) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::copy_on(x, dev)
}

pub fn pick<T: AsRef<Var>, Var>(x: T, ids: &[u32], dim: u32) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::pick(x, ids, dim)
}

pub fn slice<T: AsRef<Var>, Var>(x: T, dim: u32, lower: u32, upper: u32) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::slice(x, dim, lower, upper)
}

pub fn split<T: AsRef<Var>, Var>(x: T, dim: u32, n: u32) -> Vec<Var>
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::split(x, dim, n)
}

pub fn concat<T: AsRef<Var>, Var>(xs: &[T], dim: u32) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::concat(xs, dim)
}

pub fn reshape<T: AsRef<Var>, S: Into<Shape>, Var>(x: T, new_shape: S) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::reshape(x, new_shape)
}

pub fn flatten<T: AsRef<Var>, Var>(x: T) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::flatten(x)
}

pub fn transpose<T: AsRef<Var>, Var>(x: T) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::transpose(x)
}

pub fn matmul<T1: AsRef<Var>, T2: AsRef<Var>, Var>(a: T1, b: T2) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::matmul(a, b)
}

pub fn sqrt<T: AsRef<Var>, Var>(x: T) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::sqrt(x)
}

pub fn exp<T: AsRef<Var>, Var>(x: T) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::exp(x)
}

pub fn log<T: AsRef<Var>, Var>(x: T) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::log(x)
}

pub fn tanh<T: AsRef<Var>, Var>(x: T) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::tanh(x)
}

pub fn sigmoid<T: AsRef<Var>, Var>(x: T) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::sigmoid(x)
}

pub fn softplus<T: AsRef<Var>, Var>(x: T) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::softplus(x)
}

pub fn sin<T: AsRef<Var>, Var>(x: T) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::sin(x)
}

pub fn cos<T: AsRef<Var>, Var>(x: T) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::cos(x)
}

pub fn tan<T: AsRef<Var>, Var>(x: T) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::tan(x)
}

pub fn relu<T: AsRef<Var>, Var>(x: T) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::relu(x)
}

pub fn lrelu<T: AsRef<Var>, Var>(x: T) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::lrelu(x)
}

pub fn prelu<T: AsRef<Var>, Var>(x: T, a: f32) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::prelu(x, a)
}

pub fn elu<T: AsRef<Var>, Var>(x: T, a: f32) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::elu(x, a)
}

pub fn selu<T: AsRef<Var>, Var>(x: T) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::selu(x)
}

pub fn sum<T: AsRef<Var>, Var>(x: T, dim: u32) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::sum(x, dim)
}

pub fn sum_vars<T: AsRef<Var>, Var>(xs: &[T]) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::sum_vars(xs)
}

pub fn mean<T: AsRef<Var>, Var>(x: T, dim: u32) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::mean(x, dim)
}

pub fn mean_vars<T: AsRef<Var>, Var>(xs: &[T]) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::mean_vars(xs)
}

pub fn broadcast<T: AsRef<Var>, Var>(x: T, dim: u32, size: u32) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::broadcast(x, dim, size)
}

pub fn logsumexp<T: AsRef<Var>, Var>(x: T, dim: u32) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::logsumexp(x, dim)
}

pub fn log_softmax<T: AsRef<Var>, Var>(x: T, dim: u32) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::log_softmax(x, dim)
}

pub fn softmax<T: AsRef<Var>, Var>(x: T, dim: u32) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::softmax(x, dim)
}

pub fn softmax_cross_entropy<T1: AsRef<Var>, T2: AsRef<Var>, Var>(x: T1, t: T2, dim: u32) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::softmax_cross_entropy(x, t, dim)
}

pub fn softmax_cross_entropy_with_ids<T: AsRef<Var>, Var>(x: T, ids: &[u32], dim: u32) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::softmax_cross_entropy_with_ids(x, ids, dim)
}

pub fn stop_gradient<T: AsRef<Var>, Var>(x: T) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::stop_gradient(x)
}

pub fn conv2d<T1: AsRef<Var>, T2: AsRef<Var>, Var>(
    x: T1,
    w: T2,
    padding0: u32,
    padding1: u32,
    stride0: u32,
    stride1: u32,
    dilation0: u32,
    dilation1: u32,
) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::conv2d(
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

pub fn max_pool2d<T: AsRef<Var>, Var>(
    x: T,
    window0: u32,
    window1: u32,
    padding0: u32,
    padding1: u32,
    stride0: u32,
    stride1: u32,
) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::max_pool2d(
        x,
        window0,
        window1,
        padding0,
        padding1,
        stride0,
        stride1,
    )
}

pub fn constant<S: Into<Shape>, Var>(shape: S, k: f32) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::constant(shape, k)
}

pub fn constant_on<S: Into<Shape>, D: Device, Var>(shape: S, k: f32, dev: Option<&mut D>) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::constant_on(shape, k, dev)
}

pub fn identity<Var>(size: u32) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::identity(size)
}

pub fn identity_on<D: Device, Var>(size: u32, dev: Option<&mut D>) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::identity_on(size, dev)
}

pub fn zeros<S: Into<Shape>, Var>(shape: S) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::zeros(shape)
}

pub fn zeros_on<S: Into<Shape>, D: Device, Var>(shape: S, dev: Option<&mut D>) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::zeros_on(shape, dev)
}

pub fn ones<S: Into<Shape>, Var>(shape: S) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::ones(shape)
}

pub fn ones_on<S: Into<Shape>, D: Device, Var>(shape: S, dev: Option<&mut D>) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::ones_on(shape, dev)
}

pub fn dropout<T: AsRef<Var>, Var>(x: T, rate: f32, enabled: bool) -> Var
where
    FuncImpls<Var>: Functions<Var>,
{
    <FuncImpls<Var> as Functions<Var>>::dropout(x, rate, enabled)
}

pub mod random {
    use super::FuncImpls;
    use super::Functions;
    use Device;
    use Shape;

    pub fn bernoulli<S: Into<Shape>, Var>(shape: S, p: f32) -> Var
    where
        FuncImpls<Var>: Functions<Var>,
    {
        <FuncImpls<Var> as Functions<Var>>::random_bernoulli(shape, p)
    }

    pub fn bernoulli_on<S: Into<Shape>, D: Device, Var>(
        shape: S,
        p: f32,
        dev: Option<&mut D>,
    ) -> Var
    where
        FuncImpls<Var>: Functions<Var>,
    {
        <FuncImpls<Var> as Functions<Var>>::random_bernoulli_on(shape, p, dev)
    }

    pub fn uniform<S: Into<Shape>, Var>(shape: S, lower: f32, upper: f32) -> Var
    where
        FuncImpls<Var>: Functions<Var>,
    {
        <FuncImpls<Var> as Functions<Var>>::random_uniform(shape, lower, upper)
    }

    pub fn uniform_on<S: Into<Shape>, D: Device, Var>(
        shape: S,
        lower: f32,
        upper: f32,
        dev: Option<&mut D>,
    ) -> Var
    where
        FuncImpls<Var>: Functions<Var>,
    {
        <FuncImpls<Var> as Functions<Var>>::random_uniform_on(shape, lower, upper, dev)
    }

    pub fn normal<S: Into<Shape>, Var>(shape: S, mean: f32, sd: f32) -> Var
    where
        FuncImpls<Var>: Functions<Var>,
    {
        <FuncImpls<Var> as Functions<Var>>::random_normal(shape, mean, sd)
    }

    pub fn normal_on<S: Into<Shape>, D: Device, Var>(
        shape: S,
        mean: f32,
        sd: f32,
        dev: Option<&mut D>,
    ) -> Var
    where
        FuncImpls<Var>: Functions<Var>,
    {
        <FuncImpls<Var> as Functions<Var>>::random_normal_on(shape, mean, sd, dev)
    }

    pub fn log_normal<S: Into<Shape>, Var>(shape: S, mean: f32, sd: f32) -> Var
    where
        FuncImpls<Var>: Functions<Var>,
    {
        <FuncImpls<Var> as Functions<Var>>::random_log_normal(shape, mean, sd)
    }

    pub fn log_normal_on<S: Into<Shape>, D: Device, Var>(
        shape: S,
        mean: f32,
        sd: f32,
        dev: Option<&mut D>,
    ) -> Var
    where
        FuncImpls<Var>: Functions<Var>,
    {
        <FuncImpls<Var> as Functions<Var>>::random_log_normal_on(shape, mean, sd, dev)
    }

    pub fn gumbel<S: Into<Shape>, Var>(shape: S, mu: f32, beta: f32) -> Var
    where
        FuncImpls<Var>: Functions<Var>,
    {
        <FuncImpls<Var> as Functions<Var>>::random_gumbel(shape, mu, beta)
    }

    pub fn gumbel_on<S: Into<Shape>, D: Device, Var>(
        shape: S,
        mu: f32,
        beta: f32,
        dev: Option<&mut D>,
    ) -> Var
    where
        FuncImpls<Var>: Functions<Var>,
    {
        <FuncImpls<Var> as Functions<Var>>::random_gumbel_on(shape, mu, beta, dev)
    }
}

pub mod batch {
    use super::FuncImpls;
    use super::Functions;

    pub fn sum<T: AsRef<Var>, Var>(x: T) -> Var
    where
        FuncImpls<Var>: Functions<Var>,
    {
        <FuncImpls<Var> as Functions<Var>>::batch_sum(x)
    }

    pub fn mean<T: AsRef<Var>, Var>(x: T) -> Var
    where
        FuncImpls<Var>: Functions<Var>,
    {
        <FuncImpls<Var> as Functions<Var>>::batch_mean(x)
    }

    pub fn normalize<T: AsRef<Var>, Var>(x: T) -> Var
    where
        FuncImpls<Var>: Functions<Var>,
    {
        <FuncImpls<Var> as Functions<Var>>::batch_normalize(x)
    }
}

impl Functions<Node> for FuncImpls<Node> {
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
    fn concat<T: AsRef<Node>>(xs: &[T], dim: u32) -> Node {
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
    fn sum<T: AsRef<Node>>(x: T, dim: u32) -> Node {
        node_funcs::sum(x, dim)
    }

    #[inline]
    fn sum_vars<T: AsRef<Node>>(xs: &[T]) -> Node {
        node_funcs::sum_nodes(xs)
    }

    #[inline]
    fn mean<T: AsRef<Node>>(x: T, dim: u32) -> Node {
        node_funcs::mean(x, dim)
    }

    #[inline]
    fn mean_vars<T: AsRef<Node>>(xs: &[T]) -> Node {
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
    fn concat<T: AsRef<Tensor>>(xs: &[T], dim: u32) -> Tensor {
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
    fn sum<T: AsRef<Tensor>>(x: T, dim: u32) -> Tensor {
        tensor_funcs::sum(x, dim)
    }

    #[inline]
    fn sum_vars<T: AsRef<Tensor>>(xs: &[T]) -> Tensor {
        tensor_funcs::sum_tensors(xs)
    }

    #[inline]
    fn mean<T: AsRef<Tensor>>(x: T, dim: u32) -> Tensor {
        tensor_funcs::mean(x, dim)
    }

    #[inline]
    fn mean_vars<T: AsRef<Tensor>>(xs: &[T]) -> Tensor {
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
