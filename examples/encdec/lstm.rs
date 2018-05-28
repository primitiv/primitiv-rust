use primitiv::Parameter;
use primitiv::Variable;
use primitiv::functions as F;
use primitiv::initializers as I;

/// Hand-written LSTM with input/forget/output gates and no peepholes.
/// Formulation:
///   i = sigmoid(W_xi . x[t] + W_hi . h[t-1] + b_i)
///   f = sigmoid(W_xf . x[t] + W_hf . h[t-1] + b_f)
///   o = sigmoid(W_xo . x[t] + W_ho . h[t-1] + b_o)
///   j = tanh   (W_xj . x[t] + W_hj . h[t-1] + b_j)
///   c[t] = i * j + f * c[t-1]
///   h[t] = o * tanh(c[t])
#[derive(Model)]
pub struct LSTM<V: Variable> {
    pw: Parameter,
    pb: Parameter,
    w: V,
    b: V,
    h: V,
    c: V,
}

impl<V: Variable> LSTM<V> {
    pub fn new() -> Self {
        LSTM {
            pw: Parameter::new(),
            pb: Parameter::new(),
            w: V::new(),
            b: V::new(),
            h: V::new(),
            c: V::new(),
        }
    }

    /// Initializes the model.
    pub fn init(&mut self, in_size: u32, out_size: u32) {
        self.pw.init_by_initializer(
            [4 * out_size, in_size + out_size],
            &I::Uniform::new(-0.1, 0.1),
        );
        self.pb
            .init_by_initializer([4 * out_size], &I::Constant::new(1.0));
    }

    /// Initializes the model.
    pub fn restart(&mut self, init_c: Option<&V>, init_h: Option<&V>) {
        let out_size = self.pw.shape().at(0) / 4;
        self.w = F::parameter(&mut self.pw);
        self.b = F::parameter(&mut self.pb);
        self.c = init_c
            .and_then(|n| if n.valid() { Some(n.clone()) } else { None })
            .unwrap_or(F::zeros([out_size]));
        self.h = init_h
            .and_then(|n| if n.valid() { Some(n.clone()) } else { None })
            .unwrap_or(F::zeros([out_size]));
    }

    /// One step forwarding.
    pub fn forward<N: AsRef<V>>(&mut self, x: N) -> &V {
        let u = F::matmul(&self.w, F::concat(&vec![x.as_ref(), &self.h], 0)) + &self.b;
        let v = F::split(u, 0, 4);
        let i = F::sigmoid(&v[0]);
        let f = F::sigmoid(&v[1]);
        let o = F::sigmoid(&v[2]);
        let j = F::tanh(&v[3]);
        self.c = i * j + f * &self.c;
        self.h = o * F::tanh(&self.c);
        &self.h
    }

    pub fn get_c(&self) -> &V {
        &self.c
    }

    pub fn get_h(&self) -> &V {
        &self.h
    }
}
