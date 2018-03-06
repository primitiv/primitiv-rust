use primitiv::Model;
use primitiv::Node;
use primitiv::Parameter;
use primitiv::initializers as I;
use primitiv::functions as F;

/// Hand-written LSTM with input/forget/output gates and no peepholes.
/// Formulation:
///   i = sigmoid(W_xi . x[t] + W_hi . h[t-1] + b_i)
///   f = sigmoid(W_xf . x[t] + W_hf . h[t-1] + b_f)
///   o = sigmoid(W_xo . x[t] + W_ho . h[t-1] + b_o)
///   j = tanh   (W_xj . x[t] + W_hj . h[t-1] + b_j)
///   c[t] = i * j + f * c[t-1]
///   h[t] = o * tanh(c[t])
pub struct LSTM {
    model: Model,
    pw: Parameter,
    pb: Parameter,
    w: Node,
    b: Node,
    h: Node,
    c: Node,
}

impl LSTM {
    pub fn new() -> Self {
        let mut m = LSTM {
            model: Model::new(),
            pw: Parameter::new(),
            pb: Parameter::new(),
            w: Node::new(),
            b: Node::new(),
            h: Node::new(),
            c: Node::new(),
        };
        m.model.add_parameter("w", &mut m.pw);
        m.model.add_parameter("b", &mut m.pb);
        m
    }

    /// Initializes the model.
    pub fn init(&mut self, in_size: u32, out_size: u32) {
        self.pw.init_by_initializer(
            [4 * out_size, in_size + out_size],
            &I::Uniform::new(-0.1, 0.1),
        );
        self.pb.init_by_initializer(
            [4 * out_size],
            &I::Constant::new(1.0),
        );
    }

    /// Initializes the model.
    pub fn restart(&mut self, init_c: Option<&Node>, init_h: Option<&Node>) {
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
    pub fn forward<N: AsRef<Node>>(&mut self, x: N) -> &Node {
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

    pub fn get_c(&self) -> &Node {
        &self.c
    }

    pub fn get_h(&self) -> &Node {
        &self.h
    }
}

impl AsRef<Model> for LSTM {
    #[inline]
    fn as_ref(&self) -> &Model {
        &self.model
    }
}

impl AsMut<Model> for LSTM {
    #[inline]
    fn as_mut(&mut self) -> &mut Model {
        &mut self.model
    }
}