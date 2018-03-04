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
    pwxh: Parameter,
    pwhh: Parameter,
    pbh: Parameter,
    wxh: Node,
    whh: Node,
    bh: Node,
    h: Node,
    c: Node,
}

impl LSTM {
    pub fn new() -> Self {
        let mut m = LSTM {
            model: Model::new(),
            pwxh: Parameter::new(),
            pwhh: Parameter::new(),
            pbh: Parameter::new(),
            wxh: Node::new(),
            whh: Node::new(),
            bh: Node::new(),
            h: Node::new(),
            c: Node::new(),
        };
        m.model.add_parameter("pwxh", &mut m.pwxh);
        m.model.add_parameter("pwhh", &mut m.pwhh);
        m.model.add_parameter("pbh", &mut m.pbh);
        m
    }

    /// Initializes the model.
    pub fn init(&mut self, in_size: u32, out_size: u32) {
        self.pwxh.init_by_initializer(
            [4 * out_size, in_size],
            &I::XavierUniform::new(1.0),
        );
        self.pwhh.init_by_initializer(
            [4 * out_size, out_size],
            &I::XavierUniform::new(1.0),
        );
        self.pbh.init_by_initializer(
            [4 * out_size],
            &I::Constant::new(1.0),
        );
    }

    /// Initializes the model.
    pub fn restart(&mut self, init_c: Option<&Node>, init_h: Option<&Node>) {
        let out_size = self.pwhh.shape().at(1);
        self.wxh = F::parameter(&mut self.pwxh);
        self.whh = F::parameter(&mut self.pwhh);
        self.bh = F::parameter(&mut self.pbh);
        self.c = init_c
            .and_then(|n| if n.valid() { Some(n.clone()) } else { None })
            .unwrap_or(F::zeros([out_size]));
        self.h = init_h
            .and_then(|n| if n.valid() { Some(n.clone()) } else { None })
            .unwrap_or(F::zeros([out_size]));
    }

    /// One step forwarding.
    pub fn forward<N: AsRef<Node>>(&mut self, x: N) -> &Node {
        let out_size = self.pwhh.shape().at(1);
        let u = F::matmul(&self.wxh, x.as_ref()) + F::matmul(&self.whh, &self.h) + &self.bh;
        let i = F::sigmoid(F::slice(&u, 0, 0, out_size));
        let f = F::sigmoid(F::slice(&u, 0, out_size, 2 * out_size));
        let o = F::sigmoid(F::slice(&u, 0, 2 * out_size, 3 * out_size));
        let j = F::tanh(F::slice(&u, 0, 3 * out_size, 4 * out_size));
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
