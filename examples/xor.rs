extern crate primitiv;

use primitiv::device;
use primitiv::Graph;
use primitiv::Optimizer;
use primitiv::Parameter;
use primitiv::Shape;

use primitiv::devices as D;
use primitiv::initializers as I;
use primitiv::operators as F;
use primitiv::optimizers as O;

fn main() {
    let mut dev = D::Naive::new();
    device::set_default(&mut dev);

    let mut pw1 = Parameter::from_initializer(&Shape::from_dims(&[8, 2], 1), &I::XavierUniform::new(1.0));
    let mut pb1 = Parameter::from_initializer(&Shape::from_dims(&[8], 1), &I::Constant::new(0.0));
    let mut pw2 = Parameter::from_initializer(&Shape::from_dims(&[1, 8], 1), &I::XavierUniform::new(1.0));
    let mut pb2 = Parameter::from_initializer(&Shape::from_dims(&[], 1), &I::Constant::new(0.0));

    let mut optimizer = O::SGD::new(0.1);

    optimizer.add_parameter(&mut pw1);
    optimizer.add_parameter(&mut pb1);
    optimizer.add_parameter(&mut pw2);
    optimizer.add_parameter(&mut pb2);

    let input_data = [
         1.,  1.,
         1., -1.,
        -1.,  1.,
        -1., -1.,
    ];
    let output_data = [
         1., -1.,
        -1.,  1.,
    ];

    let mut g = Graph::new();
    Graph::set_default(&mut g);

    for i in 0..10 {
        let x = F::input(&Shape::from_dims(&[2], 4), &input_data);
        let w1 = F::parameter(&mut pw1);
        let b1 = F::parameter(&mut pb1);
        let w2 = F::parameter(&mut pw2);
        let b2 = F::parameter(&mut pb2);
        let h = F::tanh(&(F::matmul(&w1, &x) + b1));
        let y = F::matmul(&w2, &h) + b2;

        let y_val = y.to_vector();

        // g.clear();
    }
}
