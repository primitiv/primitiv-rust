extern crate primitiv;

use primitiv::Graph;
use primitiv::Optimizer;
use primitiv::Parameter;

use primitiv::devices as D;
use primitiv::functions as F;
use primitiv::initializers as I;
use primitiv::optimizers as O;

fn main() {
    let mut dev = D::Naive::new();
    D::set_default(&mut dev);

    let mut pw1 = Parameter::from_initializer([8, 2], &I::XavierUniform::new(1.0));
    let mut pb1 = Parameter::from_initializer([8], &I::Constant::new(0.0));
    let mut pw2 = Parameter::from_initializer([1, 8], &I::XavierUniform::new(1.0));
    let mut pb2 = Parameter::from_initializer([], &I::Constant::new(0.0));

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
        g.clear();

        let x = F::input(([2], 4), &input_data);
        let w1 = F::parameter(&mut pw1);
        let b1 = F::parameter(&mut pb1);
        let w2 = F::parameter(&mut pw2);
        let b2 = F::parameter(&mut pb2);
        let h = F::tanh(F::matmul(w1, x) + b1);
        let y = F::matmul(w2, h) + b2;

        let y_val = y.to_vector();
        println!("epoch {}:", i);
        for j in 0..4 {
            println!("  [{}]: {}", j, y_val[j]);
        }

        let t = F::input(([], 4), &output_data);
        let diff = t - y;
        let loss = F::batch::mean(&diff * &diff);

        let loss_val = loss.to_float();
        println!("  loss: {}", loss_val);

        optimizer.reset_gradients();
        loss.backward();
        optimizer.update();
    }
}
