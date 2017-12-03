extern crate primitiv;

use primitiv::Device;
// use primitiv::Graph;
// use primitiv::Parameter;

use primitiv::devices as D;
// use primitiv::initializers as I;
// use primitiv::operators as F;
// use primitiv::optimizers as O;

fn main() {
    let dev = D::Naive::new();
    D::Naive::set_default(&dev);
}
