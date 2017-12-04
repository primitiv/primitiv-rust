extern crate primitiv;

use primitiv::Device;
use primitiv::Graph;
// use primitiv::Parameter;

use primitiv::devices as D;
// use primitiv::initializers as I;
// use primitiv::operators as F;
// use primitiv::optimizers as O;

fn main() {
    let dev = D::Naive::new();
    <D::Naive as Device>::set_default(&dev);

    let g = Graph::new();
    Graph::set_default(&g);
}
