extern crate primitiv;

use primitiv::Device;
use primitiv::devices;
use primitiv::Graph;
use primitiv::node_functions as F;

struct NodeTest<D> {
    g: Graph,
    dev: D,
}

impl<D: Device> NodeTest<D> {
    pub fn new(dev: D) -> Self {
        NodeTest {
            g: Graph::new(),
            dev: dev,
        }
    }

    pub fn setup(&mut self) {
        Graph::set_default(&mut self.g);
        devices::set_default(&mut self.dev);
    }
}

#[test]
fn test_clone() {
    let mut case = NodeTest::new(devices::Naive::new());
    case.setup();
    let data = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
    let a = F::input(([2, 2], 3), &data);
    assert!(a.valid());
    let b = a.clone();
    assert!(a.valid());
    assert!(b.valid());
    let s1 = a.shape();
    let s2 = b.shape();
    assert_eq!(s1, s2);
    let c = F::concat(&[a, b], 0);
    assert!(c.valid());
}
