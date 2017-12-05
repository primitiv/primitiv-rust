extern crate primitiv_sys as _primitiv;

use Wrap;

pub trait Optimizer: Wrap<_primitiv::primitiv_Optimizer> {}
