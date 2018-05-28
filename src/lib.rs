extern crate primitiv_sys;
#[allow(unused_imports)]
#[macro_use]
extern crate primitiv_derive;
#[macro_use]
extern crate lazy_static;
extern crate libc;

#[doc(hidden)]
pub use primitiv_derive::*;

#[macro_use]
mod status;
pub(crate) use status::*;

#[macro_use]
mod util;
pub use util::*;
#[macro_use]
mod device;
pub use device::Device;
mod graph;
pub use graph::{Graph, Node};
#[macro_use]
mod initializer;
pub use initializer::Initializer;
#[macro_use]
mod model;
pub(crate) use model::internal as model_internal;
pub use model::Model;
mod parameter;
pub use parameter::Parameter;
mod shape;
pub use shape::Shape;
mod tensor;
pub use tensor::Tensor;
#[macro_use]
mod optimizer;
pub use optimizer::Optimizer;
pub mod functions;
pub use functions::node_funcs as node_functions;
pub use functions::tensor_funcs as tensor_functions;
pub use functions::Variable;
pub mod devices;
pub mod initializers;
pub mod optimizers;
