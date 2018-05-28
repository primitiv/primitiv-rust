use primitiv_sys as _primitiv;
use std::ptr;
use ApiResult;
use Optimizer;
use Wrap;

/// Simple stochastic gradient descent.
#[derive(Debug)]
pub struct SGD {
    inner: *mut _primitiv::primitivOptimizer_t,
}

impl_optimizer!(SGD);

impl SGD {
    /// Creates a new SGD object.
    pub fn new(eta: f32) -> Self {
        unsafe {
            let mut optimizer_ptr: *mut _primitiv::primitivOptimizer_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCreateSgdOptimizer(
                eta,
                &mut optimizer_ptr,
            ));
            assert!(!optimizer_ptr.is_null());
            SGD {
                inner: optimizer_ptr,
            }
        }
    }
}

impl Default for SGD {
    fn default() -> Self {
        Self::new(0.1)
    }
}

/// Stochastic gradient descent with momentum.
#[derive(Debug)]
pub struct MomentumSGD {
    inner: *mut _primitiv::primitivOptimizer_t,
}

impl_optimizer!(MomentumSGD);

impl MomentumSGD {
    /// Creates a new MomentumSGD object.
    pub fn new(eta: f32, momentum: f32) -> Self {
        unsafe {
            let mut optimizer_ptr: *mut _primitiv::primitivOptimizer_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCreateMomentumSgdOptimizer(
                eta,
                momentum,
                &mut optimizer_ptr,
            ));
            assert!(!optimizer_ptr.is_null());
            MomentumSGD {
                inner: optimizer_ptr,
            }
        }
    }
}

impl Default for MomentumSGD {
    fn default() -> Self {
        Self::new(0.01, 0.9)
    }
}

/// AdaGrad optimizer.
#[derive(Debug)]
pub struct AdaGrad {
    inner: *mut _primitiv::primitivOptimizer_t,
}

impl_optimizer!(AdaGrad);

impl AdaGrad {
    /// Creates a new AdaGrad object.
    pub fn new(eta: f32, eps: f32) -> Self {
        unsafe {
            let mut optimizer_ptr: *mut _primitiv::primitivOptimizer_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCreateAdaGradOptimizer(
                eta,
                eps,
                &mut optimizer_ptr,
            ));
            assert!(!optimizer_ptr.is_null());
            AdaGrad {
                inner: optimizer_ptr,
            }
        }
    }
}

impl Default for AdaGrad {
    fn default() -> Self {
        Self::new(0.001, 1e-8)
    }
}

/// RMSProp optimizer.
#[derive(Debug)]
pub struct RMSProp {
    inner: *mut _primitiv::primitivOptimizer_t,
}

impl_optimizer!(RMSProp);

impl RMSProp {
    /// Creates a new RMSProp object.
    pub fn new(eta: f32, alpha: f32, eps: f32) -> Self {
        unsafe {
            let mut optimizer_ptr: *mut _primitiv::primitivOptimizer_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCreateRmsPropOptimizer(
                eta,
                alpha,
                eps,
                &mut optimizer_ptr,
            ));
            assert!(!optimizer_ptr.is_null());
            RMSProp {
                inner: optimizer_ptr,
            }
        }
    }
}

impl Default for RMSProp {
    fn default() -> Self {
        Self::new(0.01, 0.9, 1e-8)
    }
}

/// AdaDelta optimizer.
#[derive(Debug)]
pub struct AdaDelta {
    inner: *mut _primitiv::primitivOptimizer_t,
}

impl_optimizer!(AdaDelta);

impl AdaDelta {
    /// Creates a new AdaDelta object.
    pub fn new(rho: f32, eps: f32) -> Self {
        unsafe {
            let mut optimizer_ptr: *mut _primitiv::primitivOptimizer_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCreateAdaDeltaOptimizer(
                rho,
                eps,
                &mut optimizer_ptr,
            ));
            assert!(!optimizer_ptr.is_null());
            AdaDelta {
                inner: optimizer_ptr,
            }
        }
    }
}

impl Default for AdaDelta {
    fn default() -> Self {
        Self::new(0.95, 1e-6)
    }
}

/// Adam optimizer.
#[derive(Debug)]
pub struct Adam {
    inner: *mut _primitiv::primitivOptimizer_t,
}

impl_optimizer!(Adam);

impl Adam {
    /// Creates a new Adam object.
    pub fn new(alpha: f32, beta1: f32, beta2: f32, eps: f32) -> Self {
        unsafe {
            let mut optimizer_ptr: *mut _primitiv::primitivOptimizer_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCreateAdamOptimizer(
                alpha,
                beta1,
                beta2,
                eps,
                &mut optimizer_ptr,
            ));
            assert!(!optimizer_ptr.is_null());
            Adam {
                inner: optimizer_ptr,
            }
        }
    }
}

impl Default for Adam {
    fn default() -> Self {
        Self::new(0.001, 0.9, 0.999, 1e-8)
    }
}
