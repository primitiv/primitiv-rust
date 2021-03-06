use model_internal;
use primitiv_sys as _primitiv;
use std::ffi::CString;
use std::io;
use std::path::Path;
use ApiResult;
use Model;
use Parameter;
use Wrap;

/// `Optimizer` trait
pub trait Optimizer: Wrap<_primitiv::primitivOptimizer_t> + Default {
    /// Loads configurations from a file.
    fn load<P: AsRef<Path>>(&mut self, path: P) -> io::Result<()> {
        unsafe {
            let path_c = CString::new(path.as_ref().to_str().unwrap()).unwrap();
            let path_ptr = path_c.as_ptr();
            Result::from_api_status(
                _primitiv::primitivLoadOptimizer(self.as_mut_ptr(), path_ptr),
                (),
            ).map_err(|status| io::Error::new(io::ErrorKind::Other, status.message()))
        }
    }

    /// Saves current configurations to a file.
    fn save<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        unsafe {
            let path_c = CString::new(path.as_ref().to_str().unwrap()).unwrap();
            let path_ptr = path_c.as_ptr();
            Result::from_api_status(
                _primitiv::primitivSaveOptimizer(self.as_ptr(), path_ptr),
                (),
            ).map_err(|status| io::Error::new(io::ErrorKind::Other, status.message()))
        }
    }

    /// Retrieves current epoch.
    fn get_epoch(&self) -> u32 {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(_primitiv::primitivGetOptimizerEpoch(
                self.as_ptr(),
                &mut retval as *mut _,
            ));
            retval
        }
    }

    /// Sets current epoch.
    fn set_epoch(&mut self, epoch: u32) {
        unsafe {
            check_api_status!(_primitiv::primitivSetOptimizerEpoch(
                self.as_mut_ptr(),
                epoch,
            ));
        }
    }

    /// Retrieves current learning rate scaling factor.
    fn get_learning_rate_scaling(&self) -> f32 {
        unsafe {
            let mut retval: f32 = 0.0;
            check_api_status!(_primitiv::primitivGetOptimizerLearningRateScaling(
                self.as_ptr(),
                &mut retval as *mut _,
            ));
            retval
        }
    }

    /// Sets learning rate scaling factor.
    fn set_learning_rate_scaling(&mut self, scale: f32) {
        unsafe {
            check_api_status!(_primitiv::primitivSetOptimizerLearningRateScaling(
                self.as_mut_ptr(),
                scale,
            ));
        }
    }

    /// Retrieves current L2 decay strength.
    fn get_weight_decay(&self) -> f32 {
        unsafe {
            let mut retval: f32 = 0.0;
            check_api_status!(_primitiv::primitivGetOptimizerWeightDecay(
                self.as_ptr(),
                &mut retval as *mut _,
            ));
            retval
        }
    }

    /// Sets L2 decay strength.
    fn set_weight_decay(&mut self, strength: f32) {
        unsafe {
            check_api_status!(_primitiv::primitivSetOptimizerWeightDecay(
                self.as_mut_ptr(),
                strength,
            ));
        }
    }

    /// Retrieves current gradient clipping threshold.<Paste>
    fn get_gradient_clipping(&self) -> f32 {
        unsafe {
            let mut retval: f32 = 0.0;
            check_api_status!(_primitiv::primitivGetOptimizerGradientClipping(
                self.as_ptr(),
                &mut retval as *mut _,
            ));
            retval
        }
    }

    /// Sets gradient clipping threshold.
    fn set_gradient_clipping(&mut self, threshold: f32) {
        unsafe {
            check_api_status!(_primitiv::primitivSetOptimizerGradientClipping(
                self.as_mut_ptr(),
                threshold,
            ));
        }
    }

    /// Registers a parameter.
    fn add_parameter(&mut self, param: &mut Parameter) {
        unsafe {
            check_api_status!(_primitiv::primitivAddParameterToOptimizer(
                self.as_mut_ptr(),
                param.as_mut_ptr(),
            ));
        }
    }

    /// Registers multiple parameters.
    fn add_parameters(&mut self, params: &mut [Parameter]) {
        unsafe {
            let mut param_ptrs = params
                .iter_mut()
                .map(|param| param.as_mut_ptr())
                .collect::<Vec<_>>();
            check_api_status!(_primitiv::primitivAddParametersToOptimizer(
                self.as_mut_ptr(),
                param_ptrs.as_mut_ptr(),
                param_ptrs.len(),
            ));
        }
    }

    /// Registers a model.
    fn add_model<M: Model>(&mut self, model: &mut M) {
        unsafe {
            model.register_parameters();
            let lock = model_internal::get_entity_mut(model);
            let mut entity = lock.write().unwrap();
            check_api_status!(_primitiv::primitivAddModelToOptimizer(
                self.as_mut_ptr(),
                entity.as_mut_ptr(),
            ));
        }
    }

    /// Registers multiple models.
    fn add_models<M: Model>(&mut self, models: &mut [M]) {
        unsafe {
            let locks = models
                .iter_mut()
                .map(|model| {
                    model.register_parameters();
                    model_internal::get_entity_mut(model)
                })
                .collect::<Vec<_>>();
            let mut guards = locks
                .iter()
                .map(|lock| lock.write().unwrap())
                .collect::<Vec<_>>();
            let mut model_ptrs = guards
                .iter_mut()
                .map(|entity| entity.as_mut_ptr())
                .collect::<Vec<_>>();
            check_api_status!(_primitiv::primitivAddModelsToOptimizer(
                self.as_mut_ptr(),
                model_ptrs.as_mut_ptr(),
                model_ptrs.len(),
            ));
        }
    }

    /// Resets all gradients of registered parameters.
    fn reset_gradients(&mut self) {
        unsafe {
            check_api_status!(_primitiv::primitivResetOptimizerGradients(
                self.as_mut_ptr(),
            ));
        }
    }

    /// Updates parameter values.
    fn update(&mut self) {
        unsafe {
            check_api_status!(_primitiv::primitivExecuteOptimizerUpdate(self.as_mut_ptr()));
        }
    }

    /// Gets a configuration value.
    fn get_uint_config(&self, key: &str) -> u32 {
        unsafe {
            let mut retval: u32 = 0;
            let key_c = CString::new(key).unwrap();
            let key_ptr = key_c.as_ptr();
            check_api_status!(_primitiv::primitivGetOptimizerIntConfig(
                self.as_ptr(),
                key_ptr,
                &mut retval as *mut _,
            ));
            retval
        }
    }

    /// Sets a configuration value.
    fn set_uint_config(&mut self, key: &str, value: u32) {
        unsafe {
            let key_c = CString::new(key).unwrap();
            let key_ptr = key_c.as_ptr();
            check_api_status!(_primitiv::primitivSetOptimizerIntConfig(
                self.as_mut_ptr(),
                key_ptr,
                value,
            ));
        }
    }

    /// Gets a configuration value.
    fn get_float_config(&self, key: &str) -> f32 {
        unsafe {
            let mut retval: f32 = 0.0;
            let key_c = CString::new(key).unwrap();
            let key_ptr = key_c.as_ptr();
            check_api_status!(_primitiv::primitivGetOptimizerFloatConfig(
                self.as_ptr(),
                key_ptr,
                &mut retval as *mut _,
            ));
            retval
        }
    }

    /// Sets a configuration value.
    fn set_float_config(&mut self, key: &str, value: f32) {
        unsafe {
            let key_c = CString::new(key).unwrap();
            let key_ptr = key_c.as_ptr();
            check_api_status!(_primitiv::primitivSetOptimizerFloatConfig(
                self.as_mut_ptr(),
                key_ptr,
                value,
            ));
        }
    }
}

macro_rules! impl_optimizer {
    ($name:ident) => {
        impl_wrap_owned!($name, primitivOptimizer_t);
        impl_drop!($name, primitivDeleteOptimizer);
        impl Optimizer for $name {}
    };
}
