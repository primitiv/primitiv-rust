use primitiv_sys as _primitiv;
use std::ffi::CString;
use std::ptr;
use ApiResult;
use Device;
use devices::AnyDevice;
use Parameter;
use Wrap;

/// Set of parameters and specific algorithms.
#[derive(Debug)]
pub struct Model {
    inner: *mut _primitiv::primitivModel_t,
    owned: bool,
}

impl_wrap!(Model, primitivModel_t);
impl_drop!(Model, primitivDeleteModel);

impl Model {
    /// Creates a new Model object.
    pub fn new() -> Self {
        unsafe {
            let mut model_ptr: *mut _primitiv::primitivModel_t = ptr::null_mut();
            check_api_status!(_primitiv::primitivCreateModel(&mut model_ptr));
            assert!(!model_ptr.is_null());
            Model {
                inner: model_ptr,
                owned: true,
            }
        }
    }

    /// Loads all parameters from a file.
    pub fn load(&mut self, path: &str, with_stats: bool) {
        self.load_into::<AnyDevice>(path, with_stats, None);
    }

    /// Loads all parameters from a file.
    pub fn load_into<D: Device>(&mut self, path: &str, with_stats: bool, device: Option<&mut D>) {
        unsafe {
            let path_c = CString::new(path).unwrap();
            let path_ptr = path_c.as_ptr();
            check_api_status!(_primitiv::primitivLoadModel(
                self.as_mut_ptr(),
                path_ptr,
                with_stats as u32,
                device.map(|d| d.as_mut_ptr()).unwrap_or(ptr::null_mut()),
            ));
        }
    }

    /// Saves all parameters to a file.
    pub fn save(&self, path: &str, with_stats: bool) {
        unsafe {
            let path_c = CString::new(path).unwrap();
            let path_ptr = path_c.as_ptr();
            check_api_status!(_primitiv::primitivSaveModel(
                self.as_ptr(),
                path_ptr,
                with_stats as u32,
            ));
        }
    }

    /// Registers a new parameter.
    pub fn add_parameter(&mut self, name: &str, param: &mut Parameter) {
        unsafe {
            let name_c = CString::new(name).unwrap();
            let name_ptr = name_c.as_ptr();
            check_api_status!(_primitiv::primitivAddParameterToModel(
                self.as_mut_ptr(),
                name_ptr,
                param.as_mut_ptr(),
            ));
        }
    }

    /// Registers a new submodel.
    pub fn add_submodel(&mut self, name: &str, model: &mut Model) {
        unsafe {
            let name_c = CString::new(name).unwrap();
            let name_ptr = name_c.as_ptr();
            check_api_status!(_primitiv::primitivAddSubmodelToModel(
                self.as_mut_ptr(),
                name_ptr,
                model.as_mut_ptr(),
            ));
        }
    }

    /// Retrieves a parameter with specified name.
    pub fn get_parameter(&mut self, name: &str) -> Parameter {
        self.find_parameter(&[name; 1])
    }

    /// Recursively searches a parameter with specified name hierarchy.
    pub fn find_parameter(&mut self, names: &[&str]) -> Parameter {
        unsafe {
            let mut parameter_ptr: *const _primitiv::primitivParameter_t = ptr::null_mut();
            let names_c_vec = names
                .iter()
                .map(|name| CString::new(*name).unwrap())
                .collect::<Vec<_>>();
            let mut names_ptr_vec = names_c_vec
                .iter()
                .map(|name_c| name_c.as_ptr())
                .collect::<Vec<_>>();
            check_api_status!(_primitiv::primitivGetParameterFromModel(
                self.as_ptr(),
                names_ptr_vec.as_mut_ptr(),
                names_ptr_vec.len(),
                &mut parameter_ptr,
            ));
            assert!(!parameter_ptr.is_null());
            Parameter::from_raw(parameter_ptr as *mut _, false)
        }
    }

    /// Retrieves a submodel with specified name.
    pub fn get_submodel(&mut self, name: &str) -> Model {
        self.find_submodel(&[name; 1])
    }

    /// Recursively searches a submodel with specified name hierarchy.
    pub fn find_submodel(&mut self, names: &[&str]) -> Model {
        unsafe {
            let mut model_ptr: *const _primitiv::primitivModel_t = ptr::null_mut();
            let names_c_vec = names
                .iter()
                .map(|name| CString::new(*name).unwrap())
                .collect::<Vec<_>>();
            let mut names_ptr_vec = names_c_vec
                .iter()
                .map(|name_c| name_c.as_ptr())
                .collect::<Vec<_>>();
            check_api_status!(_primitiv::primitivGetSubmodelFromModel(
                self.as_ptr(),
                names_ptr_vec.as_mut_ptr(),
                names_ptr_vec.len(),
                &mut model_ptr,
            ));
            assert!(!model_ptr.is_null());
            Model::from_raw(model_ptr as *mut _, false)
        }
    }

    // TODO: implement get_all_parameters()
    // TODO: implement get_trainable_parameters()
}
