use primitiv_sys as _primitiv;
use std::ffi::CString;
use std::io;
use std::path::Path;
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
    pub fn load<P: AsRef<Path>>(&mut self, path: P, with_stats: bool) -> io::Result<()> {
        self.load_on::<P, AnyDevice>(path, with_stats, None)
    }

    /// Loads all parameters from a file.
    pub fn load_on<P: AsRef<Path>, D: Device>(
        &mut self,
        path: P,
        with_stats: bool,
        device: Option<&mut D>,
    ) -> io::Result<()> {
        unsafe {
            let path_c = CString::new(path.as_ref().to_str().unwrap()).unwrap();
            let path_ptr = path_c.as_ptr();
            Result::from_api_status(
                _primitiv::primitivLoadModel(
                    self.as_mut_ptr(),
                    path_ptr,
                    with_stats as u32,
                    device.map(|d| d.as_mut_ptr()).unwrap_or(ptr::null_mut()),
                ),
                (),
            ).map_err(|status| {
                io::Error::new(io::ErrorKind::Other, status.message())
            })
        }
    }

    /// Saves all parameters to a file.
    pub fn save<P: AsRef<Path>>(&self, path: P, with_stats: bool) -> io::Result<()> {
        unsafe {
            let path_c = CString::new(path.as_ref().to_str().unwrap()).unwrap();
            let path_ptr = path_c.as_ptr();
            Result::from_api_status(_primitiv::primitivSaveModel(
                self.as_ptr(),
                path_ptr,
                with_stats as u32,
                ),
                (),
            ).map_err(|status| {
                io::Error::new(io::ErrorKind::Other, status.message())
            })
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
    pub fn add_submodel<M: AsMut<Model>>(&mut self, name: &str, model: &mut M) {
        unsafe {
            let name_c = CString::new(name).unwrap();
            let name_ptr = name_c.as_ptr();
            check_api_status!(_primitiv::primitivAddSubmodelToModel(
                self.as_mut_ptr(),
                name_ptr,
                model.as_mut().as_mut_ptr(),
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

impl AsRef<Model> for Model {
    #[inline]
    fn as_ref(&self) -> &Model {
        self
    }
}

impl AsMut<Model> for Model {
    #[inline]
    fn as_mut(&mut self) -> &mut Model {
        self
    }
}

pub trait ModelImpl {
    fn load<P: AsRef<Path>>(&mut self, path: P, with_stats: bool) -> io::Result<()>;

    fn load_on<P: AsRef<Path>, D: Device>(
        &mut self,
        path: P,
        with_stats: bool,
        device: Option<&mut D>,
    ) -> io::Result<()>;

    fn save<P: AsRef<Path>>(&self, path: P, with_stats: bool) -> io::Result<()>;

    fn add_parameter(&mut self, name: &str, param: &mut Parameter);

    fn add_submodel<M: AsMut<Model>>(&mut self, name: &str, model: &mut M);
}

impl<T: AsRef<Model> + AsMut<Model>> ModelImpl for T {
    fn load<P: AsRef<Path>>(&mut self, path: P, with_stats: bool) -> io::Result<()> {
        Model::load(self.as_mut(), path, with_stats)
    }

    fn load_on<P: AsRef<Path>, D: Device>(
        &mut self,
        path: P,
        with_stats: bool,
        device: Option<&mut D>,
    ) -> io::Result<()> {
        Model::load_on(self.as_mut(), path, with_stats, device)
    }

    fn save<P: AsRef<Path>>(&self, path: P, with_stats: bool) -> io::Result<()> {
        Model::save(self.as_ref(), path, with_stats)
    }

    fn add_parameter(&mut self, name: &str, param: &mut Parameter) {
        Model::add_parameter(self.as_mut(), name, param)
    }

    fn add_submodel<M: AsMut<Model>>(&mut self, name: &str, model: &mut M) {
        Model::add_submodel(self.as_mut(), name, model.as_mut())
    }
}

#[macro_export]
macro_rules! impl_model {
    ($name:ident, $field:ident) => {
        impl AsRef<Model> for $name {
            #[inline]
            fn as_ref(&self) -> &Model {
                &self.$field
            }
        }

        impl AsMut<Model> for $name {
            #[inline]
            fn as_mut(&mut self) -> &mut Model {
                &mut self.$field
            }
        }
    }
}
