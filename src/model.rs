use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;
use std::io;
use std::path::Path;
use Device;
use Parameter;
use Wrap;

pub trait Model: Sized {
    /// Loads all parameters from a file.
    fn load<P: AsRef<Path>>(&mut self, path: P, with_stats: bool) -> io::Result<()> {
        self.register_parameters();
        let lock = internal::get_entity_mut(self);
        let mut entity = lock.write().unwrap();
        entity.load(path, with_stats)
    }

    /// Loads all parameters from a file.
    fn load_on<P: AsRef<Path>, D: Device>(
        &mut self,
        path: P,
        with_stats: bool,
        device: Option<&mut D>,
    ) -> io::Result<()> {
        self.register_parameters();
        let lock = internal::get_entity_mut(self);
        let mut entity = lock.write().unwrap();
        entity.load_on(path, with_stats, device)
    }

    /// Saves all parameters to a file.
    fn save<P: AsRef<Path>>(&self, path: P, with_stats: bool) -> io::Result<()> {
        let lock = internal::get_entity(self);
        let entity = lock.read().unwrap();
        entity.save(path, with_stats)
    }

    /// Registers a new parameter.
    fn add_parameter(&mut self, name: &str, param: &mut Parameter) {
        let lock = internal::get_entity_mut(self);
        let mut entity = lock.write().unwrap();
        entity.add_parameter(name, param)
    }

    /// Registers a new submodel.
    fn add_submodel<M: Model>(&mut self, name: &str, model: &mut M) {
        let mut entity_other = {
            let lock_other = internal::get_entity_mut(model);
            let mut entity_other = lock_other.write().unwrap();
            internal::ModelEntity::from_raw(entity_other.as_mut_ptr(), false)
        };
        let lock_self = internal::get_entity_mut(self);
        let mut entity_self = lock_self.write().unwrap();
        entity_self.add_submodel(name, &mut entity_other)
    }

    /// Retrieves a parameter with specified name.
    fn get_parameter(&self, name: &str) -> Option<Parameter> {
        let lock = internal::get_entity(self);
        let entity = lock.read().unwrap();
        entity.get_parameter(name)
    }

    /// Recursively searches a parameter with specified name hierarchy.
    fn find_parameter(&self, names: &[&str]) -> Option<Parameter> {
        let lock = internal::get_entity(self);
        let entity = lock.read().unwrap();
        entity.find_parameter(names)
    }

    /// Retrieves a submodel with specified name.
    fn get_submodel(&self, name: &str) -> Option<AnyModel> {
        {
            let lock = internal::get_entity(self);
            let entity = lock.read().unwrap();
            entity.get_submodel(name)
        }.map(|sub_entity| {
            let mut model = AnyModel;
            internal::add_entity(&mut model, sub_entity);
            model
        })
    }

    /// Recursively searches a submodel with specified name hierarchy.
    fn find_submodel(&self, names: &[&str]) -> Option<AnyModel> {
        {
            let lock = internal::get_entity(self);
            let entity = lock.read().unwrap();
            entity.find_submodel(names)
        }.map(|sub_entity| {
            let mut model = AnyModel;
            internal::add_entity(&mut model, sub_entity);
            model
        })
    }

    fn register_parameters(&mut self);

    fn identifier(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        hasher.write(format!("{:p}", self).as_bytes());
        hasher.finish()
    }

    fn invalidate(&mut self) {
        internal::remove_entity(self);
    }
}

#[derive(Debug)]
pub struct AnyModel;

impl Model for AnyModel {
    fn register_parameters(&mut self) {}
}

impl Drop for AnyModel {
    fn drop(&mut self) {
        self.invalidate();
    }
}

pub(crate) mod internal {
    use super::Model;
    use devices::AnyDevice;
    use primitiv_sys as _primitiv;
    use std::collections::HashMap;
    use std::ffi::CString;
    use std::io;
    use std::path::Path;
    use std::ptr;
    use std::sync::{self, Arc, RwLock};
    use ApiResult;
    use Device;
    use Parameter;
    use Wrap;

    lazy_static! {
        static ref MODEL_MAP: RwLock<HashMap<u64, Arc<RwLock<ModelEntity>>>> =
            RwLock::new(HashMap::new());
    }

    pub(crate) struct UnwritableLock<T>(Arc<RwLock<T>>);

    impl<T> UnwritableLock<T> {
        pub fn read(&self) -> sync::LockResult<sync::RwLockReadGuard<T>> {
            self.0.read()
        }

        #[allow(unused)]
        pub fn try_read(&self) -> sync::TryLockResult<sync::RwLockReadGuard<T>> {
            self.0.try_read()
        }
    }

    pub(crate) struct WritableLock<T>(Arc<RwLock<T>>);

    impl<T> WritableLock<T> {
        #[allow(unused)]
        pub fn read(&self) -> sync::LockResult<sync::RwLockReadGuard<T>> {
            self.0.read()
        }

        #[allow(unused)]
        pub fn try_read(&self) -> sync::TryLockResult<sync::RwLockReadGuard<T>> {
            self.0.try_read()
        }

        pub fn write(&self) -> sync::LockResult<sync::RwLockWriteGuard<T>> {
            self.0.write()
        }

        #[allow(unused)]
        pub fn try_write(&self) -> sync::TryLockResult<sync::RwLockWriteGuard<T>> {
            self.0.try_write()
        }
    }

    pub(crate) fn add_entity<M: Model + Sized>(model: &mut M, entity: ModelEntity) {
        MODEL_MAP
            .write()
            .unwrap()
            .insert(model.identifier(), Arc::new(RwLock::new(entity)));
    }

    pub(crate) fn remove_entity<M: Model + Sized>(model: &mut M) {
        MODEL_MAP.write().unwrap().remove(&model.identifier());
    }

    pub(crate) fn get_entity<M: Model + Sized>(model: &M) -> UnwritableLock<ModelEntity> {
        let key = model.identifier();
        let entity = {
            let map = MODEL_MAP.read().unwrap();
            map.get(&key).map(|e| e.clone())
        }.unwrap_or_else(|| {
            let mut map = MODEL_MAP.write().unwrap();
            let e = Arc::new(RwLock::new(ModelEntity::new()));
            map.insert(key, e.clone());
            e
        });
        UnwritableLock(entity)
    }

    pub(crate) fn get_entity_mut<M: Model + Sized>(model: &mut M) -> WritableLock<ModelEntity> {
        let mut map = MODEL_MAP.write().unwrap();
        let entity = map
            .entry(model.identifier())
            .or_insert_with(|| Arc::new(RwLock::new(ModelEntity::new())))
            .clone();
        WritableLock(entity)
    }

    /// Set of parameters and specific algorithms.
    #[derive(Debug)]
    pub(crate) struct ModelEntity {
        inner: *mut _primitiv::primitivModel_t,
        owned: bool,
    }

    impl_wrap!(ModelEntity, primitivModel_t);
    impl_drop!(ModelEntity, primitivDeleteModel);

    unsafe impl Send for ModelEntity {}
    unsafe impl Sync for ModelEntity {}

    impl ModelEntity {
        /// Creates a new ModelEntity object.
        pub fn new() -> Self {
            unsafe {
                let mut model_ptr: *mut _primitiv::primitivModel_t = ptr::null_mut();
                check_api_status!(_primitiv::primitivCreateModel(&mut model_ptr));
                assert!(!model_ptr.is_null());
                ModelEntity {
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
                ).map_err(|status| io::Error::new(io::ErrorKind::Other, status.message()))
            }
        }

        /// Saves all parameters to a file.
        pub fn save<P: AsRef<Path>>(&self, path: P, with_stats: bool) -> io::Result<()> {
            unsafe {
                let path_c = CString::new(path.as_ref().to_str().unwrap()).unwrap();
                let path_ptr = path_c.as_ptr();
                Result::from_api_status(
                    _primitiv::primitivSaveModel(self.as_ptr(), path_ptr, with_stats as u32),
                    (),
                ).map_err(|status| io::Error::new(io::ErrorKind::Other, status.message()))
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
        pub fn add_submodel(&mut self, name: &str, model: &mut ModelEntity) {
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
        pub fn get_parameter(&self, name: &str) -> Option<Parameter> {
            self.find_parameter(&[name; 1])
        }

        /// Recursively searches a parameter with specified name hierarchy.
        pub fn find_parameter(&self, names: &[&str]) -> Option<Parameter> {
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
                let result = Result::from_api_status(
                    _primitiv::primitivGetParameterFromModel(
                        self.as_ptr(),
                        names_ptr_vec.as_mut_ptr(),
                        names_ptr_vec.len(),
                        &mut parameter_ptr,
                    ),
                    (),
                ).map(|()| {
                    assert!(!parameter_ptr.is_null());
                    Parameter::from_raw(parameter_ptr as *mut _, false)
                });
                result.ok()
            }
        }

        /// Retrieves a submodel with specified name.
        pub fn get_submodel(&self, name: &str) -> Option<ModelEntity> {
            self.find_submodel(&[name; 1])
        }

        /// Recursively searches a submodel with specified name hierarchy.
        pub fn find_submodel(&self, names: &[&str]) -> Option<ModelEntity> {
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
                let result = Result::from_api_status(
                    _primitiv::primitivGetSubmodelFromModel(
                        self.as_ptr(),
                        names_ptr_vec.as_mut_ptr(),
                        names_ptr_vec.len(),
                        &mut model_ptr,
                    ),
                    (),
                ).map(|()| {
                    assert!(!model_ptr.is_null());
                    ModelEntity::from_raw(model_ptr as *mut _, false)
                });
                result.ok()
            }
        }

        // TODO(chantera): Implement get_all_parameters().
        // TODO(chantera): Implement get_trainable_parameters().
    }

    impl Default for ModelEntity {
        fn default() -> ModelEntity {
            ModelEntity::new()
        }
    }
}
