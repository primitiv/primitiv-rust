pub trait Wrap<T>: Drop {
    fn from_raw(ptr: *mut T, owned: bool) -> Self
    where
        Self: Sized;
    fn as_ptr(&self) -> *const T;
    fn as_mut_ptr(&mut self) -> *mut T;
    fn is_owned(&self) -> bool;
}

macro_rules! impl_wrap {
    ($name:ident, $type:ident) => {
        impl Wrap<_primitiv::$type> for $name {
            #[inline(always)]
            fn from_raw(ptr: *mut _primitiv::$type, owned: bool) -> Self {
                assert!(!ptr.is_null());
                $name { inner: ptr, owned: owned }
            }

            #[inline(always)]
            fn as_ptr(&self) -> *const _primitiv::$type {
                self.inner
            }

            #[inline(always)]
            fn as_mut_ptr(&mut self) -> *mut _primitiv::$type {
                self.inner
            }

            #[inline(always)]
            fn is_owned(&self) -> bool {
                self.owned
            }
        }
    }
}

macro_rules! impl_wrap_owned {
    ($name:ident, $type:ident) => {
        impl Wrap<_primitiv::$type> for $name {
            #[inline(always)]
            fn from_raw(ptr: *mut _primitiv::$type, _owned: bool) -> Self {
                assert!(!ptr.is_null());
                $name { inner: ptr }
            }

            #[inline(always)]
            fn as_ptr(&self) -> *const _primitiv::$type {
                self.inner
            }

            #[inline(always)]
            fn as_mut_ptr(&mut self) -> *mut _primitiv::$type {
                self.inner
            }

            #[inline(always)]
            fn is_owned(&self) -> bool {
                true
            }
        }
    }
}

macro_rules! impl_drop {
    ($name:ident, $call:ident) => {
        impl Drop for $name {
            fn drop(&mut self) {
                if self.is_owned() {
                    unsafe {
                        check_api_status!(_primitiv::$call(self.inner));
                    }
                }
            }
        }
    }
}
