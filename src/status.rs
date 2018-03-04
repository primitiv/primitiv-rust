extern crate backtrace;
use primitiv_sys as _primitiv;
use libc::c_uint;
use self::backtrace::Backtrace;
use std::env;
use std::ffi::CString;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::mem;
use std::ptr;
use std::result;

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Copy, Clone)]
pub(crate) enum Code {
    Ok,
    Error,
    UnrecognizedEnumValue(c_uint),
}

impl Code {
    fn from_int(value: c_uint) -> Self {
        match value {
            0 => Code::Ok,
            4294967295 => Code::Error,
            c => Code::UnrecognizedEnumValue(c),
        }
    }

    fn to_int(&self) -> c_uint {
        match self {
            &Code::UnrecognizedEnumValue(c) => c, 
            &Code::Ok => 0,
            &Code::Error => 4294967295,
        }
    }

    #[allow(dead_code)]
    fn to_c(&self) -> _primitiv::PRIMITIV_C_STATUS {
        unsafe { mem::transmute(self.to_int()) }
    }

    #[allow(dead_code)]
    fn from_c(value: _primitiv::PRIMITIV_C_STATUS) -> Self {
        Self::from_int(value as c_uint)
    }

    fn is_ok(value: c_uint) -> bool {
        match value {
            0 => true,
            _ => false,
        }
    }
}

impl fmt::Display for Code {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            &Code::Ok => f.write_str("Ok"),
            &Code::Error => f.write_str("Error"),
            &Code::UnrecognizedEnumValue(c) => write!(f, "UnrecognizedEnumValue({})", c),
        }
    }
}

pub(crate) struct Status {
    code: Code,
    message: String,
    trace: Option<Backtrace>,
}

impl Status {
    fn new(code: Code, message: String, trace: Option<Backtrace>) -> Self {
        Status {
            code: code,
            message: message,
            trace: trace,
        }
    }

    fn code(&self) -> Code {
        self.code
    }

    fn message(&self) -> &str {
        self.message.as_str()
    }
}

impl Display for Status {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        try!(write!(
            f,
            "Code: \"{}({})\", Message: \"{}\"\n",
            self.code(),
            self.code().to_int(),
            self.message()
        ));
        Ok(())
    }
}

impl Debug for Status {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        try!(write!(f, "Status: {{"));
        try!(write!(
            f,
            "code: \"{}({})\", message: \"{}\"",
            self.code(),
            self.code().to_int(),
            self.message()
        ));
        match self.trace {
            Some(ref trace) => {
                try!(write!(f, ", backtrace: \"\n{:?}\n\"", trace));
            }
            None => {}
        }
        try!(write!(f, "}}"));
        Ok(())
    }
}

pub(crate) trait ApiResult<T, E> {
    fn from_api_status(status: c_uint, ok_val: T) -> result::Result<T, E>;
}

impl<T> ApiResult<T, Status> for result::Result<T, Status> {
    fn from_api_status(status: c_uint, ok_val: T) -> Self {
        let code = Code::from_int(status);
        match code {
            Code::Ok => Ok(ok_val),
            _ => unsafe {
                let trace = Backtrace::new();

                let mut size: usize = 0;
                let s = _primitiv::primitivGetMessage(ptr::null_mut(), &mut size as *mut _);
                assert!(Code::is_ok(s));
                let buffer = CString::new(Vec::with_capacity(size)).unwrap().into_raw();
                let s = _primitiv::primitivGetMessage(buffer, &mut size as *mut _);
                assert!(Code::is_ok(s));
                let message = CString::from_raw(buffer).into_string().unwrap();

                let enabled = match env::var_os("RUST_BACKTRACE") {
                    Some(ref val) if val != "0" => true,
                    _ => false,
                };
                Err(Status::new(
                    code,
                    message,
                    if enabled { Some(trace) } else { None },
                ))
            },
        }
    }
}

pub(crate) type Result<T> = result::Result<T, Status>;

macro_rules! check_api_status {
    ($status:expr) => {
        match Result::from_api_status($status, 0) {
            Ok(_) => {},
            Err(s) => {
                panic!("{:?}", s);
            },
        }
    }
}
