extern crate backtrace;

use primitiv_sys as _primitiv;
use libc::c_uint;
use self::backtrace::{Backtrace, BacktraceFrame, BacktraceSymbol};
use std::error::Error;
use std::ffi::CStr;
use std::ffi::CString;
use std::ffi::NulError;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt;
use std::mem;
use std::result;
use std::str::Utf8Error;
use Wrap;

fn prev_symbol(level: u32) -> Option<BacktraceSymbol> {
    let (trace, curr_file, curr_line) = (Backtrace::new(), file!(), line!());
    let frames = trace.frames();
    frames
        .iter()
        .flat_map(BacktraceFrame::symbols)
        .skip_while(|s| {
            s.filename().map(|p| !p.ends_with(curr_file)).unwrap_or(
                true,
            ) || s.lineno() != Some(curr_line)
        })
        .nth(1 + level as usize)
        .cloned()
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Copy, Clone)]
pub enum Code {
    Ok,
    Error,
    UnrecognizedEnumValue(c_uint),
}

impl Code {
    fn from_int(value: c_uint) -> Self {
        match value {
            0 => Code::Ok,
            1 => Code::Error,
            c => Code::UnrecognizedEnumValue(c),
        }
    }

    fn to_int(&self) -> c_uint {
        match self {
            &Code::UnrecognizedEnumValue(c) => c, 
            &Code::Ok => 0,
            &Code::Error => 1,
        }
    }

    fn to_c(&self) -> _primitiv::primitiv_Status {
        unsafe { mem::transmute(self.to_int()) }
    }

    fn from_c(value: _primitiv::primitiv_Status) -> Self {
        Self::from_int(value as c_uint)
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

pub struct Status {
    code: Code,
    message: String,
}

//  macro_rules! into_result {
//      ($expr:expr, $retval:ident) => {
//          let code: Code::from_int($expr);
//          match code {
//              Code::Ok => Ok($retval),
//              _ => Err(Status::new(code, _primitiv::primitiv_Status_get_message()),
//          }
//      }
//  }
//

// trait FromApi<T> {
//     fn from_api_status<T>(status: c_uint, ok_val: T) -> Self;
// }
pub trait ApiResult<T, E> {
    fn from_api_status(status: c_uint, ok_val: T) -> result::Result<T, E>;
}

impl<T> ApiResult<T, Status> for result::Result<T, Status> {
    fn from_api_status(status: c_uint, ok_val: T) -> Self {
        let code = Code::from_int(status);
        match code {
            Code::Ok => Ok(ok_val),
            _ => unsafe {
                Err(Status::new(
                    code,
                    CStr::from_ptr(_primitiv::primitiv_Status_get_message())
                        .to_str()
                        .unwrap()
                        .to_string(),
                ))
            },
        }
    }
}

pub type Result<T> = result::Result<T, Status>;

impl Status {
    pub fn new(code: Code, message: String) -> Self {
        Status {
            code: code,
            message: message,
        }
        // unsafe {
        //     let inner = _primitiv::primitiv_Status_new();
        //     assert!(!inner.is_null());
        //     Status { inner: inner }
        // }
    }
    //
    //      /*
    //      fn from_code<T>(code: Code) -> Result<T, Self> {
    //  _primitiv::primitiv_Status_get_message()
    //          Status {}
    //      }
    //      */
    //

    pub fn code(&self) -> Code {
        self.code
    }

    pub fn message(&self) -> &str {
        self.message.as_str()
    }
}

/*
pub struct Status {
    inner: *mut _primitiv::primitiv_Status,
}

impl_wrap!(Status, primitiv_Status);

impl Status {
    pub fn new() -> Self {
        unsafe {
            let inner = _primitiv::primitiv_Status_new();
            assert!(!inner.is_null());
            Status { inner: inner }
        }
    }
}

impl Drop for Status {
    fn drop(&mut self) {
        unsafe {
            _primitiv::primitiv_Status_delete(self.inner);
        }
    }
}

impl Status {
    pub fn new_set(code: Code, message: &str) -> Result<Status, NulError> {
        let symbol = prev_symbol(1);
        let file = symbol.as_ref().and_then(BacktraceSymbol::filename).unwrap();
        let line = symbol.as_ref().and_then(BacktraceSymbol::lineno).unwrap();
        let mut status = Status::new();
        status.set(code, file.to_str().unwrap(), line, message)?;
        Ok(status)
    }

    pub fn code(&self) -> Code {
        unsafe { Code::from_int(_primitiv::primitiv_Status_get_code(self.as_inner_ptr())) }
    }

    pub fn is_ok(&self) -> bool {
        self.code() == Code::Ok
    }

    pub(crate) fn into_result(self) -> Result<(), Status> {
        if self.is_ok() { Ok(()) } else { Err(self) }
    }

    pub fn set(
        &mut self,
        code: Code,
        file: &str,
        line: u32,
        message: &str,
    ) -> Result<(), NulError> {
        let file = CString::new(file)?;
        let message = CString::new(message)?;
        unsafe {
            _primitiv::primitiv_Status_set_status(
                self.as_inner_mut_ptr(),
                code.to_c(),
                file.as_ptr(),
                line,
                message.as_ptr(),
            );
        }
        Ok(())
    }
}

macro_rules! invalid_arg {
    ($fmt:expr) => {
        Status::new_set(Code::InvalidArgument, $fmt).unwrap()
    };
    ($fmt:expr, $($arg:tt)*) => ({
        let messsag = format!($fmt, $($arg)*);
        Status::new_set(Code::InvalidArgument, &message).unwrap()
    });
}

impl Display for Status {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}: ", self.code())?;
        let message = unsafe {
            match CStr::from_ptr(_primitiv::primitiv_Status_get_message(self.as_inner_ptr()))
                .to_str() {
                Ok(s) => s,
                Err(_) => "<invalid UTF-8 in message>",
            }
        };
        f.write_str(message)
    }
}
*/

impl Debug for Status {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}: ", self.code())?;
        // let messsage = unsafe {
        //     match CStr::from_ptr(_primitiv::primitiv_Status_get_message(self.as_inner_ptr()))
        //         .to_str() {
        //         Ok(s) => s,
        //         Err(_) => "<invalid UTF-8 in message>",
        //     }
        // };
        f.write_str(self.message())?;
        write!(f, "}}")?;
        Ok(())
    }
}
/*

impl From<NulError> for Status {
    fn from(_e: NulError) -> Self {
        invalid_arg!("String contained NUL byte")
    }
}

impl From<Utf8Error> for Status {
    fn from(_e: Utf8Error) -> Self {
        invalid_arg!("String contained invalid UTF-8")
    }
}

impl Error for Status {
    fn description(&self) -> &str {
        unsafe {
            match CStr::from_ptr(_primitiv::primitiv_Status_get_message(self.as_inner_ptr()))
                .to_str() {
                Ok(s) => s,
                Err(_) => "<invalid UTF-8 in message>",
            }
        }
    }

    fn cause(&self) -> Option<&Error> {
        None
    }
}
*/
