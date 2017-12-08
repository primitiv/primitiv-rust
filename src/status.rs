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
    Cancelled,
    Unknown,
    InvalidArgument,
    DeadlineExceeded,
    NotFound,
    AlreadyExists,
    PermissionDenied,
    ResourceExhausted,
    FailedPrecondition,
    Aborted,
    OutOfRange,
    Unimplemented,
    Internal,
    Unavailable,
    DataLoss,
    Unauthenticated,
    UnrecognizedEnumValue(c_uint),
}

impl Code {
    fn from_int(value: c_uint) -> Self {
        match value {
            0 => Code::Ok,
            1 => Code::Cancelled,
            2 => Code::Unknown,
            3 => Code::InvalidArgument,
            4 => Code::DeadlineExceeded,
            5 => Code::NotFound,
            6 => Code::AlreadyExists,
            7 => Code::PermissionDenied,
            8 => Code::ResourceExhausted,
            9 => Code::FailedPrecondition,
            10 => Code::Aborted,
            11 => Code::OutOfRange,
            12 => Code::Unimplemented,
            13 => Code::Internal,
            14 => Code::Unavailable,
            15 => Code::DataLoss,
            16 => Code::Unauthenticated,
            c => Code::UnrecognizedEnumValue(c),
        }
    }

    fn to_int(&self) -> c_uint {
        match self {
            &Code::UnrecognizedEnumValue(c) => c, 
            &Code::Ok => 0,
            &Code::Cancelled => 1,
            &Code::Unknown => 2,
            &Code::InvalidArgument => 3,
            &Code::DeadlineExceeded => 4,
            &Code::NotFound => 5,
            &Code::AlreadyExists => 6,
            &Code::PermissionDenied => 7,
            &Code::ResourceExhausted => 8,
            &Code::FailedPrecondition => 9,
            &Code::Aborted => 10,
            &Code::OutOfRange => 11,
            &Code::Unimplemented => 12,
            &Code::Internal => 13,
            &Code::Unavailable => 14,
            &Code::DataLoss => 15,
            &Code::Unauthenticated => 16,
        }
    }

    fn to_c(&self) -> _primitiv::primitiv_Code {
        unsafe { mem::transmute(self.to_int()) }
    }

    fn from_c(value: _primitiv::primitiv_Code) -> Self {
        Self::from_int(value as c_uint)
    }
}

impl fmt::Display for Code {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            &Code::Ok => f.write_str("Ok"),
            &Code::Cancelled => f.write_str("Cancelled"),
            &Code::Unknown => f.write_str("Unknown"),
            &Code::InvalidArgument => f.write_str("InvalidArgument"),
            &Code::DeadlineExceeded => f.write_str("DeadlineExceeded"),
            &Code::NotFound => f.write_str("NotFound"),
            &Code::AlreadyExists => f.write_str("AlreadyExists"),
            &Code::PermissionDenied => f.write_str("PermissionDenied"),
            &Code::ResourceExhausted => f.write_str("ResourceExhausted"),
            &Code::FailedPrecondition => f.write_str("FailedPrecondition"),
            &Code::Aborted => f.write_str("Aborted"),
            &Code::OutOfRange => f.write_str("OutOfRange"),
            &Code::Unimplemented => f.write_str("Unimplemented"),
            &Code::Internal => f.write_str("Internal"),
            &Code::Unavailable => f.write_str("Unavailable"),
            &Code::DataLoss => f.write_str("DataLoss"),
            &Code::Unauthenticated => f.write_str("Unauthenticated"),
            &Code::UnrecognizedEnumValue(c) => write!(f, "UnrecognizedEnumValue({})", c),
        }
    }
}

pub struct Status {
    inner: *mut _primitiv::primitiv_Status,
}

impl_wrap!(Status, primitiv_Status);
impl_new!(Status, primitiv_Status_new);
impl_drop!(Status, primitiv_Status_delete);

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

impl Debug for Status {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{{inner:{:?}, ", self.as_inner_ptr())?;
        write!(f, "{}: ", self.code())?;
        let messsage = unsafe {
            match CStr::from_ptr(_primitiv::primitiv_Status_get_message(self.as_inner_ptr()))
                .to_str() {
                Ok(s) => s,
                Err(_) => "<invalid UTF-8 in message>",
            }
        };
        f.write_str(messsage)?;
        write!(f, "}}")?;
        Ok(())
    }
}

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
