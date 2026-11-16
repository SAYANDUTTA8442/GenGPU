//! Shared error types for GenGPU.

use std::fmt;

/// Broad category describing why a GenGPU operation failed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ErrorKind {
    /// An argument supplied by the caller is invalid.
    InvalidArgument,
    /// The requested operation or feature is not supported.
    Unsupported,
    /// A requested resource or entity could not be found.
    NotFound,
    /// The operation is incompatible with the current state.
    InvalidState,
    /// A required allocation could not be satisfied.
    OutOfMemory,
    /// An underlying GPU driver operation failed.
    Driver,
    /// An input/output operation failed.
    Io,
    /// An unexpected internal failure occurred.
    Internal,
}

/// An error returned by a GenGPU operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Error {
    kind: ErrorKind,
    message: String,
}

impl Error {
    /// Creates an error with a category and a human-readable message.
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// Returns the broad category of this error.
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// Returns the human-readable error message.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for Error {}

/// Result type used by GenGPU APIs.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::{Error, ErrorKind};

    #[test]
    fn error_exposes_kind_and_message() {
        let error = Error::new(ErrorKind::InvalidArgument, "device index is invalid");

        assert_eq!(error.kind(), ErrorKind::InvalidArgument);
        assert_eq!(error.message(), "device index is invalid");
    }

    #[test]
    fn error_formats_category_and_message() {
        let error = Error::new(ErrorKind::Driver, "initialization failed");

        assert_eq!(error.to_string(), "Driver: initialization failed");
    }

    #[test]
    fn error_implements_standard_error_trait() {
        fn assert_error<T: std::error::Error>() {}
        assert_error::<Error>();
    }
}
