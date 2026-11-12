//! Error type of the cluster crate: a stable code, the JSON-style path of the field it names and
//! a message. The S29 code constants below are the only codes this crate produces.

/// The input mesh is not valid for the build (`cluster.invalid_mesh`).
pub const INVALID_MESH: &str = "cluster.invalid_mesh";
/// The build options are out of range (`cluster.invalid_options`).
pub const INVALID_OPTIONS: &str = "cluster.invalid_options";
/// No triangle survives quantisation (`cluster.empty`).
pub const EMPTY: &str = "cluster.empty";
/// The output exceeds the 2^24 cluster or 2^32 - 1 index limits (`cluster.too_large`).
pub const TOO_LARGE: &str = "cluster.too_large";
/// The bake has not finished (`cluster.not_finished`).
pub const NOT_FINISHED: &str = "cluster.not_finished";
/// A section or record does not satisfy its structural rules (`cluster.invalid_data`).
pub const INVALID_DATA: &str = "cluster.invalid_data";

/// A refusal of the cluster crate. `path` names the field, or is empty when the refusal concerns
/// the whole input; `message` says why.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterError {
    pub code: &'static str,
    pub message: String,
    pub path: String,
}

impl ClusterError {
    /// A refusal without a field path.
    pub fn new(code: &'static str, message: impl Into<String>) -> ClusterError {
        ClusterError {
            code,
            message: message.into(),
            path: String::new(),
        }
    }

    /// A refusal naming the field at `path`.
    pub fn at(
        code: &'static str,
        path: impl Into<String>,
        message: impl Into<String>,
    ) -> ClusterError {
        ClusterError {
            code,
            message: message.into(),
            path: path.into(),
        }
    }

    /// The stable code of this refusal.
    pub fn code(&self) -> &'static str {
        self.code
    }
}

impl std::fmt::Display for ClusterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.path.is_empty() {
            write!(f, "{}: {}", self.code, self.message)
        } else {
            write!(f, "{} at {}: {}", self.code, self.path, self.message)
        }
    }
}

impl std::error::Error for ClusterError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_accessor_returns_the_code_of_each_constructor() {
        let plain = ClusterError::new(EMPTY, "no triangle survives quantisation");
        assert_eq!(plain.code(), "cluster.empty");
        let at = ClusterError::at(INVALID_OPTIONS, "/minReduction", "must be in (0, 1)");
        assert_eq!(at.code(), "cluster.invalid_options");
        assert_eq!(at.path, "/minReduction");
        assert_eq!(at.message, "must be in (0, 1)");
        assert_eq!(plain.path, "");
    }

    #[test]
    fn display_names_the_path_when_there_is_one() {
        let at = ClusterError::at(INVALID_DATA, "/clusters/3", "count out of range");
        assert_eq!(
            at.to_string(),
            "cluster.invalid_data at /clusters/3: count out of range"
        );
    }

    #[test]
    fn display_omits_the_path_when_it_is_empty() {
        let plain = ClusterError::new(TOO_LARGE, "more than 2^24 clusters");
        assert_eq!(
            plain.to_string(),
            "cluster.too_large: more than 2^24 clusters"
        );
    }

    #[test]
    fn code_constants_are_the_s29_strings() {
        assert_eq!(INVALID_MESH, "cluster.invalid_mesh");
        assert_eq!(INVALID_OPTIONS, "cluster.invalid_options");
        assert_eq!(EMPTY, "cluster.empty");
        assert_eq!(TOO_LARGE, "cluster.too_large");
        assert_eq!(NOT_FINISHED, "cluster.not_finished");
        assert_eq!(INVALID_DATA, "cluster.invalid_data");
    }

    #[test]
    fn is_a_std_error() {
        fn as_error(e: &dyn std::error::Error) -> String {
            e.to_string()
        }
        let e = ClusterError::new(NOT_FINISHED, "bake not done");
        assert_eq!(as_error(&e), "cluster.not_finished: bake not done");
    }
}
