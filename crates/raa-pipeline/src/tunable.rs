//! S56 tunable specs shared by every feature crate's `set_tunable`.

/// One promoted shader value: its default, inclusive range and rebuild class.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TunableSpec {
    pub key: &'static str,
    pub default: f64,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub integer: bool,
    /// "none", "pipeline" or "resources".
    pub rebuild: &'static str,
    /// file:line (from the repository root) of the declaration holding the default.
    pub source: &'static str,
}

/// The first spec whose key equals `key`.
pub fn find_tunable<'a>(specs: &'a [TunableSpec], key: &str) -> Option<&'a TunableSpec> {
    specs.iter().find(|s| s.key == key)
}

/// Checks, in order: unknown key, not finite, not an integer, below min, above max.
pub fn validate_tunable<'a>(
    specs: &'a [TunableSpec],
    key: &str,
    value: f64,
) -> Result<&'a TunableSpec, String> {
    let spec = find_tunable(specs, key).ok_or_else(|| format!("unknown tunable `{key}`"))?;
    if !value.is_finite() {
        return Err(format!("`{key}` must be finite, got {value}"));
    }
    if spec.integer && value.fract() != 0.0 {
        return Err(format!("`{key}` must be an integer, got {value}"));
    }
    if let Some(min) = spec.min
        && value < min
    {
        return Err(format!("`{key}` must be >= {min}, got {value}"));
    }
    if let Some(max) = spec.max
        && value > max
    {
        return Err(format!("`{key}` must be <= {max}, got {value}"));
    }
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPECS: &[TunableSpec] = &[
        TunableSpec {
            key: "x",
            default: 0.5,
            min: Some(0.0),
            max: Some(1.0),
            integer: false,
            rebuild: "pipeline",
            source: "a.wgsl:1",
        },
        TunableSpec {
            key: "n",
            default: 4.0,
            min: Some(1.0),
            max: Some(8.0),
            integer: true,
            rebuild: "pipeline",
            source: "a.wgsl:2",
        },
        TunableSpec {
            key: "x",
            default: 9.0,
            min: None,
            max: None,
            integer: false,
            rebuild: "none",
            source: "a.wgsl:3",
        },
    ];

    #[test]
    fn find_returns_first_match_or_none() {
        assert_eq!(find_tunable(SPECS, "x").map(|s| s.source), Some("a.wgsl:1"));
        assert_eq!(find_tunable(SPECS, "n").map(|s| s.default), Some(4.0));
        assert!(find_tunable(SPECS, "nope").is_none());
    }

    #[test]
    fn validate_messages_in_check_order() {
        let err = |k: &str, v: f64| validate_tunable(SPECS, k, v).unwrap_err();
        assert_eq!(err("nope", f64::NAN), "unknown tunable `nope`");
        assert_eq!(err("x", f64::NAN), "`x` must be finite, got NaN");
        assert_eq!(err("n", 2.5), "`n` must be an integer, got 2.5");
        assert_eq!(err("n", 20.5), "`n` must be an integer, got 20.5");
        assert_eq!(err("x", -0.25), "`x` must be >= 0, got -0.25");
        assert_eq!(err("x", 1.5), "`x` must be <= 1, got 1.5");
        assert_eq!(err("n", 0.0), "`n` must be >= 1, got 0");
        assert_eq!(validate_tunable(SPECS, "x", 0.75), Ok(&SPECS[0]));
        assert_eq!(validate_tunable(SPECS, "n", 8.0), Ok(&SPECS[1]));
    }
}
