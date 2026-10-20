//! Named provider parameters and tessellation quality.

use crate::GeomError;

/// One parameter value: a scalar or a 3-vector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ParamValue {
    Scalar(f32),
    Vec3([f32; 3]),
}

/// Named parameters, kept in insertion order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Params {
    entries: Vec<(String, ParamValue)>,
}

impl Params {
    pub fn new() -> Params {
        Params::default()
    }

    /// Builder form of [`Params::set`].
    pub fn with(mut self, name: &str, v: ParamValue) -> Self {
        self.set(name, v);
        self
    }

    /// Sets `name`; an existing entry is replaced in place, keeping its position.
    pub fn set(&mut self, name: &str, v: ParamValue) {
        match self.entries.iter_mut().find(|(n, _)| n == name) {
            Some(entry) => entry.1 = v,
            None => self.entries.push((name.to_string(), v)),
        }
    }

    pub fn get(&self, name: &str) -> Option<&ParamValue> {
        self.entries.iter().find(|(n, _)| n == name).map(|(_, v)| v)
    }

    /// The scalar `name` of a provider of `kind`.
    pub fn scalar(&self, kind: &str, name: &str) -> Result<f32, GeomError> {
        match self.get(name) {
            Some(ParamValue::Scalar(v)) => Ok(*v),
            Some(_) => Err(invalid(kind, name, "expected a scalar")),
            None => Err(missing(kind, name)),
        }
    }

    /// The 3-vector `name` of a provider of `kind`.
    pub fn vec3(&self, kind: &str, name: &str) -> Result<[f32; 3], GeomError> {
        match self.get(name) {
            Some(ParamValue::Vec3(v)) => Ok(*v),
            Some(_) => Err(invalid(kind, name, "expected a vec3")),
            None => Err(missing(kind, name)),
        }
    }

    /// Parameter names in insertion order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|(n, _)| n.as_str())
    }
}

fn missing(kind: &str, name: &str) -> GeomError {
    GeomError::MissingParam {
        kind: kind.to_string(),
        name: name.to_string(),
    }
}

fn invalid(kind: &str, name: &str, reason: &str) -> GeomError {
    GeomError::InvalidParam {
        kind: kind.to_string(),
        name: name.to_string(),
        reason: reason.to_string(),
    }
}

/// Tessellation quality as a segment count, always within [3, 256].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Quality {
    pub segments: u32,
}

impl Quality {
    pub const LOW: Quality = Quality { segments: 8 };
    pub const LOD0: Quality = Quality { segments: 24 };
    pub const HIGH: Quality = Quality { segments: 64 };

    /// Quality with `segments` clamped to [3, 256].
    pub fn new(segments: u32) -> Quality {
        Quality {
            segments: segments.clamp(3, 256),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_keep_insertion_order() {
        let p = Params::new()
            .with("radius", ParamValue::Scalar(1.0))
            .with("halfExtents", ParamValue::Vec3([1.0, 2.0, 3.0]))
            .with("angle", ParamValue::Scalar(0.5));
        let names: Vec<&str> = p.names().collect();
        assert_eq!(names, ["radius", "halfExtents", "angle"]);
    }

    #[test]
    fn set_replaces_in_place() {
        let mut p = Params::new()
            .with("a", ParamValue::Scalar(1.0))
            .with("b", ParamValue::Scalar(2.0));
        p.set("a", ParamValue::Scalar(5.0));
        let names: Vec<&str> = p.names().collect();
        assert_eq!(names, ["a", "b"]);
        assert_eq!(p.get("a"), Some(&ParamValue::Scalar(5.0)));
        assert_eq!(p.scalar("k", "a"), Ok(5.0));
    }

    #[test]
    fn vec3_reads_value() {
        let p = Params::new().with("v", ParamValue::Vec3([1.0, 2.0, 3.0]));
        assert_eq!(p.vec3("k", "v"), Ok([1.0, 2.0, 3.0]));
    }

    #[test]
    fn missing_param_has_code_and_names() {
        let p = Params::new();
        let e = p.scalar("sphere", "radius").unwrap_err();
        assert_eq!(e.code(), "geom.missing_param");
        assert_eq!(
            e,
            GeomError::MissingParam {
                kind: "sphere".into(),
                name: "radius".into()
            }
        );
        assert_eq!(
            p.vec3("box", "halfExtents").unwrap_err().code(),
            "geom.missing_param"
        );
    }

    #[test]
    fn wrong_variant_is_invalid_param() {
        let p = Params::new()
            .with("radius", ParamValue::Vec3([1.0, 1.0, 1.0]))
            .with("halfExtents", ParamValue::Scalar(1.0));
        let e = p.scalar("sphere", "radius").unwrap_err();
        assert_eq!(e.code(), "geom.invalid_param");
        assert_eq!(
            e,
            GeomError::InvalidParam {
                kind: "sphere".into(),
                name: "radius".into(),
                reason: "expected a scalar".into()
            }
        );
        let e = p.vec3("box", "halfExtents").unwrap_err();
        assert_eq!(e.code(), "geom.invalid_param");
        assert_eq!(
            e,
            GeomError::InvalidParam {
                kind: "box".into(),
                name: "halfExtents".into(),
                reason: "expected a vec3".into()
            }
        );
    }

    #[test]
    fn quality_clamps_to_range() {
        assert_eq!(Quality::new(0).segments, 3);
        assert_eq!(Quality::new(1000).segments, 256);
        assert_eq!(Quality::new(24), Quality::LOD0);
    }

    #[test]
    fn quality_constants() {
        assert_eq!(Quality::LOW.segments, 8);
        assert_eq!(Quality::LOD0.segments, 24);
        assert_eq!(Quality::HIGH.segments, 64);
    }
}
