//! Type-keyed storage for the state of feature crates (one value per type), reached by
//! the api files through `EngineState::features` and `InstallCtx::features`.

use std::any::{Any, TypeId};

use raa_core::RaaError;

use crate::convert::api_error;

pub(crate) const FEATURE_MISSING: &str = "api.feature_missing";

#[derive(Default)]
pub(crate) struct FeatureMap {
    entries: Vec<(TypeId, Box<dyn Any>)>,
}

impl FeatureMap {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Stores `v`, replacing a previous value of the same type.
    pub(crate) fn insert<T: 'static>(&mut self, v: T) {
        let id = TypeId::of::<T>();
        match self.entries.iter_mut().find(|(k, _)| *k == id) {
            Some(slot) => slot.1 = Box::new(v),
            None => self.entries.push((id, Box::new(v))),
        }
    }

    pub(crate) fn get<T: 'static>(&self) -> Option<&T> {
        let id = TypeId::of::<T>();
        self.entries
            .iter()
            .find(|(k, _)| *k == id)
            .and_then(|(_, v)| v.downcast_ref::<T>())
    }

    pub(crate) fn get_mut<T: 'static>(&mut self) -> Option<&mut T> {
        let id = TypeId::of::<T>();
        self.entries
            .iter_mut()
            .find(|(k, _)| *k == id)
            .and_then(|(_, v)| v.downcast_mut::<T>())
    }

    /// Like `get_mut`, refusing with `api.feature_missing` naming `what` when absent.
    pub(crate) fn require_mut<T: 'static>(&mut self, what: &str) -> Result<&mut T, RaaError> {
        self.get_mut::<T>().ok_or_else(|| {
            api_error(
                FEATURE_MISSING,
                format!("feature `{what}` is not installed in this engine"),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct Fog(f32);
    #[derive(Debug, PartialEq)]
    struct Bloom(u32);

    #[test]
    fn insert_then_get_returns_the_value_of_that_type() {
        let mut m = FeatureMap::new();
        m.insert(Fog(0.5));
        m.insert(Bloom(8));
        assert_eq!(m.get::<Fog>(), Some(&Fog(0.5)));
        assert_eq!(m.get::<Bloom>(), Some(&Bloom(8)));
        assert!(m.get::<u64>().is_none());
    }

    #[test]
    fn insert_replaces_a_value_of_the_same_type() {
        let mut m = FeatureMap::new();
        m.insert(Bloom(1));
        m.insert(Bloom(2));
        assert_eq!(m.get::<Bloom>(), Some(&Bloom(2)));
        assert_eq!(m.entries.len(), 1);
    }

    #[test]
    fn get_mut_edits_in_place() {
        let mut m = FeatureMap::new();
        m.insert(Fog(0.1));
        if let Some(f) = m.get_mut::<Fog>() {
            f.0 = 0.9;
        }
        assert_eq!(m.get::<Fog>(), Some(&Fog(0.9)));
        assert!(m.get_mut::<Bloom>().is_none());
    }

    #[test]
    fn require_mut_present_and_missing() {
        let mut m = FeatureMap::new();
        m.insert(Bloom(3));
        assert_eq!(m.require_mut::<Bloom>("bloom").map(|b| b.0).ok(), Some(3));
        let e = m.require_mut::<Fog>("fog").err();
        let e = e.expect("missing feature refused");
        assert_eq!(e.code(), "api.feature_missing");
        assert!(e.to_string().contains("fog"));
    }
}
