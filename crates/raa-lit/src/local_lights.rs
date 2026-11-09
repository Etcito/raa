//! Bounded, stable-handle storage for local-light descriptors.
//!
//! Handoff: `LocalLights::add(&mut self, desc: LocalLight) -> Result<LightHandle, RaaError>`
//! in `crates/raa-lit/src/local_lights.rs` validates through `LocalLight::validate`, then
//! inserts into this fixed-capacity, ascending-`LightHandle` registry. Consumers attach this
//! module with `mod local_lights;` and its `pub use local_lights::LocalLights` export in
//! `crates/raa-lit/src/lib.rs`; the registry state is keyed by the stable `LightHandle` from
//! `local_light.rs`. The attachment lifecycle extends this registry and keeps its same keys.

use raa_core::RaaError;

use crate::local_light::{LightHandle, LocalLight, MAX_LIGHT_HANDLE};

/// Maximum number of local lights held by the renderer.
pub const MAX_LOCAL_LIGHTS: usize = 256;

#[derive(Copy, Clone, Debug, PartialEq)]
struct Entry {
    handle: LightHandle,
    desc: LocalLight,
}

/// Fixed-capacity local-light registry. Slots are kept in ascending stable-handle order.
pub struct LocalLights {
    entries: [Option<Entry>; MAX_LOCAL_LIGHTS],
    len: usize,
    next_handle: Option<LightHandle>,
    dirty: bool,
}

impl Default for LocalLights {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalLights {
    /// Creates an empty registry with all bounded storage allocated up front.
    pub fn new() -> Self {
        Self {
            entries: [None; MAX_LOCAL_LIGHTS],
            len: 0,
            next_handle: Some(1),
            dirty: false,
        }
    }

    /// Validates and adds a light, refusing when capacity or safe handles are exhausted.
    pub fn add(&mut self, desc: LocalLight) -> Result<LightHandle, RaaError> {
        if self.len == MAX_LOCAL_LIGHTS {
            return Err(error("lights.invalid", "local-light capacity is 256"));
        }
        let desc = desc.validate()?;
        let handle = self
            .next_handle
            .ok_or_else(|| error("lights.invalid", "safe local-light handles are exhausted"))?;
        self.next_handle = handle
            .checked_add(1)
            .filter(|next| *next <= MAX_LIGHT_HANDLE);
        self.entries[self.len] = Some(Entry { handle, desc });
        self.len += 1;
        self.dirty = true;
        Ok(handle)
    }

    /// Returns a light by stable handle.
    pub fn get(&self, handle: LightHandle) -> Result<&LocalLight, RaaError> {
        let index = self.index_of(handle).ok_or_else(unknown_handle)?;
        Ok(&self.entries[index]
            .as_ref()
            .expect("occupied registry entry")
            .desc)
    }

    /// Atomically replaces a descriptor after validation. Equal records remain clean.
    pub fn replace(
        &mut self,
        handle: LightHandle,
        desc: LocalLight,
        _position_authored: bool,
    ) -> Result<(), RaaError> {
        let index = self.index_of(handle).ok_or_else(unknown_handle)?;
        let desc = desc.validate()?;
        let entry = self.entries[index]
            .as_mut()
            .expect("occupied registry entry");
        if entry.desc != desc {
            entry.desc = desc;
            self.dirty = true;
        }
        Ok(())
    }

    /// Removes a light without making its stable handle available again.
    pub fn remove(&mut self, handle: LightHandle) -> Result<(), RaaError> {
        let index = self.index_of(handle).ok_or_else(unknown_handle)?;
        for slot in index..self.len - 1 {
            self.entries[slot] = self.entries[slot + 1];
        }
        self.len -= 1;
        self.entries[self.len] = None;
        self.dirty = true;
        Ok(())
    }

    /// Number of live lights.
    pub fn active_count(&self) -> u32 {
        self.len as u32
    }

    /// Number of attached lights; attachments are managed by the lifecycle extension.
    pub fn attached_count(&self) -> u32 {
        0
    }

    /// Whether a registry mutation requires its backing GPU data to be refreshed.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Marks the current registry contents as uploaded.
    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }

    fn index_of(&self, handle: LightHandle) -> Option<usize> {
        self.entries[..self.len]
            .binary_search_by_key(&handle, |entry| {
                entry.expect("occupied registry entry").handle
            })
            .ok()
    }
}

fn error(code: &'static str, message: &str) -> RaaError {
    RaaError::FrameGraph {
        code,
        message: message.to_owned(),
    }
}

fn unknown_handle() -> RaaError {
    error("api.unknown_handle", "unknown local-light handle")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_capacity_handles_and_stale_references() {
        let mut lights = LocalLights::new();
        let handles: Vec<_> = (0..MAX_LOCAL_LIGHTS)
            .map(|_| lights.add(LocalLight::default()).unwrap())
            .collect();
        assert_eq!(lights.active_count(), 256);
        assert_eq!(
            lights.add(LocalLight::default()).unwrap_err().code(),
            "lights.invalid"
        );
        assert!(
            lights
                .add(LocalLight::default())
                .unwrap_err()
                .to_string()
                .contains("256")
        );
        lights.remove(handles[41]).unwrap();
        assert_eq!(
            lights.get(handles[41]).unwrap_err().code(),
            "api.unknown_handle"
        );
        let fresh = lights.add(LocalLight::default()).unwrap();
        assert!(fresh > *handles.last().unwrap());
    }

    #[test]
    fn mutation_is_atomic_and_unchanged_replace_stays_clean() {
        let mut lights = LocalLights::new();
        let handle = lights.add(LocalLight::default()).unwrap();
        lights.clear_dirty();
        lights
            .replace(handle, LocalLight::default(), false)
            .unwrap();
        assert!(!lights.is_dirty());
        let mut invalid = LocalLight::default();
        invalid.range = f32::NAN;
        assert_eq!(
            lights.replace(handle, invalid, false).unwrap_err().code(),
            "light.invalid_desc"
        );
        assert_eq!(*lights.get(handle).unwrap(), LocalLight::default());
        let mut changed = LocalLight::default();
        changed.intensity = 4.0;
        lights.replace(handle, changed, false).unwrap();
        assert!(lights.is_dirty());
    }

    #[test]
    fn removal_preserves_ascending_handle_order_and_exhaustion_refuses() {
        let mut lights = LocalLights::new();
        let handles: Vec<_> = (0..4)
            .map(|_| lights.add(LocalLight::default()).unwrap())
            .collect();
        lights.remove(handles[1]).unwrap();
        let ordered: Vec<_> = lights.entries[..lights.len]
            .iter()
            .map(|entry| entry.unwrap().handle)
            .collect();
        assert_eq!(ordered, [handles[0], handles[2], handles[3]]);
        lights.next_handle = Some(MAX_LIGHT_HANDLE);
        let last = lights.add(LocalLight::default()).unwrap();
        assert_eq!(last, MAX_LIGHT_HANDLE);
        assert!(
            lights
                .add(LocalLight::default())
                .unwrap_err()
                .to_string()
                .contains("exhausted")
        );
    }
}
