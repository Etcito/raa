//! Bounded, stable-handle storage for local-light descriptors.
//!
//! Handoff: `LocalLights::add(&mut self, desc: LocalLight) -> Result<LightHandle, RaaError>`
//! in `crates/raa-lit/src/local_lights.rs` validates through `LocalLight::validate`, then
//! inserts into this fixed-capacity, ascending-`LightHandle` registry. Consumers attach this
//! module with `mod local_lights;` and its `pub use local_lights::LocalLights` export in
//! `crates/raa-lit/src/lib.rs`; the registry state is keyed by the stable `LightHandle` from
//! `local_light.rs`. The attachment lifecycle extends this registry and keeps its same keys.

use raa_core::RaaError;
use raa_math::{Mat4, Vec3};

use crate::local_light::{LightHandle, LocalLight, MAX_LIGHT_HANDLE};

/// Maximum number of local lights held by the renderer.
pub const MAX_LOCAL_LIGHTS: usize = 256;

#[derive(Copy, Clone, Debug, PartialEq)]
struct Attachment {
    parent: u32,
    offset: [f32; 3],
    world: Mat4,
    world_direction: [f32; 3],
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct Entry {
    handle: LightHandle,
    desc: LocalLight,
    attach: Option<Attachment>,
}

/// Fixed-capacity local-light registry. Slots are kept in ascending stable-handle order.
/// Handoff: `LocalLight::validate(self) -> Result<Self, RaaError>` from
/// `crates/raa-lit/src/local_light.rs` validates descriptors before registry mutation.
/// `crates/raa-lit/src/lib.rs` re-exports this `LocalLights` registry keyed by `LightHandle`.
/// Engine passes JS instance-group handles to `validate_parent` before scene changes, then
/// `update_parent`/`remove_parent` before tiles. `get` retains local direction; `world_desc` is world.
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
        self.entries[self.len] = Some(Entry {
            handle,
            desc,
            attach: None,
        });
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

    /// Validates atomically; attached direction is local and authored position is refused.
    pub fn replace(
        &mut self,
        handle: LightHandle,
        desc: LocalLight,
        position_authored: bool,
    ) -> Result<(), RaaError> {
        let index = self.index_of(handle).ok_or_else(unknown_handle)?;
        let desc = desc.validate()?;
        let entry = self.entries[index]
            .as_mut()
            .expect("occupied registry entry");
        let old = world_desc(entry);
        let mut next = desc;
        if let Some(attachment) = &entry.attach {
            if position_authored {
                return Err(error(
                    "light.attached_position",
                    "an attached light's position follows its parent; change the offset via attach",
                ));
            }
            let (position, world_direction) =
                transformed(attachment.offset, desc.direction, &attachment.world)?;
            next.position = position;
            entry.attach = Some(Attachment {
                world_direction,
                ..*attachment
            });
        }
        entry.desc = next;
        if world_desc(entry) != old {
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

    /// Attaches to a JS instance-group handle, retaining range and size in metres.
    pub fn attach(
        &mut self,
        handle: LightHandle,
        parent: u32,
        offset: [f32; 3],
        world: &Mat4,
    ) -> Result<(), RaaError> {
        let index = self.index_of(handle).ok_or_else(unknown_handle)?;
        let entry = self.entries[index]
            .as_mut()
            .expect("occupied registry entry");
        let (position, world_direction) = transformed(offset, entry.desc.direction, world)?;
        let old = world_desc(entry);
        entry.desc.position = position;
        entry.attach = Some(Attachment {
            parent,
            offset,
            world: *world,
            world_direction,
        });
        self.dirty |= world_desc(entry) != old;
        Ok(())
    }

    /// Detaches without changing the latest world position and direction.
    pub fn detach(&mut self, handle: LightHandle) -> Result<(), RaaError> {
        let index = self.index_of(handle).ok_or_else(unknown_handle)?;
        let entry = self.entries[index]
            .as_mut()
            .expect("occupied registry entry");
        if let Some(attachment) = entry.attach.take() {
            entry.desc.direction = attachment.world_direction;
        }
        Ok(())
    }

    /// Validates every light attached to `parent` can adopt `world` before scene changes.
    pub fn validate_parent(&self, parent: u32, world: &Mat4) -> Result<(), RaaError> {
        for entry in self.entries[..self.len].iter().flatten() {
            if let Some(attachment) = &entry.attach
                && attachment.parent == parent
            {
                transformed(attachment.offset, entry.desc.direction, world)?;
            }
        }
        Ok(())
    }

    /// Applies a parent transform atomically, validating all affected lights before mutation.
    pub fn update_parent(&mut self, parent: u32, world: &Mat4) -> Result<(), RaaError> {
        self.validate_parent(parent, world)?;
        for entry in self.entries[..self.len].iter_mut().flatten() {
            if let Some(attachment) = entry.attach
                && attachment.parent == parent
            {
                let (position, world_direction) =
                    transformed(attachment.offset, entry.desc.direction, world)?;
                let old = world_desc(entry);
                entry.desc.position = position;
                entry.attach = Some(Attachment {
                    world: *world,
                    world_direction,
                    ..attachment
                });
                self.dirty |= world_desc(entry) != old;
            }
        }
        Ok(())
    }

    /// Detaches every light from `parent` at its last valid world transform.
    pub fn remove_parent(&mut self, parent: u32) {
        for entry in self.entries[..self.len].iter_mut().flatten() {
            if let Some(attachment) = entry.attach
                && attachment.parent == parent
            {
                entry.attach = None;
                entry.desc.direction = attachment.world_direction;
            }
        }
    }

    /// Number of live lights.
    pub fn active_count(&self) -> u32 {
        self.len as u32
    }

    /// Number of lights currently attached to an instance-group parent.
    pub fn attached_count(&self) -> u32 {
        self.entries[..self.len]
            .iter()
            .flatten()
            .filter(|entry| entry.attach.is_some())
            .count() as u32
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

/// Effective world-space descriptor: latest world position and, while attached, world direction.
fn world_desc(entry: &Entry) -> LocalLight {
    let mut desc = entry.desc;
    if let Some(attachment) = &entry.attach {
        desc.direction = attachment.world_direction;
    }
    desc
}

/// Transforms an offset and local direction, refusing nonfinite or degenerate results.
fn transformed(
    offset: [f32; 3],
    local_direction: [f32; 3],
    world: &Mat4,
) -> Result<([f32; 3], [f32; 3]), RaaError> {
    let position = world.transform_point(Vec3::from_array(offset)).to_array();
    let vector = world.transform_vector(Vec3::from_array(local_direction));
    let length_squared =
        f64::from(vector.x).powi(2) + f64::from(vector.y).powi(2) + f64::from(vector.z).powi(2);
    if !(position.iter().all(|value| value.is_finite())
        && world.cols.iter().flatten().all(|value| value.is_finite())
        && length_squared.is_finite()
        && length_squared > 0.0)
    {
        return Err(error(
            "light.invalid_attachment",
            "attachment must map to a finite position and nonzero direction",
        ));
    }
    let inverse_length = 1.0 / length_squared.sqrt();
    let direction =
        [vector.x, vector.y, vector.z].map(|value| (f64::from(value) * inverse_length) as f32);
    Ok((position, direction))
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

    use core::f32::consts::FRAC_PI_2;
    use raa_math::Quat;

    fn posed() -> Mat4 {
        Mat4::from_trs(
            Vec3::new(10.0, -2.0, 4.0),
            Quat::from_rotation_y(FRAC_PI_2),
            Vec3::new(2.0, 1.0, 0.5),
        )
    }

    fn at(lights: &LocalLights, handle: LightHandle) -> LocalLight {
        *lights.get(handle).unwrap()
    }

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    fn invalid_attachment(result: Result<(), RaaError>) {
        assert_eq!(result.unwrap_err().code(), "light.invalid_attachment");
    }

    #[test]
    fn attach_transforms_offset_and_local_direction_and_detach_keeps_world() {
        let mut lights = LocalLights::new();
        let desc = LocalLight {
            direction: [1.0, 0.0, -1.0],
            ..LocalLight::default()
        };
        let handle = lights.add(desc).unwrap();
        lights.attach(handle, 7, [1.0, 2.0, 3.0], &posed()).unwrap();
        let world = world_desc(lights.entries[0].as_ref().unwrap());
        assert!(close(world.position, [11.5, 0.0, 2.0]));
        let k = 1.0 / 17.0_f32.sqrt();
        assert!(close(world.direction, [-k, 0.0, -4.0 * k]));
        assert_eq!((world.range, world.size), (6.0, [1.0, 1.0]));
        lights.clear_dirty();
        lights.update_parent(7, &posed()).unwrap();
        lights.detach(handle).unwrap();
        assert_eq!(at(&lights, handle), world);
        assert!(!lights.is_dirty());
    }

    #[test]
    fn reattach_replaces_parent_and_offset_atomically() {
        let mut lights = LocalLights::new();
        let handle = lights.add(LocalLight::default()).unwrap();
        lights.attach(handle, 1, [1.0, 2.0, 3.0], &posed()).unwrap();
        let moved = Mat4::from_translation(Vec3::new(-3.0, 1.0, 0.0));
        lights.attach(handle, 2, [0.0, 1.0, 0.0], &moved).unwrap();
        assert_eq!(lights.attached_count(), 1);
        assert!(close(at(&lights, handle).position, [-3.0, 2.0, 0.0]));
        let kept = lights.entries;
        lights.clear_dirty();
        let mut invalid = posed();
        invalid.cols[3][3] = f32::INFINITY;
        invalid_attachment(lights.attach(handle, 3, [0.0; 3], &invalid));
        invalid_attachment(lights.attach(handle, 3, [f32::NAN; 3], &moved));
        lights.update_parent(1, &posed()).unwrap();
        lights.attach(handle, 2, [0.0, 1.0, 0.0], &moved).unwrap();
        assert_eq!(lights.entries, kept);
        assert!(!lights.is_dirty());
    }

    #[test]
    fn parent_updates_are_atomic_and_removal_detaches_at_last_world() {
        let mut lights = LocalLights::new();
        let first = lights
            .add(LocalLight {
                direction: [1.0, 0.0, 0.0],
                ..LocalLight::default()
            })
            .unwrap();
        let second = lights.add(LocalLight::default()).unwrap();
        let other = lights.add(LocalLight::default()).unwrap();
        lights.attach(first, 7, [0.0; 3], &Mat4::IDENTITY).unwrap();
        lights
            .attach(second, 7, [1.0, 0.0, 0.0], &Mat4::IDENTITY)
            .unwrap();
        lights.attach(other, 8, [0.0; 3], &Mat4::IDENTITY).unwrap();
        lights.clear_dirty();
        lights.update_parent(7, &posed()).unwrap();
        assert!(lights.is_dirty());
        assert!(close(at(&lights, second).position, [10.0, -2.0, 2.0]));
        assert_eq!(at(&lights, other).position, [0.0; 3]);
        let kept = lights.entries;
        lights.clear_dirty();
        let flatten = Mat4::from_scale(Vec3::new(1.0, 1.0, 0.0));
        invalid_attachment(lights.validate_parent(7, &flatten));
        invalid_attachment(lights.update_parent(7, &flatten));
        assert_eq!(lights.entries, kept);
        lights.remove_parent(7);
        assert_eq!(lights.attached_count(), 1);
        assert_eq!(at(&lights, second), world_desc(kept[1].as_ref().unwrap()));
        lights.update_parent(7, &Mat4::IDENTITY).unwrap();
        assert!(!lights.is_dirty());
        lights.remove(other).unwrap();
        assert_eq!(lights.attached_count(), 0);
        assert_eq!(
            lights.detach(other).unwrap_err().code(),
            "api.unknown_handle"
        );
    }

    #[test]
    fn attached_position_is_refused_and_direction_only_updates_apply_locally() {
        let mut lights = LocalLights::new();
        let handle = lights.add(LocalLight::default()).unwrap();
        lights.attach(handle, 4, [1.0, 2.0, 3.0], &posed()).unwrap();
        let before = at(&lights, handle);
        let mut changed = before;
        changed.position = [99.0, 0.0, 0.0];
        assert_eq!(
            lights.replace(handle, changed, true).unwrap_err().code(),
            "light.attached_position"
        );
        assert_eq!(at(&lights, handle), before);
        changed.direction = [0.0, 0.0, 2.0];
        lights.replace(handle, changed, false).unwrap();
        assert_eq!(at(&lights, handle).direction, [0.0, 0.0, 1.0]);
        assert_eq!(at(&lights, handle).position, before.position);
        assert!(close(
            world_desc(lights.entries[0].as_ref().unwrap()).direction,
            [1.0, 0.0, 0.0]
        ));
    }
}
