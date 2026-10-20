//! CPU side of a frame: S1 instance bytes, the S1 FrameUniform and the draw list.
//! No GPU objects, so `raa-bench` can measure it alone.

use raa_core::RaaError;
use raa_math::Mat4;
use raa_scene::{RenderList, Scene};

use crate::{DrawRange, FrameCounts};

/// Bytes per S1 Instance.
pub const INSTANCE_STRIDE: usize = 128;
/// Bytes of the S1 FrameUniform.
pub const FRAME_UNIFORM_SIZE: usize = 144;

/// Most draws preallocated up front; more only grow the Vec during warm-up.
const DRAWS_PREALLOC: u32 = 1024;

/// Preallocated CPU buffers for one frame, refilled by [`FramePrep::prepare_cpu`]
/// without allocating.
#[derive(Clone, Debug)]
pub struct FramePrep {
    max_instances: u32,
    instances: Vec<u8>,
    used_instances: usize,
    uniform: [u8; FRAME_UNIFORM_SIZE],
    draws: Vec<DrawRange>,
}

impl FramePrep {
    /// Preallocates `max_instances` x 128 B of instance bytes, the 144 B frame uniform and
    /// room for min(max_instances, 1024) draws.
    pub fn new(max_instances: u32) -> FramePrep {
        FramePrep {
            max_instances,
            instances: vec![0; max_instances as usize * INSTANCE_STRIDE],
            used_instances: 0,
            uniform: [0; FRAME_UNIFORM_SIZE],
            draws: Vec::with_capacity(max_instances.min(DRAWS_PREALLOC) as usize),
        }
    }

    /// Writes the S1 Instance of every item, the S1 FrameUniform and the draw ranges.
    ///
    /// More items than `max_instances` is refused with [`RaaError::Resource`] and nothing
    /// is written. `RaaError::code()` then reports `core.resource`; the render code
    /// `render.too_many_instances` starts the message. `triangles` stays 0 here.
    pub fn prepare_cpu(
        &mut self,
        scene: &Scene,
        list: &RenderList,
        surface_is_srgb: bool,
    ) -> Result<FrameCounts, RaaError> {
        let n = list.items.len();
        if n > self.max_instances as usize {
            return Err(RaaError::Resource(format!(
                "render.too_many_instances: {n} items exceed max_instances {}",
                self.max_instances
            )));
        }

        for (chunk, item) in self
            .instances
            .chunks_exact_mut(INSTANCE_STRIDE)
            .zip(list.items.iter())
        {
            write_instance(chunk, &item.world, item.material.0);
        }
        self.used_instances = n;

        write_frame_uniform(&mut self.uniform, scene, surface_is_srgb);

        self.draws.clear();
        self.draws.extend(list.batches.iter().map(|b| DrawRange {
            mesh: b.mesh,
            first_instance: b.first,
            instance_count: b.count,
        }));

        Ok(FrameCounts {
            // n <= max_instances, a u32.
            instances: n as u32,
            draw_calls: self.draws.len() as u32,
            triangles: 0,
        })
    }

    /// Instance bytes of the last prepared frame (128 B per item).
    pub fn instance_bytes(&self) -> &[u8] {
        let end = (self.used_instances * INSTANCE_STRIDE).min(self.instances.len());
        &self.instances[..end]
    }

    /// The 144 B FrameUniform of the last prepared frame.
    pub fn frame_uniform_bytes(&self) -> &[u8] {
        &self.uniform
    }

    /// Draw ranges of the last prepared frame, one per batch.
    pub fn draws(&self) -> &[DrawRange] {
        &self.draws
    }
}

/// Writes `values` as consecutive f32 LE into `out`.
fn put_f32s(out: &mut [u8], values: &[f32]) {
    for (chunk, v) in out.chunks_exact_mut(4).zip(values) {
        chunk.copy_from_slice(&v.to_le_bytes());
    }
}

/// `out` is one 128 B instance slot (invariant: callers pass `chunks_exact_mut(128)`
/// chunks, so the fixed ranges below are in bounds).
fn write_instance(out: &mut [u8], world: &Mat4, material: u32) {
    world.write_le_bytes(&mut out[0..64]);
    let nm = world.normal_matrix();
    put_f32s(&mut out[64..112], nm.as_flattened());
    out[112..116].copy_from_slice(&material.to_le_bytes());
    out[116..128].fill(0);
}

fn write_frame_uniform(out: &mut [u8; FRAME_UNIFORM_SIZE], scene: &Scene, surface_is_srgb: bool) {
    scene.camera.view_proj().write_le_bytes(&mut out[0..64]);
    let eye = scene.camera.eye;
    put_f32s(&mut out[64..80], &[eye.x, eye.y, eye.z, 1.0]);
    let d = scene.light.direction.normalize_or_zero();
    put_f32s(&mut out[80..96], &[d.x, d.y, d.z, 0.0]);
    let (c, i) = (scene.light.color, scene.light.intensity);
    put_f32s(&mut out[96..112], &[c[0] * i, c[1] * i, c[2] * i, 0.0]);
    let (c, i) = (scene.ambient.color, scene.ambient.intensity);
    put_f32s(&mut out[112..128], &[c[0] * i, c[1] * i, c[2] * i, 0.0]);
    let encode = u32::from(!surface_is_srgb);
    for (chunk, v) in out[128..144].chunks_exact_mut(4).zip([encode, 0, 0, 0]) {
        chunk.copy_from_slice(&v.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raa_core::{MaterialId, MeshId};
    use raa_math::{Quat, Vec3};
    use raa_scene::{Batch, RenderItem};

    fn item(world: Mat4, material: u32) -> RenderItem {
        RenderItem {
            mesh: MeshId(3),
            material: MaterialId(material),
            world,
        }
    }

    fn list_of(items: &[RenderItem]) -> RenderList {
        let mut list = RenderList::with_capacity(items.len());
        list.items.extend_from_slice(items);
        list.batches.push(Batch {
            mesh: MeshId(3),
            first: 0,
            count: items.len() as u32,
        });
        list
    }

    fn u32_at(bytes: &[u8], off: usize) -> u32 {
        u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap())
    }

    fn f32s_at(bytes: &[u8], off: usize, n: usize) -> Vec<f32> {
        bytes[off..off + 4 * n]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn one_item_writes_s1_instance_bytes() {
        let world = Mat4::from_trs(
            Vec3::new(1.0, 2.0, 3.0),
            Quat::from_rotation_y(0.7),
            Vec3::new(2.0, 1.0, 0.5),
        );
        let mut prep = FramePrep::new(4);
        let counts = prep
            .prepare_cpu(&Scene::new(), &list_of(&[item(world, 7)]), true)
            .unwrap();
        assert_eq!(
            counts,
            FrameCounts {
                instances: 1,
                draw_calls: 1,
                triangles: 0
            }
        );
        let bytes = prep.instance_bytes();
        assert_eq!(bytes.len(), 128);

        let mut model = [0u8; 64];
        world.write_le_bytes(&mut model);
        assert_eq!(&bytes[0..64], &model);
        assert_eq!(f32s_at(bytes, 64, 12), world.normal_matrix().as_flattened());
        assert_eq!(u32_at(bytes, 112), 7);
        assert!(bytes[116..128].iter().all(|&b| b == 0));
        assert_eq!(
            prep.draws(),
            &[DrawRange {
                mesh: MeshId(3),
                first_instance: 0,
                instance_count: 1
            }]
        );
    }

    #[test]
    fn frame_uniform_flags_follow_surface_srgb() {
        let scene = Scene::new();
        let list = list_of(&[]);
        let mut prep = FramePrep::new(1);

        prep.prepare_cpu(&scene, &list, true).unwrap();
        let u = prep.frame_uniform_bytes();
        assert_eq!(u.len(), 144);
        assert_eq!([0, 4, 8, 12].map(|o| u32_at(u, 128 + o)), [0, 0, 0, 0]);
        let mut vp = [0u8; 64];
        scene.camera.view_proj().write_le_bytes(&mut vp);
        assert_eq!(&u[0..64], &vp);
        let eye = scene.camera.eye;
        assert_eq!(f32s_at(u, 64, 4), [eye.x, eye.y, eye.z, 1.0]);
        let (c, i) = (scene.ambient.color, scene.ambient.intensity);
        assert_eq!(f32s_at(u, 112, 4), [c[0] * i, c[1] * i, c[2] * i, 0.0]);

        prep.prepare_cpu(&scene, &list, false).unwrap();
        let u = prep.frame_uniform_bytes();
        assert_eq!([0, 4, 8, 12].map(|o| u32_at(u, 128 + o)), [1, 0, 0, 0]);
    }

    #[test]
    fn too_many_instances_is_refused_and_keeps_earlier_bytes() {
        let scene = Scene::new();
        let mut prep = FramePrep::new(2);
        let first = list_of(&[item(Mat4::IDENTITY, 1), item(Mat4::IDENTITY, 2)]);
        prep.prepare_cpu(&scene, &first, true).unwrap();
        let before = prep.instance_bytes().to_vec();
        let uniform_before = prep.frame_uniform_bytes().to_vec();
        let draws_before = prep.draws().to_vec();

        let t = Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0));
        let too_many = list_of(&[item(t, 9), item(t, 9), item(t, 9)]);
        let err = prep.prepare_cpu(&scene, &too_many, false).unwrap_err();
        assert_eq!(err.code(), "core.resource");
        let msg = err.to_string();
        assert!(msg.contains("render.too_many_instances:"), "{msg}");
        assert!(msg.contains('3') && msg.contains('2'), "{msg}");

        assert_eq!(prep.instance_bytes(), &before[..]);
        assert_eq!(prep.frame_uniform_bytes(), &uniform_before[..]);
        assert_eq!(prep.draws(), &draws_before[..]);
    }
}
