//! Optional per-vertex colour streams of the two GI voxelizers: albedo = base colour x instance
//! tint x vertex colour. A draw without a stream gets white, so every existing caller keeps its
//! albedo.
//!
//! Two streams exist, because the two drivers read different vertex arrays:
//! - driver A (forward) reads one colour per vertex of the lit mesh, in the order the mesh was
//!   uploaded with, registered by [`GiVertexColors::set_mesh`] under a sub-mesh id;
//! - driver B (virtualized geometry) reads one colour per vertex of a cluster section, in the
//!   order of `ClusterMesh::vertices`, registered by [`GiVertexColors::set_cluster`] under the
//!   cluster-store key and the sub-mesh ordinal. The 16 B cluster vertex has no spare bits and
//!   driver B already binds eight storage buffers, so its colours travel as an `Rgba8Unorm`
//!   texture parallel to the VG vertex pool, rebuilt whenever the pool is repacked.
//!
//! Colours are linear rgba floats (alpha is ignored), stored as `pack4x8unorm` words. The public
//! entry points are `GiHandle::set_vertex_colors` and `GiHandle::set_cluster_vertex_colors`.
//!
//! This is RAA code, not adapted from solstice.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use raa_core::{GpuContext, MeshId, RaaError};
use raa_vg::scene::VgScene;

use crate::error::invalid;

/// A packed white texel: the colour of every vertex without a stream.
pub const WHITE_TEXEL: u32 = 0xFFFF_FFFF;
/// Width in texels of the pool colour texture; the WGSL `RAA_GI_VG_COLOR_WIDTH`.
pub const POOL_COLOR_WIDTH: u32 = 4096;
/// Largest row count of the pool colour texture (the WebGPU default texture dimension).
pub const POOL_COLOR_MAX_ROWS: u32 = 8192;

/// Packs linear rgba floats (4 per vertex) into `pack4x8unorm` words. `key` names the argument
/// in the refusal: a length that is not a multiple of 4 or a non-finite component is
/// `gi.invalid`.
pub fn pack_colors(key: &str, colors: &[f32]) -> Result<Vec<u32>, RaaError> {
    if !colors.len().is_multiple_of(4) {
        return Err(invalid(
            key,
            &format!(
                "{} floats are not rgba per vertex (a multiple of 4)",
                colors.len()
            ),
        ));
    }
    if colors.iter().any(|c| !c.is_finite()) {
        return Err(invalid(key, "colours must be finite"));
    }
    Ok(colors
        .chunks_exact(4)
        .map(|c| {
            let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u32;
            byte(c[0]) | byte(c[1]) << 8 | byte(c[2]) << 16 | byte(c[3]) << 24
        })
        .collect())
}

struct MeshColors {
    packed: Vec<u32>,
    version: u64,
    buffer: Option<wgpu::Buffer>,
}

#[derive(Default)]
struct ColorState {
    meshes: BTreeMap<MeshId, MeshColors>,
    clusters: BTreeMap<(u32, usize), Vec<u32>>,
    cluster_generation: u64,
    counter: u64,
}

/// The registered colour streams; clones share one state (it lives in `GiShared`).
#[derive(Clone, Default)]
pub struct GiVertexColors(Rc<RefCell<ColorState>>);

impl GiVertexColors {
    /// Registers (`Some`) or removes (`None`) the driver A colours of the lit mesh that sub-mesh
    /// `mesh` belongs to: rgba linear per vertex of the whole uploaded vertex array. A stream
    /// whose vertex count differs from the mesh's is ignored when voxelizing (white). A refusal
    /// changes nothing.
    pub fn set_mesh(&self, mesh: MeshId, colors: Option<&[f32]>) -> Result<(), RaaError> {
        let packed = colors.map(|c| pack_colors("vertexColors", c)).transpose()?;
        let mut state = self.0.borrow_mut();
        state.counter += 1;
        let version = state.counter;
        match packed {
            Some(packed) => {
                state.meshes.insert(
                    mesh,
                    MeshColors {
                        packed,
                        version,
                        buffer: None,
                    },
                );
            }
            None => {
                state.meshes.remove(&mesh);
            }
        }
        Ok(())
    }

    /// Registers (`Some`) or removes (`None`) the driver B colours of section `ordinal` of
    /// cluster-store key `key`: rgba linear per vertex of `ClusterMesh::vertices`. A refusal
    /// changes nothing.
    pub fn set_cluster(
        &self,
        key: u32,
        ordinal: usize,
        colors: Option<&[f32]>,
    ) -> Result<(), RaaError> {
        let packed = colors
            .map(|c| pack_colors("clusterVertexColors", c))
            .transpose()?;
        let mut state = self.0.borrow_mut();
        let changed = match packed {
            Some(packed) => {
                state.clusters.insert((key, ordinal), packed);
                true
            }
            None => state.clusters.remove(&(key, ordinal)).is_some(),
        };
        if changed {
            state.cluster_generation += 1;
        }
        Ok(())
    }

    /// The driver A buffer (`array<u32>` storage) and its version for `mesh`, created on first
    /// use; `None` when the mesh has no stream or its length is not `vertex_count`.
    pub(crate) fn mesh_buffer(
        &self,
        gpu: &GpuContext,
        mesh: MeshId,
        vertex_count: u32,
    ) -> Option<(wgpu::Buffer, u64)> {
        let mut state = self.0.borrow_mut();
        let entry = state.meshes.get_mut(&mesh)?;
        if entry.packed.len() != vertex_count as usize || entry.packed.is_empty() {
            return None;
        }
        if entry.buffer.is_none() {
            let buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("gi.vertex_colors"),
                size: (entry.packed.len() * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bytes: Vec<u8> = entry.packed.iter().flat_map(|w| w.to_le_bytes()).collect();
            gpu.queue().write_buffer(&buffer, 0, &bytes);
            entry.buffer = Some(buffer);
        }
        entry.buffer.clone().map(|b| (b, entry.version))
    }

    /// Whether any driver B stream is registered.
    pub(crate) fn has_clusters(&self) -> bool {
        !self.0.borrow().clusters.is_empty()
    }

    /// Advances whenever a stream of either driver changes; GI rebuilds its voxels then.
    pub(crate) fn revision(&self) -> u64 {
        let state = self.0.borrow();
        state.counter + state.cluster_generation
    }

    /// Advances whenever the driver B streams change.
    pub(crate) fn cluster_generation(&self) -> u64 {
        self.0.borrow().cluster_generation
    }

    /// Writes every registered driver B stream into `out` at the vertex base of its section in
    /// the pool of `scene`; texels outside `out` or of a section the pool does not hold are left
    /// as they are.
    pub(crate) fn fill_pool(&self, scene: &VgScene, out: &mut [u32]) {
        for (&(key, ordinal), packed) in &self.0.borrow().clusters {
            let Some(range) = scene.section_range(key, ordinal) else {
                continue;
            };
            let base = range.vertex_base as usize;
            let Some(end) = base.checked_add(packed.len()).filter(|&e| e <= out.len()) else {
                continue;
            };
            out[base..end].copy_from_slice(packed);
        }
    }

    /// Drops the driver A GPU buffers (they are recreated on demand).
    pub(crate) fn release(&self) {
        for entry in self.0.borrow_mut().meshes.values_mut() {
            entry.buffer = None;
        }
    }
}

/// Rows of the pool colour texture for `vertices` pool vertices, or `None` past the texture
/// limit.
pub(crate) fn pool_rows(vertices: u64) -> Option<u32> {
    let rows = vertices.div_ceil(u64::from(POOL_COLOR_WIDTH)).max(1);
    u32::try_from(rows)
        .ok()
        .filter(|&r| r <= POOL_COLOR_MAX_ROWS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packing_clamps_and_validates() {
        let packed = pack_colors("k", &[1.0, 0.0, 0.5, 1.0, 2.0, -1.0, 0.0, 0.0]).unwrap();
        assert_eq!(packed, vec![0xFF80_00FF, 0x0000_00FF]);
        assert_eq!(WHITE_TEXEL, pack_colors("k", &[1.0; 4]).unwrap()[0]);
        assert!(pack_colors("k", &[1.0; 3]).is_err());
        assert!(pack_colors("k", &[1.0, f32::NAN, 1.0, 1.0]).is_err());
    }

    #[test]
    fn pool_rows_follow_the_texture_limit() {
        assert_eq!(pool_rows(0), Some(1));
        assert_eq!(pool_rows(4096), Some(1));
        assert_eq!(pool_rows(4097), Some(2));
        assert_eq!(pool_rows(4096 * 8192), Some(8192));
        assert_eq!(pool_rows(4096 * 8192 + 1), None);
    }

    #[test]
    fn a_refused_registration_changes_nothing() {
        let colors = GiVertexColors::default();
        assert!(colors.set_cluster(1, 0, Some(&[1.0; 5])).is_err());
        assert!(!colors.has_clusters());
        colors.set_cluster(1, 0, Some(&[0.5; 8])).unwrap();
        assert!(colors.has_clusters());
        let generation = colors.cluster_generation();
        assert!(
            colors
                .set_cluster(1, 0, Some(&[1.0, f32::INFINITY, 0.0, 0.0]))
                .is_err()
        );
        assert_eq!(colors.cluster_generation(), generation);
        colors.set_cluster(1, 0, None).unwrap();
        assert!(!colors.has_clusters());
    }
}
