//! Shared draw-list types of the lit renderer: one [`LitDraw`] record and the
//! [`MaterialClass`] of a material id, plus the size of one S1 instance record. The
//! compacted id lists themselves are built by [`IdPrep`](crate::IdPrep), and the 128-byte
//! instance and extras records live in [`InstanceStore`](crate::InstanceStore).

use raa_core::{MaterialId, MeshId};
use raa_shader::MaterialFnId;

use crate::material::{DrawClass, SurfaceMaterial, SurfaceModel};

/// Size in bytes of one S1 instance record.
pub const INSTANCE_GPU_SIZE: usize = 128;

/// One instanced draw: `instance_count` ids from `first_instance` of the compacted id
/// stream, all with one sub-mesh and one material.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LitDraw {
    /// Prepared binding id; `IdPrep::mesh_binding` recovers the source submesh.
    pub mesh: MeshId,
    pub material: MaterialId,
    pub material_fn: MaterialFnId,
    pub class: DrawClass,
    pub first_instance: u32,
    pub instance_count: u32,
    /// View-space distance in front of the camera of the instance's world translation;
    /// for a batch, the nearest of its instances.
    pub view_depth: f32,
}

/// What a material id draws as: its class and its material function.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MaterialClass {
    pub class: DrawClass,
    pub material_fn: MaterialFnId,
}

impl MaterialClass {
    /// The class and function of `m` (flat and standard use the standard function).
    pub fn of(m: &SurfaceMaterial) -> Self {
        let material_fn = match m.model {
            SurfaceModel::Function(id) => id,
            SurfaceModel::Flat | SurfaceModel::Standard => MaterialFnId::STANDARD,
        };
        MaterialClass {
            class: m.class(),
            material_fn,
        }
    }
}
