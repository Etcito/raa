//! The HDR lit renderer (S10, S11, S12). This part holds the surface materials: the
//! [`SurfaceMaterial`] record with its 256-byte `SurfaceMaterialGpu` layout, the
//! [`DrawClass`] a material is drawn in and the [`SurfaceLibrary`] of dense material ids, the
//! per-instance [`InstanceExtra`], the GPU mesh store ([`LitMeshId`], with one internal
//! [`MeshId`] per sub-mesh), the default slot providers (S11) the renderer installs, the
//! 512-byte S10 `FrameGpu` writer ([`write_frame_gpu`]) and the CPU draw-list builder
//! ([`DrawPrep`], [`LitDraw`]).

mod extras;
mod frame;
mod library;
mod material;
mod meshes;
mod prep;
mod slots;

pub use extras::{INSTANCE_EXTRA_GPU_SIZE, InstanceExtra};
pub use frame::{FRAME_GPU_SIZE, FrameParams, SUN_ANGULAR_RADIUS_DEFAULT, write_frame_gpu};
pub use library::SurfaceLibrary;
pub use material::{
    DrawClass, FLAG_DOUBLE_SIDED, FLAG_FUR, FLAG_NO_SHADOW, FLAG_TRANSMISSIVE, FLAG_TRANSPARENT,
    FLAG_UNLIT, FLAG_VERTEX_FN, SURFACE_MATERIAL_GPU_SIZE, SurfaceMaterial, SurfaceModel,
};
pub use meshes::{LitMeshId, LitMeshInfo};
pub use prep::{DrawPrep, INSTANCE_GPU_SIZE, LitDraw, MaterialClass, PrepCapacity, PrepInput};
pub use raa_core::{MaterialId, MeshId};
pub use raa_mesh::{AttributedMesh, SubMesh};
pub use raa_shader::MaterialFnId;
