//! The HDR lit renderer (S10, S11, S12). This part holds the surface materials: the
//! [`SurfaceMaterial`] record with its 256-byte `SurfaceMaterialGpu` layout, the
//! [`DrawClass`] a material is drawn in and the [`SurfaceLibrary`] of dense material ids, the
//! per-instance [`InstanceExtra`] and the default slot providers (S11) the renderer installs.

mod extras;
mod library;
mod material;
mod slots;

pub use extras::{INSTANCE_EXTRA_GPU_SIZE, InstanceExtra};
pub use library::SurfaceLibrary;
pub use material::{
    DrawClass, FLAG_DOUBLE_SIDED, FLAG_FUR, FLAG_NO_SHADOW, FLAG_TRANSMISSIVE, FLAG_TRANSPARENT,
    FLAG_UNLIT, FLAG_VERTEX_FN, SURFACE_MATERIAL_GPU_SIZE, SurfaceMaterial, SurfaceModel,
};
pub use raa_core::MaterialId;
pub use raa_shader::MaterialFnId;
