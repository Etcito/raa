//! The HDR lit renderer (S10, S11, S12). This part holds the surface materials: the
//! [`SurfaceMaterial`] record with its 256-byte `SurfaceMaterialGpu` layout, the
//! [`DrawClass`] a material is drawn in and the [`SurfaceLibrary`] of dense material ids, the
//! per-instance [`InstanceExtra`], the GPU mesh store ([`LitMeshId`], with one internal
//! [`MeshId`] per sub-mesh), the default slot providers (S11) the renderer installs, the
//! 512-byte S10 `FrameGpu` writer ([`write_frame_gpu`]) and the CPU draw-list builder
//! ([`DrawPrep`], [`LitDraw`]), and what passes and contributors share: the
//! [`LitHandle`] with its pipeline cache per shader variant and [`PipelineSpec`], and the
//! [`ForwardContributor`] extension point of the forward pass. [`LitRenderer`] owns all of
//! it, prepares each frame ([`LitOptions`], [`FrameCounts`]) and registers the two required
//! feature passes of the lit pipeline (S12): `prepass` (order 30: Opaque and Fur draws,
//! single sample, into the `depth`, `normal` and `material` targets) and `forward` (order
//! 60: the same draws into MSAA HDR colour and depth, then the enabled contributors,
//! resolved into `hdr.dst()`).

mod contributor;
mod editor;
pub use editor::WireframeMode;
mod extras;
mod frame;
mod handle;
mod library;
mod material;
mod meshes;
mod passes;
mod prep;
mod renderer;
mod slots;
mod textures;
pub mod skin_views;
pub mod skin_draw;
pub mod skin_bounds;
pub mod skin_editor;
pub use skin_views::{RestMeshView, SkinViews, SkinnedView, SKINNED_EDGE_STRIDE, SKINNED_STREAM0_STRIDE, SKINNED_TANGENT_STRIDE};
pub use skin_editor::{OverlayGeometry, REST_EDGE_STRIDE};

pub use contributor::ForwardContributor;
pub use extras::{INSTANCE_EXTRA_GPU_SIZE, InstanceExtra};
pub use frame::{FRAME_GPU_SIZE, FrameParams, SUN_ANGULAR_RADIUS_DEFAULT, write_frame_gpu};
pub use handle::{LitHandle, PipelineSpec};
pub use library::SurfaceLibrary;
pub use material::{
    DrawClass, FLAG_DOUBLE_SIDED, FLAG_FUR, FLAG_NO_SHADOW, FLAG_TRANSMISSIVE, FLAG_TRANSPARENT,
    FLAG_UNLIT, FLAG_VERTEX_FN, SURFACE_MATERIAL_GPU_SIZE, SurfaceMaterial, SurfaceModel,
};
pub use meshes::{LitMeshId, LitMeshInfo};
pub use prep::{DrawPrep, INSTANCE_GPU_SIZE, LitDraw, MaterialClass, PrepCapacity, PrepInput};
pub use raa_core::{MaterialId, MeshId};
pub use raa_mesh::{AttributedMesh, SubMesh};
pub use raa_shader::{MaterialFnId, TintMode};
pub use renderer::{FrameCounts, LitOptions, LitRenderer};
pub use textures::{
    ATLAS_FORMAT, ATLAS_SIZE, AtlasRect, MAX_TEXTURES, ShelfPacker, TEXTURE_RECT_SIZE, TexelData,
    quantize,
};
