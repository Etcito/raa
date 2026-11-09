//! The HDR lit renderer (S10, S11, S12). This part holds the surface materials: the
//! [`SurfaceMaterial`] record with its 256-byte `SurfaceMaterialGpu` layout, the
//! [`DrawClass`] a material is drawn in and the [`SurfaceLibrary`] of dense material ids, the
//! per-instance [`InstanceExtra`], the GPU mesh store ([`LitMeshId`], with one internal
//! [`MeshId`] per sub-mesh), the default slot providers (S11) the renderer installs, the
//! 768-byte S10 `FrameGpu` writer ([`write_frame_gpu`]) and the CPU id-list builder
//! ([`IdPrep`], [`LitDraw`]), and what passes and contributors share: the
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
mod frame_history;
mod handle;
mod instance_store;
mod library;
mod light_math;
mod light_stats;
mod local_light;
mod local_lights;
mod lod;
mod material;
mod material_data;
mod meshes;
mod motion_display;
mod passes;
mod prep;
mod previous_instances;
mod renderer;
mod select;
pub mod skin_bounds;
pub mod skin_draw;
pub mod skin_editor;
pub mod skin_views;
mod slots;
mod textures;
#[cfg(test)]
pub(crate) mod tile_selection;
mod tiles;
pub use skin_editor::{OverlayGeometry, REST_EDGE_STRIDE};
pub use skin_views::{
    INSTANCE_PALETTE_OFFSET, PALETTE_BLOCK, PALETTE_MATRIX_BYTES, RestMeshView, SKIN_RECORD_STRIDE,
    SKINNED_EDGE_STRIDE, SKINNED_STREAM0_STRIDE, SKINNED_TANGENT_STRIDE, SkinDraw, SkinDrawSource,
    SkinViews, SkinnedView,
};

pub use contributor::ForwardContributor;
pub use extras::{INSTANCE_EXTRA_GPU_SIZE, InstanceExtra};
pub use frame::{FRAME_GPU_SIZE, FrameParams, SUN_ANGULAR_RADIUS_DEFAULT, write_frame_gpu};
pub use handle::{LitHandle, PipelineSpec};
pub use instance_store::{InstanceStore, StoreKind, write_instance_record};
pub use library::SurfaceLibrary;
pub use light_stats::LocalLightStats;
pub use local_light::{LightHandle, LocalLight, LocalLightKind, MAX_LIGHT_HANDLE};
pub use local_lights::{LocalLights, MAX_LOCAL_LIGHTS};
pub use lod::{
    LodLevel, LodParams, LodRegistry, MAX_LOD_LEVELS, camera_level, lod_error, shadow_level,
};
pub use material::{
    DrawClass, FLAG_DOUBLE_SIDED, FLAG_FUR, FLAG_NO_SHADOW, FLAG_TRANSMISSIVE, FLAG_TRANSPARENT,
    FLAG_UNLIT, FLAG_VERTEX_FN, SURFACE_MATERIAL_GPU_SIZE, SurfaceMaterial, SurfaceModel,
};
pub use material_data::{
    DataAllocator, MATERIAL_DATA_INITIAL_WORDS, MATERIAL_DATA_RESERVED_WORDS, MaterialDataStats,
};
pub use meshes::{LitMeshId, LitMeshInfo};
pub use motion_display::install_motion_display;
pub use prep::{INSTANCE_GPU_SIZE, LitDraw, MaterialClass};
pub use raa_core::{MaterialId, MeshId};
pub use raa_mesh::{AttributedMesh, SubMesh};
pub use raa_shader::{MaterialFnId, TintMode};
pub use renderer::{FrameCounts, InstanceFrameStats, LitOptions, LitRenderer};
pub use select::{IdPrep, SelectCapacity, SelectInput};
pub use textures::{
    ATLAS_FORMAT, ATLAS_SIZE, AtlasRect, MAX_TEXTURES, ShelfPacker, TEXTURE_RECT_SIZE, TexelData,
    quantize,
};

pub use tiles::{LIGHT_TILES_WGSL, light_tiles_source};
