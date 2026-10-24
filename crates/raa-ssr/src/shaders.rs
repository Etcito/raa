//! WGSL source of the SSR passes: the inputs (Hi-Z pyramid and HDR mip chain), then the
//! trace and the composite; their entry points and the sizes of their uniforms.
//!
//! Standalone (no `crate::` paths) so tests/variants.rs can compile it with `#[path]`.
#![allow(dead_code)]

/// Module (no PRELUDE), group 0 only: 0 `HizParams` uniform, 1 `depth` (texture_depth_2d,
/// cs_hiz_copy), 2 Hi-Z level n (cs_hiz_reduce), 3 the r32float write-only storage level
/// written (both compute entries), 4 previous HDR mip (fs_downsample), 5 linear clamp
/// sampler (fs_downsample). Each pipeline's layout holds only the bindings its entry uses.
pub(crate) const SSR_HIZ_WGSL: &str = include_str!("wgsl/ssr_hiz.wgsl");

/// Copies `depth` into Hi-Z mip 0; dispatch `ceil(w / 8) x ceil(h / 8)`.
pub(crate) const CS_HIZ_COPY: &str = "cs_hiz_copy";
/// Writes Hi-Z level n + 1 from level n by the rule of `hiz::reduce_min`; dispatch over the
/// destination size in 8x8 workgroups.
pub(crate) const CS_HIZ_REDUCE: &str = "cs_hiz_reduce";
/// Compute workgroup size along x and y.
pub(crate) const HIZ_WORKGROUP: u32 = 8;
/// Fullscreen triangle vertex entry of both modules (3 vertices), varying `uv` (target
/// coordinates, (0, 0) top-left).
pub(crate) const VS_FULLSCREEN: &str = "vs_fullscreen";
/// 13-tap filtered downsample of the previous HDR mip into the next one.
pub(crate) const FS_DOWNSAMPLE: &str = "fs_downsample";

/// Byte size of `HizParams`: `size: vec4<u32>` = (src w, src h, dst w, dst h).
pub(crate) const HIZ_PARAMS_SIZE: u64 = 16;

/// Render module (no PRELUDE). Group 0 (private): 0 `SsrParams` uniform, 1 `depth`
/// (texture_depth_2d), 2 `normal`, 3 `material` (both textureLoad), 4 the Hi-Z pyramid
/// (r32float, all mips, unfilterable, textureLoad), 5 the HDR mip chain (HDR_FORMAT, all
/// mips, mip 0 a copy of hdr.src(), which the composite reads as its hdr), 6 the trace result,
/// 7 linear clamp sampler (filtering). `fs_trace` reads only 0, 1, 2 and 4, and renders into
/// the trace texture, so its bind group puts a dummy at 6. Group 1: the lighting bindings 3
/// (env_specular), 4 (`EnvSh`), 5 (brdf_lut) and 6 (linear_clamp) of
/// `LightingResources::layout()`, so the pipeline layout is [group 0, that layout].
pub(crate) const SSR_TRACE_WGSL: &str = include_str!("wgsl/ssr_trace.wgsl");

/// Full res into an Rgba16Float target: hit uv, confidence, cone mip (0 when no reflection).
pub(crate) const FS_TRACE: &str = "fs_trace";
/// Full res into hdr.dst(): the HDR mip-chain reflection swapped in for the IBL specular.
pub(crate) const FS_COMPOSITE: &str = "fs_composite";

/// Byte size of `SsrParams`: proj @0, inv_proj @64, view @128, inv_view @192 (mat4x4 each),
/// viewport @256 (w, h, 1 / w, 1 / h), params @272 (intensity, max_roughness, thickness_m,
/// Hi-Z mip count), params2 @288 (HDR mip-chain mip count, 0, 0, 0).
pub(crate) const SSR_PARAMS_SIZE: u64 = 304;
