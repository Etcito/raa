//! WGSL source of the `dof` pass, its entry points and the size of its uniform.
//!
//! No `crate::` paths (only `super::DofDesc`, re-exported at the crate root) so
//! tests/variants.rs can compile it with `#[path]` next to `use raa_dof::DofDesc`.

/// Compute module (no PRELUDE), group 0 only: 0 `DofParams` uniform, 1 depth
/// (`texture_depth_2d`), 2 focus distance (read_write storage `f32`, 4 B), 3 CoC out
/// (r32float write), 4 CoC in, 5 tiles out (rg32float write), 6 tiles in, 7 dilated tiles
/// out (rg32float write). Each entry binds only the bindings its comment in the WGSL lists.
pub(crate) const DOF_COC_WGSL: &str = include_str!("wgsl/dof_coc.wgsl");

/// Autofocus, 1 thread, dispatch (1, 1, 1): bindings 0, 1, 2. Smoothed per frame with
/// `autofocus.w`; the host sends 1 on the first frame (snap).
pub(crate) const CS_AUTOFOCUS: &str = "cs_autofocus";
/// Full-res signed CoC px, [`COC_WORKGROUP`] squared threads: bindings 0, 1, 2, 3.
pub(crate) const CS_COC: &str = "cs_coc";
/// Tile min/max CoC, one workgroup per [`TILE_SIZE`] squared tile: bindings 4, 5.
pub(crate) const CS_TILES: &str = "cs_tiles";
/// 3x3 tile dilation, [`COC_WORKGROUP`] squared threads over the tile grid: bindings 6, 7.
pub(crate) const CS_DILATE: &str = "cs_dilate";

/// Workgroup edge of `cs_coc` and `cs_dilate`.
pub(crate) const COC_WORKGROUP: u32 = 8;
/// Tile edge in pixels (`cs_tiles` workgroup edge).
pub(crate) const TILE_SIZE: u32 = 16;

/// Render module (no PRELUDE), group 0 only: 0 `DofParams` uniform (the same struct as
/// [`DOF_COC_WGSL`]), 1 hdr.src() (full res), 2 CoC (full res, from [`CS_COC`]), 3 dilated
/// tiles (from [`CS_DILATE`]), 4 depth (`texture_depth_2d`), 5 near layer (half res),
/// 6 far layer (half res). All textureLoad, no sampler. Every fragment entry is drawn with
/// the fullscreen triangle of [`VS_MAIN`] (3 vertices) into `HDR_FORMAT` targets; each binds
/// only the bindings its doc lists.
pub(crate) const DOF_GATHER_WGSL: &str = include_str!("wgsl/dof_gather.wgsl");

/// Fullscreen triangle vertex entry, varying `uv` (target coordinates, (0, 0) top-left).
pub(crate) const VS_MAIN: &str = "vs_main";
/// Half res, MRT 2 targets (0 near, 1 far, premultiplied, a = coverage): 4-tap CoC-weighted
/// average of the full-res hdr. Bindings 1, 2.
pub(crate) const FS_PREFILTER: &str = "fs_prefilter";
/// Half res: 48-tap spiral gather of the prefiltered near layer (binding 5); a = coverage.
/// Bindings 2, 3, 5.
pub(crate) const FS_GATHER_NEAR: &str = "fs_gather_near";
/// Half res: depth-ordered 48-tap spiral gather of the prefiltered far layer (binding 6).
/// Bindings 2, 3, 6.
pub(crate) const FS_GATHER_FAR: &str = "fs_gather_far";
/// Full res into hdr.dst(): bilateral upsample of the gathered layers over the sharp hdr.
/// Bindings 0, 1, 2, 4, 5, 6.
pub(crate) const FS_COMPOSITE: &str = "fs_composite";
/// Taps of the golden-angle spiral of each gather.
pub(crate) const GATHER_TAPS: u32 = 48;

/// Byte size of `DofParams`: inv_proj @0 (mat4x4); lens @64 (focus s m, aperture term
/// `K = A * f / (s - f)` in px at viewport_h, max CoC px at viewport_h, camera near m =
/// lower bound of the autofocus distance); lens2 @80 (focal length f m, 0, 0, 0); viewport
/// @96 (w, h, 1/w, 1/h); autofocus @112 (u, v, enabled 0|1, blend factor per frame).
pub(crate) const PARAMS_SIZE: u64 = 128;
/// Byte offset of `DofParams.lens2` (focal length in metres in `x`).
pub(crate) const PARAMS_LENS2_OFFSET: u64 = 80;

/// `DofParams.lens.y`: the aperture term `K = A * f / (s - f)` of `d` mapped to pixels at
/// `viewport_h` (36 mm sensor), in f32 as the host writes it.
pub(crate) fn aperture_term_px(d: &super::DofDesc, viewport_h: f32) -> f32 {
    let f = d.focal_length_mm / 1000.0;
    let a = f / d.f_stop;
    a * f / (d.focus_distance_m - f) * 1000.0 / 36.0 * viewport_h
}

/// CPU twin of `cs_coc` with autofocus: the signed CoC px of depth `z` when the GPU focus
/// distance is `s_prime`. Like the shader, `s_prime` is kept beyond the focal point
/// (`max(s', f + 0.001)`), the aperture term becomes `K * (s - f) / (s' - f)` and the result
/// is clamped to `max_coc_px * viewport_h / 720`; it matches `coc_px` with
/// `focus_distance_m = s_prime`. A non-finite or non-positive `z` returns 0.
pub(crate) fn coc_px_at_focus(z: f32, d: &super::DofDesc, viewport_h: f32, s_prime: f32) -> f32 {
    if !(z > 0.0 && z < 3.0e38) {
        return 0.0;
    }
    let f = d.focal_length_mm / 1000.0;
    let s = d.focus_distance_m;
    let sp = s_prime.max(f + 0.001);
    let k = aperture_term_px(d, viewport_h) * (s - f) / (sp - f);
    let limit = d.max_coc_px * viewport_h / 720.0;
    (k * (z - sp) / z).clamp(-limit, limit)
}
