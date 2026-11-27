//! `ShadowFeature::install`, the `shadow.csm` feature pass (S12, order 20) and the
//! per-frame `ShadowHandle`.
//!
//! The pass and every handle clone share one [`ShadowState`]: the [`ShadowDesc`], the camera
//! and sun last given to [`ShadowHandle::set_view`] (None until the first call), the caster
//! array (created lazily at the description's resolution and the renderer's hero capacity,
//! recreated when either changes), the `bound` flag and the preallocated 320 B `ShadowGpu`
//! uniform. The pass computes the practical cascade splits and their view-projections, renders
//! the casters into the array ([`Casters::record`]) and gives the array view and uniform to the
//! lighting (`LightingResources::set_shadow`). Without a view, or while disabled, the lighting
//! keeps the dummy shadow (`clear_shadow`, enabled 0), so the `csm` `Shadow` provider returns
//! the contact term alone.
//!
//! The shader takes the sun angular radius from `FrameGpu.sun_dir.w` (S10); `ShadowGpu.params.x`
//! is the description's softness multiplier applied to it.

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::{GpuContext, RaaError};
use raa_lit::{LitHandle, LitRenderer, MAX_HERO_LIGHTS};
use raa_math::{Aabb, Mat4, Vec3, Vec4};
use raa_pipeline::{
    FeaturePass, ImageDesc, ImageDim, LightingResources, Pipeline, PrepareCtx, RecordCtx,
};

use crate::cascade_access::CascadeTargets;
use crate::caster::Casters;
use crate::desc::{ShadowDesc, ShadowMode};
use crate::provider::csm_provider;
use crate::vsm::{
    BLOCKER_REACH_M, FrameInput, MARK_MARGIN_TEXELS, MAX_LEVELS, PAGE, TEXEL0_M, VsmCache, VsmStats,
};
use crate::vsm_gpu::{Job, MarkInput, VsmGpu, slot_texel};

/// S12 name of the pass.
const NAME: &str = "shadow.csm";
/// S12 order of the pass (after `sky.env`, before `prepass`).
const ORDER: u32 = 20;
/// Number of cascades (CSM).
const CASCADES: usize = 4;
/// Size in bytes of the `ShadowGpu` uniform (S10 binding 2).
const SHADOW_GPU_SIZE: usize = 320;
/// Byte offset of `ShadowGpu.splits`.
const OFF_SPLITS: usize = 256;
/// Byte offset of `ShadowGpu.params`.
const OFF_PARAMS: usize = 272;
/// Byte offset of `ShadowGpu.params2`.
const OFF_PARAMS2: usize = 288;
/// Normal-offset receiver amount in shadow-map texels (CSM).
const NORMAL_OFFSET_TEXELS: f32 = 1.5;
/// Cascade blend band as a fraction of the cascade depth range (CSM 10%).
const BLEND_BAND: f32 = 0.1;
/// Owned pass of the VSM page writes (after the window renders of `shadow.csm` and `vg.*`).
const VSM_PAGES_NAME: &str = "shadow.csm#vsm-pages";
const VSM_PAGES_ORDER: u32 = 24;
/// Owned pass of the VSM page marking (after the prepass and the VG G-buffer).
const VSM_MARK_NAME: &str = "shadow.csm#vsm-mark";
const VSM_MARK_ORDER: u32 = 34;

fn refusal(code: &'static str, message: &str) -> RaaError {
    RaaError::FrameGraph {
        code,
        message: message.to_string(),
    }
}

/// Same refusal as raa-lit's for a pipeline borrowed while a frame executes.
fn pipeline_busy() -> RaaError {
    refusal(
        "pipeline.busy",
        "the pipeline is borrowed (called while a frame executes)",
    )
}

/// `pipeline.busy` for the shared shadow state (only reachable by a re-entrant call).
fn state_busy() -> RaaError {
    refusal(
        "pipeline.busy",
        "the shadow state is borrowed (re-entrant call while a frame executes)",
    )
}

/// The frame camera and sun [`ShadowHandle::set_view`] stores.
#[derive(Copy, Clone, Debug, PartialEq)]
struct ShadowView {
    view: Mat4,
    fov_y_rad: f32,
    aspect: f32,
    near: f32,
    far: f32,
    /// World travel direction of the sun.
    sun_dir: Vec3,
}

/// State shared by the pass and every clone of a `ShadowHandle`.
struct ShadowState {
    desc: ShadowDesc,
    /// None until the first `set_view`.
    view: Option<ShadowView>,
    /// Created at the description's resolution, recreated when it changes.
    casters: Option<Casters>,
    /// The lighting holds the cascade array (false: it holds the dummy).
    bound: bool,
    /// Cascade view-projections of the last prepare.
    matrices: [Mat4; CASCADES],
    /// Hero view-projections of the last prepare, `None` for empty slots.
    heroes: [Option<Mat4>; MAX_HERO_LIGHTS],
    /// Hero layers the last prepare will render (occupied slots below the capacity).
    hero_layers: u32,
    /// View-space splits of the last prepare.
    splits: [f32; CASCADES],
    /// Sun travel direction of the last prepare.
    sun_dir: Vec3,
    /// Cascade resolution of the last prepare.
    resolution: u32,
    /// Preallocated `ShadowGpu` bytes, rewritten every prepare.
    uniform: [u8; SHADOW_GPU_SIZE],
    /// Cascade view-projections and splits of the latest frame that renders (or rendered) the
    /// cascade layers. A G frame of forward "layers-interlaced" that skips the cascades keeps
    /// them in `ShadowGpu`, so the maps its readers sample agree with the matrices.
    rendered: Option<([Mat4; CASCADES], [f32; CASCADES])>,
    /// This frame renders the cascades although `skip_cascades` is set (no rendered cascades
    /// to keep: first frame, new array).
    force_cascades: bool,
    /// What the last prepare computed (diagnostics).
    stats: ShadowStats,
    /// The camera projection of the frame (`set_projection`), `None` until given.
    proj: Option<Mat4>,
    /// VSM page cache and GPU resources (VSM mode only).
    vsm: Option<VsmRuntime>,
    /// VSM: this frame's window scissors (cascade layers 0..) and whether the coarse map
    /// (layer 3) is rendered; `vsm_active` false in CSM mode or without a plan.
    vsm_windows: Vec<[u32; 4]>,
    vsm_coarse: bool,
    vsm_active: bool,
    /// Per-slot caster bounds of the previous frame and the scratch of this frame.
    prev_bounds: Vec<Option<Aabb>>,
    cur_bounds: Vec<Option<Aabb>>,
    changed: Vec<Aabb>,
}

/// The VSM page cache and its GPU side.
struct VsmRuntime {
    cache: VsmCache,
    gpu: VsmGpu,
}

/// The cascades the last `shadow.csm` prepare computed (diagnostics, `shadowStats` in JS).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ShadowStats {
    /// The description's `cover` flag.
    pub cover: bool,
    /// View-space far distance of each cascade.
    pub splits: [f32; CASCADES],
    /// How each cascade box was fitted.
    pub fits: [crate::CascadeFit; CASCADES],
    /// World size of one texel of each cascade, metres.
    pub texels_m: [f32; CASCADES],
    /// Cover distance of the scene (farthest bounds corner, laddered), `None` without bounds.
    pub cover_m: Option<f32>,
    /// The scene bounds used (`None` when `cover` is off or the scene has no bounds).
    pub scene: Option<raa_math::Aabb>,
    /// The technique of the last prepare.
    pub mode: ShadowMode,
    /// VSM page statistics (VSM mode).
    pub vsm: Option<VsmStats>,
    /// Bytes of the shadow array (cascades, hero layers and the VSM pool) plus the VSM buffers.
    pub allocated_bytes: u64,
}

impl Default for ShadowStats {
    fn default() -> Self {
        ShadowStats {
            cover: false,
            splits: [0.0; CASCADES],
            fits: [crate::CascadeFit::Slice; CASCADES],
            texels_m: [0.0; CASCADES],
            cover_m: None,
            scene: None,
            mode: ShadowMode::Csm,
            vsm: None,
            allocated_bytes: 0,
        }
    }
}

type SharedShadow = Rc<RefCell<ShadowState>>;

/// Writes one `vec4<f32>` into `out[..16]`.
fn write_vec4(out: &mut [u8], v: [f32; 4]) {
    for (chunk, x) in out.as_chunks_mut::<4>().0.iter_mut().zip(v) {
        chunk.copy_from_slice(&x.to_le_bytes());
    }
}

/// Packs `ShadowGpu` (S10): `cascade_view_proj` @0, `splits` @256, `params` @272
/// (softness, normal-offset texels, 1/resolution, enabled 1) and `params2` @288
/// (minimum filter radius m, blend band, cascade count, 0). `FrameGpu.sun_dir.w` supplies the sun angular radius.
fn pack_shadow(
    out: &mut [u8; SHADOW_GPU_SIZE],
    matrices: &[Mat4; CASCADES],
    splits: &[f32; 4],
    desc: &ShadowDesc,
) {
    for (i, m) in matrices.iter().enumerate() {
        m.write_le_bytes(&mut out[i * 64..i * 64 + 64]);
    }
    write_vec4(&mut out[OFF_SPLITS..OFF_PARAMS], *splits);
    write_vec4(
        &mut out[OFF_PARAMS..OFF_PARAMS2],
        [
            desc.softness,
            NORMAL_OFFSET_TEXELS,
            1.0 / desc.resolution as f32,
            1.0,
        ],
    );
    write_vec4(
        &mut out[OFF_PARAMS2..OFF_PARAMS2 + 16],
        [desc.filter_m, BLEND_BAND, desc.cascades as f32, 0.0],
    );
}

/// The `shadow.csm` feature pass.
struct ShadowPass {
    state: SharedShadow,
    /// The lit renderer's draws and pipelines, used to record the casters.
    lit: LitHandle,
}

impl FeaturePass for ShadowPass {
    fn name(&self) -> &str {
        NAME
    }

    fn order(&self) -> u32 {
        ORDER
    }

    /// Disabling clears the shadow map (dummy, enabled 0), drops `bound` and resets the hero
    /// layer count; enabling lets the next prepare set it again when a view exists.
    fn set_enabled(&mut self, enabled: bool, lighting: &mut LightingResources) {
        if !enabled {
            lighting.clear_shadow();
            if let Ok(mut st) = self.state.try_borrow_mut() {
                st.bound = false;
                st.heroes = [None; MAX_HERO_LIGHTS];
                st.hero_layers = 0;
                st.rendered = None;
                // Casters may move while disabled: the cached pages are dropped.
                st.vsm = None;
                st.vsm_active = false;
            }
        }
    }

    /// Without a view: `clear_shadow` (once) and nothing to render. Otherwise builds the
    /// cascade array at the description's resolution, computes the splits and matrices,
    /// writes the caster uniform and `ShadowGpu`, and gives the array to the lighting.
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let mut st = self.state.try_borrow_mut().map_err(|_| state_busy())?;
        st.vsm_active = false;
        let Some(view) = st.view else {
            if st.bound {
                ctx.lighting.clear_shadow();
                st.bound = false;
            }
            st.rendered = None;
            return Ok(());
        };
        let desc = st.desc;
        if desc.mode == ShadowMode::Vsm {
            return prepare_vsm(&self.lit, ctx, &mut st, view);
        }
        st.vsm = None;
        // `LitRenderer::prepare` wrote the area-light hero capacity and targets this frame
        // (see the comment below); recreate the array when either the resolution or the
        // number of hero layers it holds changes.
        let hero_capacity = self.lit.area_hero_capacity().min(MAX_HERO_LIGHTS as u32);
        let hero_targets = self.lit.area_hero_targets();
        if st.casters.as_ref().is_none_or(|c| {
            c.resolution() != desc.resolution
                || c.hero_layers() != hero_capacity
                || c.pool_layers() != 0
        }) {
            st.casters = Some(Casters::new(ctx.gpu, desc.resolution, hero_capacity, 0));
            st.rendered = None;
        }
        // `cover`: the scene bounds keep the last cascade over the whole scene (no sun without
        // shadow inside it); otherwise the plain range min(far, distance).
        let scene = if desc.cover {
            self.lit.scene_bounds()
        } else {
            None
        };
        let frame = crate::cascade_frame_covered_n(
            &view.view,
            view.fov_y_rad,
            view.aspect,
            view.near,
            view.far,
            desc.distance_m,
            view.sun_dir,
            desc.resolution,
            scene.as_ref(),
            desc.cascades as usize,
            desc.split_lambda,
            desc.spread,
        );
        // A G frame of forward "layers-interlaced" that skips the cascades keeps the matrices
        // and splits the cascade layers were last rendered with (OIT, fur shells and opaque
        // particles sample those maps on that frame); it renders fresh ones only when there is
        // nothing to keep.
        let skip = ctx.lighting.interlace().skip_cascades;
        let kept = st.rendered.filter(|_| skip);
        st.force_cascades = skip && kept.is_none();
        let (matrices, splits) = kept.unwrap_or((frame.matrices, frame.splits));
        if kept.is_none() {
            st.rendered = Some((matrices, splits));
        }
        st.stats = ShadowStats {
            cover: desc.cover,
            splits: frame.splits,
            fits: frame.fits,
            texels_m: frame.texels_m,
            cover_m: frame.cover_m,
            scene,
            mode: ShadowMode::Csm,
            vsm: None,
            allocated_bytes: st.casters.as_ref().map_or(0, Casters::allocated_bytes),
        };
        // Caster list `i` uses `all[i]`: the four cascades, then each hero target of a slot
        // below the capacity (identity for an empty slot).
        let mut all = [Mat4::IDENTITY; CASCADES + MAX_HERO_LIGHTS];
        all[..CASCADES].copy_from_slice(&matrices);
        let mut heroes = [None; MAX_HERO_LIGHTS];
        for (h, target) in hero_targets.iter().take(hero_capacity as usize).enumerate() {
            if let Some(matrix) = target {
                all[CASCADES + h] = *matrix;
                heroes[h] = *target;
            }
        }
        // `LitRenderer::prepare` has already run this frame: `WebRuntime::update`
        // (crates/raa-js/src/engine.rs) prepares the lit frame before the frame graph runs
        // `raa.pipeline`, whose pass prepares include this one. The shared scene snapshot and
        // camera id section are therefore current when the cascades select their casters, and
        // the hero targets it wrote are the ones this prepare renders.
        let occupied = heroes
            .iter()
            .rposition(Option::is_some)
            .map_or(0, |slot| slot + 1);
        let used = CASCADES + occupied;
        self.lit
            .select_casters(ctx.gpu, &all[..used], desc.resolution)?;
        let Some(casters) = st.casters.as_ref() else {
            return Ok(());
        };
        casters.write(ctx.gpu, &all);
        let array_view = casters.array_view().clone();
        // S55 `shadowMap`: the array as allocated, its cascades plus the hero layers it holds.
        let size = casters.resolution();
        let layers = CASCADES as u32 + casters.hero_layers();
        let map_desc = ImageDesc::new(
            ImageDim::D2Array,
            wgpu::TextureFormat::Depth32Float,
            size,
            size,
            layers,
            1,
        );
        ctx.resources
            .publish(NAME, "shadowMap", &array_view, map_desc);
        pack_shadow(&mut st.uniform, &matrices, &splits, &desc);
        ctx.lighting.set_shadow(array_view, &st.uniform);
        st.matrices = matrices;
        st.heroes = heroes;
        st.hero_layers = heroes.iter().filter(|target| target.is_some()).count() as u32;
        st.splits = splits;
        st.sun_dir = view.sun_dir;
        st.resolution = desc.resolution;
        st.bound = true;
        Ok(())
    }

    /// Renders the caster draws into the four cascade layers and each occupied hero layer
    /// (nothing without a caster array). On a G frame of forward "layers-interlaced" whose
    /// cascade readers allow it (`LightingInterlace::skip_cascades`, raa-lit) only the hero
    /// layers, which the local lighting reads every frame, are rendered; the cascade layers and
    /// `ShadowGpu` keep the last rendered cascades (see `prepare`).
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let st = self.state.try_borrow().map_err(|_| state_busy())?;
        let Some(casters) = st.casters.as_ref() else {
            return Ok(());
        };
        if st.vsm_active {
            return casters.record_vsm(
                ctx.gpu,
                ctx.encoder,
                &self.lit,
                &st.heroes,
                &st.vsm_windows,
                st.vsm_coarse,
            );
        }
        let cascades = if !ctx.lighting.interlace().skip_cascades || st.force_cascades {
            st.desc.cascades.clamp(1, CASCADES as u32) as usize
        } else {
            0
        };
        casters.record(ctx.gpu, ctx.encoder, &self.lit, &st.heroes, cascades)
    }
}

/// Packs `ShadowGpu` for the virtual shadow map (`params.w` 2; layout in `wgsl/vsm.wgsl`).
#[allow(clippy::too_many_arguments, reason = "the VSM uniform fields")]
fn pack_vsm(
    out: &mut [u8; SHADOW_GPU_SIZE],
    desc: &ShadowDesc,
    light: Mat4,
    origins: &[[i32; 2]; MAX_LEVELS as usize],
    coarse: Mat4,
    levels: u32,
    pool_base: u32,
    pages_per_side: u32,
    pixel_angle: f32,
    footprint_m: f32,
    depth_range_m: f32,
) {
    let mut o = [0.0f32; 16];
    for (l, origin) in origins.iter().enumerate() {
        o[2 * l] = origin[0] as f32;
        o[2 * l + 1] = origin[1] as f32;
    }
    let col = |i: usize| Vec4::new(o[4 * i], o[4 * i + 1], o[4 * i + 2], o[4 * i + 3]);
    let origins_m = Mat4::from_cols(col(0), col(1), col(2), col(3));
    let params_m = Mat4::from_cols(
        Vec4::new(
            TEXEL0_M,
            levels as f32,
            pool_base as f32,
            pages_per_side as f32,
        ),
        Vec4::new(pixel_angle, desc.vsm_bias, BLOCKER_REACH_M, depth_range_m),
        Vec4::new(footprint_m, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, 0.0),
    );
    for (i, m) in [light, origins_m, params_m, coarse].iter().enumerate() {
        m.write_le_bytes(&mut out[i * 64..i * 64 + 64]);
    }
    // Shafts and the coarse fallback read cascade 3 for every depth.
    write_vec4(&mut out[OFF_SPLITS..OFF_PARAMS], [0.0, 0.0, 0.0, 1.0e9]);
    write_vec4(
        &mut out[OFF_PARAMS..OFF_PARAMS2],
        [
            desc.softness,
            NORMAL_OFFSET_TEXELS,
            1.0 / desc.resolution as f32,
            2.0,
        ],
    );
    write_vec4(
        &mut out[OFF_PARAMS2..OFF_PARAMS2 + 16],
        [desc.filter_m, BLEND_BAND, CASCADES as f32, 0.0],
    );
}

/// VSM prepare: caster array with the page pool, the newest page requests, the plan of this
/// frame (windows, page copies, page tables), the caster lists (windows 0..2, coarse map 3,
/// heroes 4..), the page-write jobs, the marking uniform and `ShadowGpu`.
fn prepare_vsm(
    lit: &LitHandle,
    ctx: &mut PrepareCtx<'_>,
    st: &mut ShadowState,
    view: ShadowView,
) -> Result<(), RaaError> {
    ensure_vsm(lit, ctx.gpu, st);
    let (Some(mut rt), Some(casters)) = (st.vsm.take(), st.casters.take()) else {
        return Ok(());
    };
    let result = prepare_vsm_frame(lit, ctx, st, view, &mut rt, &casters);
    st.vsm = Some(rt);
    st.casters = Some(casters);
    result
}

/// Ensures the caster array (with the page pool) and the VSM runtime for the description.
fn ensure_vsm(lit: &LitHandle, gpu: &GpuContext, st: &mut ShadowState) {
    let desc = st.desc;
    let res = desc.resolution;
    let pps = (res / PAGE).max(1);
    let per_layer = pps * pps;
    let pool_layers = (desc.vsm_pages + MAX_LEVELS).div_ceil(per_layer);
    let slots = pool_layers * per_layer;
    let hero_capacity = lit.area_hero_capacity().min(MAX_HERO_LIGHTS as u32);
    let new_array = st.casters.as_ref().is_none_or(|c| {
        c.resolution() != res || c.hero_layers() != hero_capacity || c.pool_layers() != pool_layers
    });
    if new_array {
        st.casters = Some(Casters::new(gpu, res, hero_capacity, pool_layers));
        st.rendered = None;
    }
    let rebuild = new_array
        || st
            .vsm
            .as_ref()
            .is_none_or(|v| v.cache.slots() != slots || v.cache.window_pages() != pps as i32);
    if rebuild {
        let mut vsm_gpu = match st.vsm.take() {
            Some(v) => v.gpu,
            None => VsmGpu::new(gpu),
        };
        vsm_gpu.invalidate_page_group();
        st.vsm = Some(VsmRuntime {
            cache: VsmCache::new(slots, res),
            gpu: vsm_gpu,
        });
        st.prev_bounds.clear();
    }
}

#[allow(clippy::too_many_lines, reason = "one VSM frame, in order")]
fn prepare_vsm_frame(
    lit: &LitHandle,
    ctx: &mut PrepareCtx<'_>,
    st: &mut ShadowState,
    view: ShadowView,
    vsm: &mut VsmRuntime,
    casters: &Casters,
) -> Result<(), RaaError> {
    let desc = st.desc;
    let res = desc.resolution;
    let pps = (res / PAGE).max(1);
    let hero_capacity = lit.area_hero_capacity().min(MAX_HERO_LIGHTS as u32);
    let hero_targets = lit.area_hero_targets();
    let proj = st.proj.unwrap_or_else(|| {
        Mat4::perspective_rh_zo(view.fov_y_rad, view.aspect, view.near, view.far)
    });
    if let Some((frame, origins, words)) = vsm.gpu.take_requests() {
        let latency = ctx.frame.saturating_sub(frame).min(u64::from(u32::MAX)) as u32;
        vsm.cache.apply_requests(&words, &origins, latency);
    }
    st.changed.clear();
    lit.caster_changes(&mut st.prev_bounds, &mut st.cur_bounds, &mut st.changed);
    let inv_view = view.view.inverse().unwrap_or(Mat4::IDENTITY);
    let eye = inv_view.transform_point(Vec3::ZERO);
    let scene = lit.scene_bounds();
    vsm.cache.plan(
        &FrameInput {
            sun_dir: view.sun_dir,
            eye,
            scene,
            changed: &st.changed,
            budget: desc.vsm_budget,
        },
        res,
    );
    let cache = &vsm.cache;
    let light = cache.light().copied();

    // Caster lists: windows 0..2, the coarse map 3, heroes 4.. (a list that renders
    // nothing gets a box far outside the scene).
    let empty = light.map_or(Mat4::IDENTITY, |l| crate::vsm::empty_window(&l));
    let mut all = [empty; CASCADES + MAX_HERO_LIGHTS];
    for (i, w) in cache.windows.iter().enumerate() {
        all[i] = w.matrix;
    }
    let coarse = cache.coarse_matrix().unwrap_or(empty);
    all[3] = coarse;
    let mut heroes = [None; MAX_HERO_LIGHTS];
    for (h, target) in hero_targets.iter().take(hero_capacity as usize).enumerate() {
        if let Some(matrix) = target {
            all[CASCADES + h] = *matrix;
            heroes[h] = *target;
        }
    }
    let occupied = heroes
        .iter()
        .rposition(Option::is_some)
        .map_or(0, |slot| slot + 1);
    lit.select_casters(ctx.gpu, &all[..CASCADES + occupied], res)?;
    casters.write(ctx.gpu, &all);

    // Page writes: the window page copies, then the dirty page tables (slot = level).
    let mut jobs: Vec<(u32, Job)> = cache
        .copies
        .iter()
        .map(|c| {
            let (layer, dst) = slot_texel(c.slot, pps);
            (
                layer,
                Job {
                    src: c.src,
                    dst,
                    src_layer: c.window,
                    kind: 0,
                    level: 0,
                },
            )
        })
        .collect();
    let dirty = cache.table_dirty();
    for level in 0..cache.levels() {
        if dirty & (1 << level) == 0 {
            continue;
        }
        vsm.gpu.write_table(ctx.gpu, level, cache.table(level));
        let (layer, dst) = slot_texel(level, pps);
        jobs.push((
            layer,
            Job {
                src: (0, 0),
                dst,
                src_layer: 0,
                kind: 1,
                level,
            },
        ));
    }
    jobs.sort_by_key(|(layer, _)| *layer);
    vsm.gpu.write_jobs(ctx.gpu, &jobs);

    // Marking: the prepass depth unprojected with this frame's camera.
    let (_, height) = ctx.targets.size();
    let ortho = proj.cols[3][3] == 1.0;
    let pixel_angle = if ortho {
        0.0
    } else {
        2.0 / (proj.cols[1][1].abs().max(1e-6) * height.max(1) as f32)
    };
    let footprint_m = if ortho {
        2.0 / (proj.cols[1][1].abs().max(1e-6) * height.max(1) as f32)
    } else {
        0.0
    };
    let origins = cache.origins();
    if let Some(l) = light {
        let view_proj = proj * view.view;
        let v = view.view;
        vsm.gpu.write_mark(
            ctx.gpu,
            &MarkInput {
                inv_view_proj: view_proj.inverse().unwrap_or(Mat4::IDENTITY),
                light: l.matrix(),
                origins,
                view_z: [v.cols[0][2], v.cols[1][2], v.cols[2][2], v.cols[3][2]],
                texel0: TEXEL0_M,
                levels: cache.levels(),
                pixel_angle,
                footprint: footprint_m,
                bias: desc.vsm_bias,
                margin: MARK_MARGIN_TEXELS,
            },
        );
        vsm.gpu.mark = Some((ctx.frame, origins));
    }

    let array_view = casters.array_view().clone();
    let layers = CASCADES as u32 + casters.hero_layers() + casters.pool_layers();
    let map_desc = ImageDesc::new(
        ImageDim::D2Array,
        wgpu::TextureFormat::Depth32Float,
        res,
        res,
        layers,
        1,
    );
    ctx.resources
        .publish(NAME, "shadowMap", &array_view, map_desc);
    let depth_range = light.map_or(1.0, |l| l.depth1 - l.depth0);
    let levels = if light.is_some() { cache.levels() } else { 0 };
    pack_vsm(
        &mut st.uniform,
        &desc,
        light.map_or(Mat4::IDENTITY, |l| l.matrix()),
        &origins,
        coarse,
        levels,
        casters.pool_base(),
        pps,
        pixel_angle,
        footprint_m,
        depth_range,
    );
    ctx.lighting.set_shadow(array_view, &st.uniform);
    st.vsm_windows.clear();
    st.vsm_windows
        .extend(cache.windows.iter().map(|w| w.scissor));
    st.vsm_coarse = cache.coarse_dirty;
    st.vsm_active = true;
    st.matrices = [all[0], all[1], all[2], all[3]];
    st.splits = [view.near; CASCADES];
    st.heroes = heroes;
    st.hero_layers = heroes.iter().filter(|target| target.is_some()).count() as u32;
    st.sun_dir = view.sun_dir;
    st.resolution = res;
    st.bound = true;
    st.stats = ShadowStats {
        cover: desc.cover,
        splits: [0.0; CASCADES],
        fits: [crate::CascadeFit::Scene; CASCADES],
        texels_m: [0.0; CASCADES],
        cover_m: None,
        scene,
        mode: ShadowMode::Vsm,
        vsm: Some(cache.stats),
        allocated_bytes: casters.allocated_bytes() + vsm.gpu.allocated_bytes(),
    };
    Ok(())
}

/// The owned pass `shadow.csm#vsm-pages` (order 24): the VSM page copies and page tables of
/// this frame into the pool layers, after the window renders of `shadow.csm` and `vg.*`.
struct VsmPagesPass {
    state: SharedShadow,
}

impl FeaturePass for VsmPagesPass {
    fn name(&self) -> &str {
        VSM_PAGES_NAME
    }

    fn order(&self) -> u32 {
        VSM_PAGES_ORDER
    }

    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let mut st = self.state.try_borrow_mut().map_err(|_| state_busy())?;
        if !st.vsm_active {
            return Ok(());
        }
        let ShadowState { vsm, casters, .. } = &mut *st;
        let (Some(vsm), Some(casters)) = (vsm.as_mut(), casters.as_ref()) else {
            return Ok(());
        };
        vsm.gpu
            .record_pages(ctx.gpu, ctx.encoder, casters.scratch_view(), |i| {
                casters.pool_view(i)
            });
        Ok(())
    }
}

/// The owned pass `shadow.csm#vsm-mark` (order 34): the VSM page requests of the visible
/// pixels from the prepass depth, read back asynchronously.
struct VsmMarkPass {
    state: SharedShadow,
    /// The depth target bound by `group` (recreated with the targets).
    depth: Option<wgpu::TextureView>,
    group: Option<(u64, wgpu::BindGroup)>,
}

impl FeaturePass for VsmMarkPass {
    fn name(&self) -> &str {
        VSM_MARK_NAME
    }

    fn order(&self) -> u32 {
        VSM_MARK_ORDER
    }

    fn resize(&mut self, _gpu: &GpuContext, targets: &raa_pipeline::Targets) {
        self.depth = Some(targets.depth().clone());
        self.group = None;
    }

    fn release(&mut self, _gpu: &GpuContext) {
        self.depth = None;
        self.group = None;
    }

    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let mut st = self.state.try_borrow_mut().map_err(|_| state_busy())?;
        if !st.vsm_active {
            return Ok(());
        }
        let Some(vsm) = st.vsm.as_mut() else {
            return Ok(());
        };
        if self.depth.is_none() {
            self.depth = Some(ctx.targets.depth().clone());
            self.group = None;
        }
        let Some(depth) = self.depth.as_ref() else {
            return Ok(());
        };
        let id = vsm.gpu.id();
        if self.group.as_ref().is_none_or(|(g, _)| *g != id) {
            self.group = Some((id, vsm.gpu.mark_group(ctx.gpu, depth)));
        }
        let Some((_, group)) = self.group.as_ref() else {
            return Ok(());
        };
        vsm.gpu.record_mark(ctx.encoder, group, ctx.targets.size());
        Ok(())
    }
}

/// Installs cascaded soft sun shadows on a pipeline.
pub struct ShadowFeature;

impl ShadowFeature {
    /// Adds the `Shadow` slot provider `csm` (priority 10) to `lit` and the FeaturePass
    /// `shadow.csm` (order 20) to `pipeline`. The pass creates its cascade array lazily at the
    /// description's resolution once a view is set; until then the lighting keeps the dummy
    /// shadow and the `csm` provider returns the contact term alone.
    ///
    /// Virtualized-geometry instances do not cast into hero layers: they are excluded from
    /// the lit caster sections by `virtual_bits`, and raa-vg draws only into the 4 cascades.
    ///
    /// Refusals (nothing installed): `pipeline.busy` when the pipeline is borrowed, the
    /// `add_slot` refusal with its code (`shader.slot_conflict` when installed twice), and the
    /// `Pipeline::add` refusal (`pipeline.duplicate_pass`).
    pub fn install(
        _gpu: &GpuContext,
        pipeline: &Rc<RefCell<Pipeline>>,
        lit: &mut LitRenderer,
    ) -> Result<ShadowHandle, RaaError> {
        let mut p = pipeline.try_borrow_mut().map_err(|_| pipeline_busy())?;
        lit.add_slot(csm_provider())?;
        let state = Rc::new(RefCell::new(ShadowState {
            desc: ShadowDesc::default(),
            view: None,
            casters: None,
            bound: false,
            matrices: [Mat4::IDENTITY; CASCADES],
            heroes: [None; MAX_HERO_LIGHTS],
            hero_layers: 0,
            splits: [0.0; CASCADES],
            sun_dir: Vec3::ZERO,
            resolution: 0,
            uniform: [0; SHADOW_GPU_SIZE],
            rendered: None,
            force_cascades: false,
            stats: ShadowStats::default(),
            proj: None,
            vsm: None,
            vsm_windows: Vec::new(),
            vsm_coarse: false,
            vsm_active: false,
            prev_bounds: Vec::new(),
            cur_bounds: Vec::new(),
            changed: Vec::new(),
        }));
        let pass = ShadowPass {
            state: state.clone(),
            lit: lit.handle(),
        };
        p.add(Box::new(pass))?;
        p.add(Box::new(VsmPagesPass {
            state: state.clone(),
        }))?;
        p.add(Box::new(VsmMarkPass {
            state: state.clone(),
            depth: None,
            group: None,
        }))?;
        Ok(ShadowHandle { state })
    }
}

/// Access to the installed shadow map (clones share one state).
#[derive(Clone)]
pub struct ShadowHandle {
    state: SharedShadow,
}

impl ShadowHandle {
    /// Replaces the description after `ShadowDesc::validate` (`shadow.invalid`; the stored
    /// description is unchanged on refusal). A new resolution recreates the caster array at
    /// the next prepare. `pipeline.busy` on a re-entrant call.
    pub fn set(&self, d: ShadowDesc) -> Result<(), RaaError> {
        d.validate()?;
        self.state.try_borrow_mut().map_err(|_| state_busy())?.desc = d;
        Ok(())
    }

    /// The current description (the defaults until set).
    pub fn get(&self) -> ShadowDesc {
        self.state
            .try_borrow()
            .map(|st| st.desc)
            .unwrap_or_default()
    }

    /// The cascades the pass prepared this frame: `Some` only while the state is bound (the
    /// pass ran with a view and `shadow.csm` is enabled). `None` when disabled (also before
    /// any prepare), without a view, or on a re-entrant borrow.
    pub fn cascade_targets(&self) -> Option<CascadeTargets> {
        let st = self.state.try_borrow().ok()?;
        if !st.bound {
            return None;
        }
        let casters = st.casters.as_ref()?;
        if st.vsm_active {
            // VSM: the windows rendered this frame (layers 0..), and the coarse map (layer 3)
            // when it is rendered; unused windows hold a matrix that selects nothing.
            let count = if st.vsm_coarse {
                CASCADES as u32
            } else {
                st.vsm_windows.len() as u32
            };
            if count == 0 {
                return None;
            }
            return Some(CascadeTargets {
                count,
                resolution: st.resolution,
                view_proj: st.matrices,
                splits: st.splits,
                layer_views: casters.layer_views().clone(),
                sun_dir: st.sun_dir,
                receivers: false,
            });
        }
        Some(CascadeTargets {
            count: st.desc.cascades,
            resolution: st.resolution,
            view_proj: st.matrices,
            splits: st.splits,
            layer_views: casters.layer_views().clone(),
            sun_dir: st.sun_dir,
            receivers: true,
        })
    }

    /// What the last prepare computed (defaults before any prepare or on a re-entrant borrow).
    pub fn stats(&self) -> ShadowStats {
        self.state
            .try_borrow()
            .map(|st| st.stats)
            .unwrap_or_default()
    }

    /// Number of hero shadow-map layers the last prepare renders: the occupied hero slots
    /// below the renderer's area-light hero capacity. `0` while unbound (no view, disabled, or
    /// a re-entrant borrow). [`CascadeTargets`] is unchanged (4 cascades, 4 layer views).
    pub fn hero_layers(&self) -> u32 {
        let Ok(st) = self.state.try_borrow() else {
            return 0;
        };
        if st.bound { st.hero_layers } else { 0 }
    }

    /// The frame's camera projection (`Camera::projection`, called every frame after
    /// [`set_view`](Self::set_view)): the VSM page marking unprojects the prepass depth with it
    /// (without it, the perspective of `set_view`'s field of view is used).
    pub fn set_projection(&self, proj: &Mat4) {
        if let Ok(mut st) = self.state.try_borrow_mut() {
            st.proj = Some(*proj);
        }
    }

    /// The frame's camera and sun (called every frame): the view matrix, vertical field of
    /// view, aspect, near and far planes, and the world travel direction of the sun. Ignored
    /// on a re-entrant call.
    pub fn set_view(
        &self,
        view: &Mat4,
        fov_y_rad: f32,
        aspect: f32,
        near: f32,
        far: f32,
        sun_dir: Vec3,
    ) {
        if let Ok(mut st) = self.state.try_borrow_mut() {
            st.view = Some(ShadowView {
                view: *view,
                fov_y_rad,
                aspect,
                near,
                far,
                sun_dir,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_at(bytes: &[u8], offset: usize) -> f32 {
        f32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ])
    }

    #[test]
    fn pack_shadow_puts_splits_at_256_and_params_at_272() {
        let matrices = [Mat4::from_scale(Vec3::new(1.0, 2.0, 3.0)); CASCADES];
        let splits = [12.5, 25.0, 40.0, 60.0];
        let desc = ShadowDesc {
            softness: 2.0,
            distance_m: 60.0,
            resolution: 1024,
            cover: true,
            cascades: 4,
            filter_m: 0.02,
            split_lambda: 0.75,
            spread: true,
            ..Default::default()
        };
        let mut out = [0u8; SHADOW_GPU_SIZE];
        pack_shadow(&mut out, &matrices, &splits, &desc);
        assert_eq!(out.len(), SHADOW_GPU_SIZE);
        // cascade_view_proj @0: the first matrix's first column x.
        assert_eq!(f32_at(&out, 0), 1.0);
        // splits @256.
        for (i, s) in splits.iter().enumerate() {
            assert_eq!(f32_at(&out, OFF_SPLITS + 4 * i), *s);
        }
        // params @272 = (softness, 1.5, 1/resolution, enabled).
        assert_eq!(f32_at(&out, OFF_PARAMS), 2.0);
        assert_eq!(f32_at(&out, OFF_PARAMS + 4), NORMAL_OFFSET_TEXELS);
        assert_eq!(f32_at(&out, OFF_PARAMS + 8), 1.0 / 1024.0);
        assert_eq!(f32_at(&out, OFF_PARAMS + 12), 1.0);
        // params2 @288 = (filter m, blend band, cascades, 0).
        assert_eq!(f32_at(&out, OFF_PARAMS2), 0.02);
        assert_eq!(f32_at(&out, OFF_PARAMS2 + 4), BLEND_BAND);
        assert_eq!(f32_at(&out, OFF_PARAMS2 + 8), 4.0);
        assert_eq!(f32_at(&out, OFF_PARAMS2 + 12), 0.0);
    }
}
