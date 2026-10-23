//! `ShadowFeature::install`, the `shadow.csm` feature pass (S12, order 20) and the
//! per-frame `ShadowHandle`.
//!
//! The pass and every handle clone share one [`ShadowState`]: the [`ShadowDesc`], the camera
//! and sun last given to [`ShadowHandle::set_view`] (None until the first call), the caster
//! array (created lazily at the description's resolution and recreated when it changes), the
//! `bound` flag and the preallocated 320 B `ShadowGpu` uniform. The pass computes the
//! practical cascade splits and their view-projections, renders the casters into the array
//! ([`Casters::record`]) and gives the array view and uniform to the lighting
//! (`LightingResources::set_shadow`). Without a view, or while disabled, the lighting keeps
//! the dummy shadow (`clear_shadow`, enabled 0), so the `csm` `Shadow` provider returns the
//! contact term alone.
//!
//! The shader takes the sun angular radius from `FrameGpu.sun_dir.w` (S10); `ShadowGpu.params.x`
//! is the description's softness multiplier applied to it.

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::{GpuContext, RaaError};
use raa_lit::{LitHandle, LitRenderer};
use raa_math::{Mat4, Vec3};
use raa_pipeline::{FeaturePass, LightingResources, Pipeline, PrepareCtx, RecordCtx};

use crate::caster::Casters;
use crate::desc::ShadowDesc;
use crate::provider::csm_provider;

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
/// Practical split lambda (CSM).
const SPLIT_LAMBDA: f32 = 0.75;
/// Normal-offset receiver amount in shadow-map texels (CSM).
const NORMAL_OFFSET_TEXELS: f32 = 1.5;
/// Cascade blend band as a fraction of the cascade depth range (CSM 10%).
const BLEND_BAND: f32 = 0.1;

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
    /// Preallocated `ShadowGpu` bytes, rewritten every prepare.
    uniform: [u8; SHADOW_GPU_SIZE],
}

type SharedShadow = Rc<RefCell<ShadowState>>;

/// Writes one `vec4<f32>` into `out[..16]`.
fn write_vec4(out: &mut [u8], v: [f32; 4]) {
    for (chunk, x) in out.chunks_exact_mut(4).zip(v) {
        chunk.copy_from_slice(&x.to_le_bytes());
    }
}

/// Packs `ShadowGpu` (S10): `cascade_view_proj` @0, `splits` @256, `params` @272
/// (softness, normal-offset texels, 1/resolution, enabled 1) and `params2` @288
/// (0, blend band, 0, 0). `FrameGpu.sun_dir.w` supplies the sun angular radius.
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
        [0.0, BLEND_BAND, 0.0, 0.0],
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

    /// Disabling clears the shadow map (dummy, enabled 0) and drops `bound`; enabling lets the
    /// next prepare set it again when a view exists.
    fn set_enabled(&mut self, enabled: bool, lighting: &mut LightingResources) {
        if !enabled {
            lighting.clear_shadow();
            if let Ok(mut st) = self.state.try_borrow_mut() {
                st.bound = false;
            }
        }
    }

    /// Without a view: `clear_shadow` (once) and nothing to render. Otherwise builds the
    /// cascade array at the description's resolution, computes the splits and matrices,
    /// writes the caster uniform and `ShadowGpu`, and gives the array to the lighting.
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let mut st = self.state.try_borrow_mut().map_err(|_| state_busy())?;
        let Some(view) = st.view else {
            if st.bound {
                ctx.lighting.clear_shadow();
                st.bound = false;
            }
            return Ok(());
        };
        let desc = st.desc;
        if st
            .casters
            .as_ref()
            .is_none_or(|c| c.resolution() != desc.resolution)
        {
            st.casters = Some(Casters::new(ctx.gpu, desc.resolution));
        }
        let far = view.far.min(desc.distance_m);
        let splits = crate::cascade_splits(view.near, far, SPLIT_LAMBDA, CASCADES);
        let mut matrices = [Mat4::IDENTITY; CASCADES];
        let mut previous = view.near;
        for (i, matrix) in matrices.iter_mut().enumerate() {
            *matrix = crate::cascade_matrix(
                &view.view,
                view.fov_y_rad,
                view.aspect,
                (previous, splits[i]),
                view.sun_dir,
                desc.resolution,
            );
            previous = splits[i];
        }
        let casters = st.casters.as_ref().expect("casters created above");
        casters.write(ctx.gpu, &matrices);
        let array_view = casters.array_view().clone();
        pack_shadow(&mut st.uniform, &matrices, &splits, &desc);
        ctx.lighting.set_shadow(array_view, &st.uniform);
        st.bound = true;
        Ok(())
    }

    /// Renders the caster draws into the four cascade layers (nothing without a caster array).
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let st = self.state.try_borrow().map_err(|_| state_busy())?;
        let Some(casters) = st.casters.as_ref() else {
            return Ok(());
        };
        casters.record(ctx.gpu, ctx.encoder, &self.lit)
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
            uniform: [0; SHADOW_GPU_SIZE],
        }));
        let pass = ShadowPass {
            state: state.clone(),
            lit: lit.handle(),
        };
        p.add(Box::new(pass))?;
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
        // params2 @288 = (0, blend band, 0, 0).
        assert_eq!(f32_at(&out, OFF_PARAMS2), 0.0);
        assert_eq!(f32_at(&out, OFF_PARAMS2 + 4), BLEND_BAND);
        assert_eq!(f32_at(&out, OFF_PARAMS2 + 8), 0.0);
        assert_eq!(f32_at(&out, OFF_PARAMS2 + 12), 0.0);
    }
}
