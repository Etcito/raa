//! The GI part `ssgi` (milestone M8): screen-space indirect diffuse from a visibility bitmask. For
//! each pixel, every slice of the view plane (`SsgiQuality::slices`) marches the depth buffer's
//! near field, `cpu::RADIUS_METRES` (2 m), `SsgiQuality::steps` times per side. Each sample sets
//! the 32 sectors it occludes in the slice's u32 mask (`cpu::sample_bits`), after Therrien,
//! Levesque and Gilet, "Screen Space Indirect Lighting with Visibility Bitmask" (2023), in its
//! cosine-weighted sector form. The slice and step structure follows raa's GTAO
//! (`crates/raa-gtao/src/wgsl/gtao.wgsl` lines 126-158). The method is original to this
//! repository: solstice has no SSGI to port.
//!
//! No double counting: `cpu::accumulate` lights only the sectors a sample opens for the first time,
//! and `cpu::compose` sets the composite `E_ss + V x E_probe` with alpha `min(V, probe a)`. Screen
//! space replaces the probe and volume irradiance in the sectors it found occluded, so that
//! irradiance is never added a second time.
//!
//! This root holds the pass name, its order, the byte budget and the part `ssgi` with its pass
//! `gi.ssgi` (`install`, `SsgiPart`, `GiSsgiPass`). The CPU twins live in `cpu`, the kernels and
//! pipelines in `passes` and the GPU resources in `resources`.
//!
//! `crates/raa-gi/src/lib.rs`: out of scope for this share: consumers use the module path.
//! `crates/raa-gi/src/parts.rs`: out of scope for this share: consumers use the module path.

use std::cell::RefCell;
use std::rc::Rc;

use self::passes::SsgiPipelines;
use self::resources::{SsgiInputs, SsgiResources};
use super::{GBufferViews, GiPart, GiShared, PartFrame, PartsCtx};
use crate::settings::SsgiQuality;
use crate::tunables::GiTunables;
use raa_core::{GpuContext, RaaError};
use raa_lit::LitHandle;
use raa_pipeline::{
    FeaturePass, ImageDesc, ImageDim, LightingResources, PrepareCtx, RecordCtx, Targets,
};

#[path = "ssgi/cpu.rs"]
pub mod cpu;
#[path = "ssgi/passes.rs"]
pub mod passes;
#[path = "ssgi/resources.rs"]
mod resources;
#[path = "ssgi/stages.rs"]
pub(crate) mod stages;

/// The name of the compute pass `gi.ssgi` (S12 order 44).
pub const SSGI_PASS: &str = "gi.ssgi";

/// The order of `gi.ssgi`: after `gi.area-shadow` (43) and before `shadow.contact` (45).
pub const SSGI_ORDER: u32 = 44;

/// Bytes the part allocates for a viewport of `(W, H)` pixels, each axis clamped to at least 1:
/// `32 x W x H + 48`. That is 8 B per pixel for each of the raw texture, the two history textures
/// and the output, plus the 32-B params and two 1x1 Rgba16Float dummies of 8 B each. The sum
/// saturates at `u64::MAX` for a viewport too large for a u64.
pub fn ssgi_bytes(viewport: (u32, u32)) -> u64 {
    let width = u64::from(viewport.0.max(1));
    let height = u64::from(viewport.1.max(1));
    32u64
        .saturating_mul(width)
        .saturating_mul(height)
        .saturating_add(48)
}

/// The (slices, steps) of `quality` read from `tunables` (contract 'ssgi-counts'): Low gives
/// (slicesLow, stepsLow), Medium (slicesMedium, stepsMedium), High (slicesHigh, stepsHigh). A
/// missing key falls back to `SsgiQuality::slices` or `SsgiQuality::steps`, so a store at its
/// defaults returns the pair those methods return.
pub fn ssgi_counts(quality: SsgiQuality, tunables: &GiTunables) -> (u32, u32) {
    let (slices, steps) = match quality {
        SsgiQuality::Low => ("slicesLow", "stepsLow"),
        SsgiQuality::Medium => ("slicesMedium", "stepsMedium"),
        SsgiQuality::High => ("slicesHigh", "stepsHigh"),
    };
    (
        tunables
            .value(slices)
            .map_or(quality.slices(), |v| v as u32),
        tunables.value(steps).map_or(quality.steps(), |v| v as u32),
    )
}

/// Builds the part `ssgi` and its pass `gi.ssgi` and attaches them. `SsgiPipelines::new` builds the
/// trace and resolve pipelines on the GI compute group-1 layout. `GiSsgiPass` is added to the
/// pipeline with `Pipeline::add`, enabled (a busy pipeline refuses with `pipeline.busy`, a repeated
/// name with `pipeline.duplicate_pass`), and `SsgiPart` is added with `PartsCtx::add_part`. Nothing
/// is allocated here: `SsgiPart::prepare_frame` allocates the resources on the first active frame.
pub(crate) fn install(ctx: &mut PartsCtx<'_>) -> Result<(), RaaError> {
    let pipelines = SsgiPipelines::new(ctx.gpu, &ctx.shared.compute_group1_layout());
    let part = Rc::new(SsgiPart {
        shared: ctx.shared.clone(),
        lit: ctx.lit.handle(),
        state: RefCell::new(SsgiState {
            pipelines,
            resources: None,
            key: None,
            frame_index: 0,
            active: false,
            pass_enabled: true,
            pending: false,
        }),
    });
    {
        let mut pipeline = ctx
            .pipeline
            .try_borrow_mut()
            .map_err(|_| RaaError::FrameGraph {
                code: "pipeline.busy",
                message: "the pipeline is borrowed (called while a frame executes)".to_string(),
            })?;
        pipeline.add(Box::new(GiSsgiPass {
            part: Rc::clone(&part),
        }))?;
    }
    ctx.add_part(part);
    Ok(())
}

/// Handoff: `SsgiPart` is taken over by card 'raa-gi ssgi: software-GPU frame test of gi.ssgi',
/// which drives the pass `gi.ssgi` (`SSGI_PASS`, parts/ssgi.rs:40) and checks the bytes of
/// `ssgi_bytes` (parts/ssgi.rs:49). Today `install` (parts/ssgi.rs:63) attaches this part and
/// `GiSsgiPass`, and the generated `install_parts` calls `install` from `GiFeature::install`
/// (feature.rs:725). The state it is driven by is the key registry of `GiSettingsPatch`
/// (settings.rs:187), applied by `GiHandle::set` (feature.rs:195): `screen_space` true makes the
/// part active and allocates `ssgi_bytes`; `screen_space_quality` sets the slices and steps with no
/// allocation change; `Pipeline::set_enabled("gi.ssgi", false)` drops the bytes on the next
/// prepared frame. LIFECYCLE: SsgiPart::prepare_frame (a GiPart hook run by
/// `GiHandle::prepare_frame`, feature.rs:345, before the pipeline executes) creates SsgiResources
/// on an active frame whose viewport key changed, and drops them on an inactive frame.
/// GiPart::release, through `release_resources` (feature.rs:489), drops them. The bind groups are
/// built lazily by `SsgiResources::work` (parts/ssgi/resources.rs:244) from the gi.ssgi record and
/// dropped by `SsgiResources::invalidate_bindings` (parts/ssgi/resources.rs:234) from the gi.ssgi
/// resize hook. Per enabled pass in order, the pipeline calls resize (only when the targets
/// generation changed) and then prepare (`prepare_single`,
/// crates/raa-pipeline/src/pipeline.rs:881-887), then records every enabled pass in order
/// (`record_single`, pipeline.rs:917-919). So gi.volume (25) prepares binding 14, gi.screen (42)
/// records the probe output and gi.ssgi (44) records after it.
struct SsgiPart {
    shared: GiShared,
    lit: LitHandle,
    state: RefCell<SsgiState>,
}

/// The mutable state of [`SsgiPart`]: the pipelines, the resources of the viewport `key`, the
/// frame counter written to `SsgiParamsGpu.frame`, and the activity, enable and
/// pending-invalidation flags.
struct SsgiState {
    pipelines: SsgiPipelines,
    resources: Option<SsgiResources>,
    key: Option<(u32, u32)>,
    frame_index: u32,
    active: bool,
    pass_enabled: bool,
    pending: bool,
}

/// The activity rule of [`SsgiPart`]: `screenSpace` on, the pass enabled, GI active and both
/// viewport axes non-empty.
fn is_active(active: bool, screen_space: bool, pass_enabled: bool, viewport: (u32, u32)) -> bool {
    active && screen_space && pass_enabled && viewport.0 > 0 && viewport.1 > 0
}

impl GiPart for SsgiPart {
    fn name(&self) -> &'static str {
        "ssgi"
    }

    /// Sets the activity. An inactive frame drops the resources. An active frame reallocates them
    /// when the viewport key changed, applies a pending invalidation, advances the frame counter
    /// and writes the params of `screen_space_quality`.
    fn prepare_frame(&self, gpu: &GpuContext, frame: &PartFrame<'_>) {
        let mut guard = self.state.borrow_mut();
        let state = &mut *guard;
        let (width, height) = frame.viewport;
        state.active = is_active(
            frame.active,
            frame.settings.screen_space,
            state.pass_enabled,
            frame.viewport,
        );
        if !state.active {
            state.resources = None;
            state.key = None;
            return;
        }
        if state.key != Some((width, height)) {
            state.resources = Some(SsgiResources::new(gpu, (width, height)));
            state.key = Some((width, height));
        }
        let pending = std::mem::take(&mut state.pending);
        state.frame_index = state.frame_index.wrapping_add(1);
        state.pipelines.refresh(gpu, &self.shared);
        let quality = frame.settings.screen_space_quality;
        let (slices, steps) = ssgi_counts(quality, &state.pipelines.tunables);
        let Some(resources) = state.resources.as_mut() else {
            return;
        };
        if pending {
            resources.invalidate();
        }
        resources.write_params(gpu, slices, steps, frame.history_valid, state.frame_index);
    }

    /// The composite output of this frame while the part is active with its pass enabled.
    fn screen_output(&self) -> Option<wgpu::TextureView> {
        let state = self.state.borrow();
        if !(state.active && state.pass_enabled) {
            return None;
        }
        state.resources.as_ref().map(SsgiResources::output_view)
    }

    /// True while the part is active with its pass enabled.
    fn wants_history(&self) -> bool {
        let state = self.state.borrow();
        state.active && state.pass_enabled
    }

    /// Sets the pending invalidation, applied by the next active `prepare_frame`.
    fn invalidate_history(&self) {
        self.state.borrow_mut().pending = true;
    }

    /// Drops the resources and the key, and clears the activity.
    fn release(&self) {
        let mut state = self.state.borrow_mut();
        state.resources = None;
        state.key = None;
        state.active = false;
    }

    /// The GPU bytes of the current resources, or 0 without them.
    fn allocated_bytes(&self) -> u64 {
        self.state
            .borrow()
            .resources
            .as_ref()
            .map_or(0, SsgiResources::allocated_bytes)
    }
}

/// The `gi.ssgi` feature pass (order 44): records the trace and the resolve of one [`SsgiPart`]
/// over the frame's buffers, the previous lit HDR and the probe output of gi.screen.
struct GiSsgiPass {
    part: Rc<SsgiPart>,
}

impl FeaturePass for GiSsgiPass {
    fn name(&self) -> &str {
        SSGI_PASS
    }

    fn order(&self) -> u32 {
        SSGI_ORDER
    }

    /// Stores the enable flag and, on disable, sets the pending invalidation.
    fn set_enabled(&mut self, enabled: bool, _lighting: &mut LightingResources) {
        let mut state = self.part.state.borrow_mut();
        state.pass_enabled = enabled;
        if !enabled {
            state.pending = true;
        }
    }

    /// Drops the cached bind groups after the targets were recreated.
    fn resize(&mut self, _gpu: &GpuContext, _targets: &Targets) {
        if let Some(resources) = self.part.state.borrow_mut().resources.as_mut() {
            resources.invalidate_bindings();
        }
    }

    /// Publishes the taps `ssgi`, `raw`, `history` and `output` (docs/render-graph/taps.md) while
    /// the part is active with its pass enabled and the resources exist.
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let state = self.part.state.borrow();
        if !(state.active && state.pass_enabled) {
            return Ok(());
        }
        let Some(resources) = state.resources.as_ref() else {
            return Ok(());
        };
        let (width, height) = resources.viewport();
        let format = wgpu::TextureFormat::Rgba16Float;
        let desc = ImageDesc::new(ImageDim::D2, format, width, height, 1, 1);
        let output = resources.output_view();
        ctx.resources.publish(SSGI_PASS, "ssgi", &output, desc);
        ctx.resources
            .publish(SSGI_PASS, "raw", resources.raw_view(), desc);
        ctx.resources
            .publish(SSGI_PASS, "history", resources.history_write_view(), desc);
        ctx.resources.publish(SSGI_PASS, "output", &output, desc);
        Ok(())
    }

    /// Records `raa-gi.ssgi` while the part is active with resources and the frame buffer and
    /// group 1 exist. A recorded pass commits the history pair.
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let mut guard = self.part.state.borrow_mut();
        let state = &mut *guard;
        if !(state.active && state.pass_enabled) {
            return Ok(());
        }
        let Some(resources) = state.resources.as_mut() else {
            return Ok(());
        };
        let Some(frame) = self.part.shared.frame_buffer() else {
            return Ok(());
        };
        let Some(group1) = self.part.shared.compute_group1(ctx.lighting) else {
            return Ok(());
        };
        let lit_frame = self.part.lit.frame_buffer();
        let history = self.part.shared.history_view();
        let probe = self.part.shared.probe_output_view();
        let inputs = SsgiInputs {
            gi_frame: &frame,
            lit_frame: &lit_frame,
            history: history.as_ref(),
            probe: probe.as_ref(),
        };
        let views = GBufferViews::resolve(ctx.targets, ctx.resources, SSGI_PASS);
        let recorded = {
            let work = resources.work(ctx.gpu, &state.pipelines, &views, &inputs);
            state.pipelines.record(ctx.encoder, &group1, &work)
        };
        if recorded {
            resources.commit();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raa_core::testing::skip;
    use raa_lit::{LitOptions, LitRenderer};
    use raa_pipeline::{Pipeline, ResourceTable};

    /// Builds the part and pass as `install` does (without adding the pass to a pipeline), makes
    /// the part active with 8x8 resources and runs the pass's `prepare` with an 8x8 `Targets`
    /// and a fresh table. The pass reads no shared input in `prepare`, so none is initialised.
    #[test]
    fn prepare_publishes_the_four_ssgi_ports() {
        let test = "prepare_publishes_the_four_ssgi_ports";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let pipeline = Rc::new(RefCell::new(Pipeline::new(
            &gpu,
            wgpu::TextureFormat::Rgba8Unorm,
            false,
        )));
        let lit = LitRenderer::new(&gpu, &pipeline, LitOptions::default()).expect("lit renderer");
        let shared = GiShared::new(&gpu);
        let part = Rc::new(SsgiPart {
            lit: lit.handle(),
            state: RefCell::new(SsgiState {
                pipelines: SsgiPipelines::new(&gpu, &shared.compute_group1_layout()),
                resources: Some(SsgiResources::new(&gpu, (8, 8))),
                key: Some((8, 8)),
                frame_index: 0,
                active: true,
                pass_enabled: true,
                pending: false,
            }),
            shared,
        });
        let mut pass = GiSsgiPass { part };
        let targets = Targets::new(&gpu, (8, 8));
        let table = ResourceTable::new();
        let mut lighting = LightingResources::new(&gpu);
        let mut ctx = PrepareCtx {
            gpu: &gpu,
            size: (8, 8),
            frame: 0,
            time_s: 0.0,
            dt_s: 0.0,
            targets: &targets,
            lighting: &mut lighting,
            resources: &table,
        };
        pass.prepare(&mut ctx).expect("prepare");
        let entries = table.entries();
        for port in ["ssgi", "raw", "history", "output"] {
            let found = entries.iter().find(|e| e.0 == SSGI_PASS && e.1 == port);
            let desc = found.unwrap_or_else(|| panic!("missing {port}")).2;
            assert_eq!(desc.format, wgpu::TextureFormat::Rgba16Float, "{port}");
            assert_eq!((desc.width, desc.height, desc.dim), (8, 8, ImageDim::D2));
        }
        assert_eq!(entries.len(), 4);
    }

    #[test]
    fn ssgi_constants_match_the_s12_row() {
        assert_eq!(SSGI_PASS, "gi.ssgi");
        assert_eq!(SSGI_ORDER, 44);
    }

    #[test]
    fn ssgi_bytes_matches_the_hand_values() {
        assert_eq!(ssgi_bytes((1280, 720)), 29_491_248);
        assert_eq!(ssgi_bytes((320, 180)), 1_843_248);
        assert_eq!(ssgi_bytes((0, 0)), 80);
    }

    #[test]
    fn activity_needs_screen_space_the_pass_gi_and_a_viewport() {
        assert!(is_active(true, true, true, (1280, 720)));
        assert!(!is_active(false, true, true, (1280, 720)));
        assert!(!is_active(true, false, true, (1280, 720)));
        assert!(!is_active(true, true, false, (1280, 720)));
        assert!(!is_active(true, true, true, (0, 720)));
        assert!(!is_active(true, true, true, (1280, 0)));
    }
}
