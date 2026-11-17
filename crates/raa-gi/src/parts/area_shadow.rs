//! The M6 mid tier of the area-light shadows (S44): stratified shadow rays per pixel and mid-tier
//! slot, a reprojected and spatially denoised shadow weight and visibility pair (temporal and
//! spatial bilateral), and the ratio estimator that writes the visibility mask. This module root
//! holds the part `area-shadow` (order 43) and its `gi.area-shadow` pass; the CPU twins live in
//! `cpu`, the kernels and pipelines in `passes`, and the GPU resources in `resources`.

#[path = "area_shadow/cpu.rs"]
pub mod cpu;

#[path = "area_shadow/restir_cpu.rs"]
pub mod restir_cpu;

#[path = "area_shadow/restir.rs"]
pub mod restir;

#[path = "area_shadow/passes.rs"]
pub mod passes;

#[path = "area_shadow/resources.rs"]
mod resources;

#[path = "area_shadow/restir_resources.rs"]
mod restir_resources;

use std::cell::RefCell;
use std::rc::Rc;

use self::passes::{AreaShadowPipelines, AreaShadowWork};
use self::resources::{AreaShadowInputs, AreaShadowResources, needs_reallocation, resource_bytes};
use self::restir::RestirPipelines;
use self::restir_resources::{RestirResources, needs_restir_reallocation, restir_resource_bytes};
use super::{GiPart, GiShared, PartFrame, PartsCtx};
use raa_core::{GpuContext, RaaError};
use raa_lit::{LitHandle, StochasticMode, table_limit};
use raa_pipeline::{
    FeaturePass, ImageDesc, ImageDim, LightingResources, MAX_STOCHASTIC_LIGHTS, PrepareCtx,
    RecordCtx, ResourceTable, Targets,
};

/// The S12 name of the pass.
pub const AREA_SHADOW_PASS: &str = "gi.area-shadow";

/// The S12 position of the pass: after gi.screen (42) and before shadow.contact (45).
pub const AREA_SHADOW_ORDER: u32 = 43;

/// The GPU bytes of one `AreaShadowResources` of `viewport` and history `groups`
/// (`resources::resource_bytes`): `66_355_349` for 1280x720 and 2 groups.
pub fn area_shadow_bytes(viewport: (u32, u32), groups: u32) -> u64 {
    resource_bytes(viewport, groups)
}

/// The GPU bytes of the ReSTIR ultra tier at `viewport`: one `AreaShadowResources` of 1 history
/// group plus one `RestirResources` (`restir_resource_bytes`): `81_297_621` for 1280x720.
pub fn area_restir_bytes(viewport: (u32, u32)) -> u64 {
    resource_bytes(viewport, 1) + restir_resource_bytes(viewport)
}

/// Handoff: card 'raa-gi: AreaShadowPart and the gi.area-shadow feature pass (order 43) publishing
/// the mask' replaces this body. Today `install` adds nothing to `ctx` and returns `Ok(())`; the
/// generated `install_parts` of `parts.rs` calls it once per GI feature install, in file-name order.
pub(crate) fn install(ctx: &mut PartsCtx<'_>) -> Result<(), RaaError> {
    let pipelines = AreaShadowPipelines::new(ctx.gpu, &ctx.shared.compute_group1_layout());
    let part = Rc::new(AreaShadowPart {
        shared: ctx.shared.clone(),
        lit: ctx.lit.handle(),
        state: RefCell::new(AreaShadowState {
            pipelines,
            resources: None,
            restir: None,
            restir_pipelines: None,
            mode: StochasticMode::Stratified,
            key: None,
            generations: [0; MAX_STOCHASTIC_LIGHTS],
            sdf: None,
            limit: 0,
            frame_index: 0,
            active: false,
            pass_enabled: true,
            pending: false,
            recordable: false,
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
        pipeline.add(Box::new(GiAreaShadowPass {
            part: Rc::clone(&part),
        }))?;
    }
    ctx.add_part(part);
    Ok(())
}

/// The GI part `area-shadow`: the shared frame state, the lit handle whose prepare wrote this
/// view's mid-tier table, and the mutable per-frame state.
///
/// It consumes the `GiShared` seam (`crates/raa-gi/src/parts.rs:240 GiShared::pyramid_view`),
/// the lit frame (`crates/raa-lit/src/handle.rs:1237 LitHandle::area_stochastic`), the
/// pipelines (`crates/raa-gi/src/parts/area_shadow/passes.rs:200 AreaShadowPipelines`), the
/// resources (`crates/raa-gi/src/parts/area_shadow/resources.rs:155 AreaShadowResources`) and
/// `LightingResources::set_area_shadow` (crates/raa-pipeline/src/lighting.rs:1138).
struct AreaShadowPart {
    shared: GiShared,
    lit: LitHandle,
    state: RefCell<AreaShadowState>,
}

/// The mutable state of [`AreaShadowPart`]: the pipelines, the resources keyed by
/// (viewport, groups), the ReSTIR resources and lazily built ReSTIR pipelines, the mode of the
/// last prepared frame, the slot generations seen last frame, the SDF inputs fetched in prepare,
/// the slot limit and frame counter, and the activity, enable, pending-invalidate and recordable
/// flags.
///
/// The ReSTIR tier calls `RestirResources` (Handoff at restir_resources.rs:62
/// `RestirResources`), `RestirPipelines` (Handoff at restir.rs:199 `RestirPipelines`),
/// `AreaShadowResources::work_restir` and `shared_views` (Handoff at resources.rs:408
/// `work_restir`) and `restir_cpu::remap_table` (Handoff at restir_cpu.rs:299 `remap_table`).
struct AreaShadowState {
    pipelines: AreaShadowPipelines,
    resources: Option<AreaShadowResources>,
    restir: Option<RestirResources>,
    restir_pipelines: Option<RestirPipelines>,
    mode: StochasticMode,
    key: Option<((u32, u32), u32)>,
    generations: [u32; MAX_STOCHASTIC_LIGHTS],
    sdf: Option<(wgpu::TextureView, wgpu::Buffer)>,
    limit: u32,
    frame_index: u32,
    active: bool,
    pass_enabled: bool,
    pending: bool,
    recordable: bool,
}

impl GiPart for AreaShadowPart {
    fn name(&self) -> &'static str {
        "area-shadow"
    }

    /// Records activity only (it runs before `LitRenderer::prepare`, so the slot table is not
    /// current yet): active with GI, `localLightShadows` and a non-empty viewport.
    fn prepare_frame(&self, _gpu: &GpuContext, frame: &PartFrame<'_>) {
        let active = frame.active
            && frame.settings.local_light_shadows
            && frame.viewport.0 > 0
            && frame.viewport.1 > 0;
        self.state.borrow_mut().active = active;
    }

    /// Sets the pending flag, applied as `invalidate_all` in the next pass prepare, and
    /// invalidates the reservoir history.
    fn invalidate_history(&self) {
        let mut state = self.state.borrow_mut();
        state.pending = true;
        if let Some(restir) = state.restir.as_mut() {
            restir.invalidate();
        }
    }

    /// Drops the resources (the pipelines are kept), the key, the SDF inputs and the activity.
    fn release(&self) {
        let mut state = self.state.borrow_mut();
        state.resources = None;
        state.restir = None;
        state.key = None;
        state.sdf = None;
        state.limit = 0;
        state.active = false;
        state.recordable = false;
    }

    /// The GPU bytes of the current resources and ReSTIR resources, or 0 without them.
    fn allocated_bytes(&self) -> u64 {
        let state = self.state.borrow();
        let shadow = state
            .resources
            .as_ref()
            .map_or(0, AreaShadowResources::allocated_bytes);
        let restir = state
            .restir
            .as_ref()
            .map_or(0, RestirResources::allocated_bytes);
        shadow + restir
    }
}

/// Publishes the taps of `gi.area-shadow` (docs/render-graph/taps.md): the two-layer mask under
/// `areaShadow` and `mask`, the raw array, the depth history write half and, on ReSTIR frames,
/// the LTC atlas.
fn publish_taps(
    table: &ResourceTable,
    resources: &AreaShadowResources,
    restir: Option<&RestirResources>,
) {
    let (width, height) = resources.viewport();
    let mask = resources.mask_view();
    let shared = resources.shared_views();
    let float = wgpu::TextureFormat::Rgba16Float;
    let masks = ImageDesc::new(ImageDim::D2Array, float, width, height, 2, 1);
    let raw_layers = resources.groups().max(2);
    let raw = ImageDesc::new(ImageDim::D2Array, float, width, height, raw_layers, 1);
    let depth = wgpu::TextureFormat::R32Float;
    table.publish(AREA_SHADOW_PASS, "areaShadow", &mask, masks);
    table.publish(AREA_SHADOW_PASS, "raw", shared.raw, raw);
    let depth = ImageDesc::new(ImageDim::D2, depth, width, height, 1, 1);
    table.publish(AREA_SHADOW_PASS, "depthHistory", shared.depth_write, depth);
    table.publish(AREA_SHADOW_PASS, "mask", &mask, masks);
    if let Some(restir) = restir {
        let atlas = ImageDesc::new(
            ImageDim::D2,
            raa_pipeline::BRDF_ATLAS_FORMAT,
            raa_pipeline::BRDF_ATLAS_WIDTH,
            raa_pipeline::BRDF_ATLAS_HEIGHT,
            1,
            1,
        );
        table.publish(AREA_SHADOW_PASS, "restirAtlas", restir.atlas_view(), atlas);
    }
}

/// The `gi.area-shadow` feature pass (order 43): prepares the mask publication and records the
/// trace, temporal and spatial dispatches of one [`AreaShadowPart`].
struct GiAreaShadowPass {
    part: Rc<AreaShadowPart>,
}

impl FeaturePass for GiAreaShadowPass {
    fn name(&self) -> &str {
        AREA_SHADOW_PASS
    }

    fn order(&self) -> u32 {
        AREA_SHADOW_ORDER
    }

    /// Stores the enable flag and, on disable, clears the group-1 mask dummies and invalidates.
    fn set_enabled(&mut self, enabled: bool, lighting: &mut LightingResources) {
        let mut state = self.part.state.borrow_mut();
        state.pass_enabled = enabled;
        if !enabled {
            state.pending = true;
            state.recordable = false;
            lighting.clear_area_shadow();
        }
    }

    /// Drops the cached bind groups after the targets were recreated.
    fn resize(&mut self, _gpu: &GpuContext, _targets: &Targets) {
        if let Some(resources) = self.part.state.borrow_mut().resources.as_mut() {
            resources.invalidate_bindings();
        }
    }

    /// Allocates or reallocates the resources (and the ReSTIR resources on Restir frames),
    /// invalidates changed history groups, writes the params, fetches the optional SDF inputs and
    /// publishes the mask and slot table.
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let frame = self.part.lit.area_stochastic();
        let limit = table_limit(&frame.table).min(8);
        let mut state = self.part.state.borrow_mut();
        state.recordable = false;
        let missing_frame = self.part.shared.frame_buffer().is_none();
        if !state.active || !state.pass_enabled || limit == 0 || missing_frame {
            state.resources = None;
            state.restir = None;
            state.key = None;
            state.sdf = None;
            state.limit = 0;
            ctx.lighting.clear_area_shadow();
            return Ok(());
        }
        let restir_mode = frame.mode == StochasticMode::Restir;
        let groups = if restir_mode {
            1
        } else {
            cpu::groups_for(limit)
        };
        if needs_reallocation(state.key, ctx.size, groups) {
            state.resources = Some(AreaShadowResources::new(ctx.gpu, ctx.size, groups));
            state.key = Some(((ctx.size.0.max(1), ctx.size.1.max(1)), groups));
            state.generations = [0; MAX_STOCHASTIC_LIGHTS];
        }
        if restir_mode {
            if state.restir_pipelines.is_none() {
                let layout = self.part.shared.compute_group1_layout();
                state.restir_pipelines = Some(RestirPipelines::new(ctx.gpu, &layout));
            }
            let current = state.restir.as_ref().map(RestirResources::viewport);
            if needs_restir_reallocation(current, ctx.size) {
                state.restir = Some(RestirResources::new(ctx.gpu, ctx.size));
            }
        } else {
            state.restir = None;
        }
        let mode_changed = state.mode != frame.mode;
        state.mode = frame.mode;
        let pending = std::mem::take(&mut state.pending);
        let sdf = self.part.shared.trace().and_then(|handle| {
            let mut scene = handle.try_borrow_mut().ok()?;
            let volume = scene.volume(ctx.gpu)?;
            Some((volume.view.clone(), volume.uniform.clone()))
        });
        let AreaShadowState {
            resources,
            restir,
            generations,
            sdf: sdf_slot,
            frame_index,
            limit: limit_slot,
            recordable,
            ..
        } = &mut *state;
        let Some(resources) = resources.as_mut() else {
            return Ok(());
        };
        if pending || mode_changed {
            resources.invalidate_all();
        }
        if let Some(restir) = restir.as_mut() {
            if mode_changed {
                restir.invalidate();
            }
            let remap =
                restir_cpu::remap_table(generations, &frame.generations, restir.history_valid());
            if *generations != frame.generations {
                resources.invalidate_group(0);
            }
            *generations = frame.generations;
            *frame_index = frame_index.wrapping_add(1);
            resources.write_params(ctx.gpu, *frame_index, limit);
            restir.write_params(ctx.gpu, *frame_index, &remap);
            *sdf_slot = sdf;
            ctx.lighting
                .set_area_shadow(resources.mask_view(), &frame.table)?;
            publish_taps(ctx.resources, resources, Some(&*restir));
            *limit_slot = limit;
            *recordable = true;
            return Ok(());
        }
        for group in 0..groups {
            let first = 2 * group as usize;
            let second = first + 1;
            if frame.generations[first] != generations[first]
                || frame.generations[second] != generations[second]
            {
                resources.invalidate_group(group);
            }
            generations[first] = frame.generations[first];
            generations[second] = frame.generations[second];
        }
        *frame_index = frame_index.wrapping_add(1);
        resources.write_params(ctx.gpu, *frame_index, limit);
        *sdf_slot = sdf;
        ctx.lighting
            .set_area_shadow(resources.mask_view(), &frame.table)?;
        publish_taps(ctx.resources, resources, None);
        *limit_slot = limit;
        *recordable = true;
        Ok(())
    }

    /// Records the `raa-gi.area-shadow` pass over the prepared inputs and commits the histories
    /// when the pipelines recorded it.
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let mut state = self.part.state.borrow_mut();
        if !state.recordable {
            return Ok(());
        }
        let Some(frame) = self.part.shared.frame_buffer() else {
            return Ok(());
        };
        let Some(group1) = self.part.shared.compute_group1(ctx.lighting) else {
            return Ok(());
        };
        let lit_frame = self.part.lit.frame_buffer();
        let pyramid = self.part.shared.pyramid_view();
        let sdf = state.sdf.clone();
        let inputs = AreaShadowInputs {
            gi_frame: &frame,
            lit_frame: &lit_frame,
            table: ctx.lighting.area_stochastic_buffer(),
            sampler: ctx.lighting.linear_clamp(),
            pyramid: pyramid.as_ref(),
            sdf: sdf.as_ref().map(|(view, buffer)| (view, buffer)),
        };
        let limit = state.limit;
        let state = &mut *state;
        let AreaShadowState {
            pipelines,
            resources,
            restir,
            restir_pipelines,
            ..
        } = state;
        let Some(resources) = resources.as_mut() else {
            return Ok(());
        };
        if let (Some(restir), Some(restir_pipelines)) = (restir.as_mut(), restir_pipelines.as_ref())
        {
            let shared = resources.shared_views();
            let group2 = restir
                .bind_group(ctx.gpu, restir_pipelines, ctx.targets, &inputs, &shared)
                .clone();
            let recorded = {
                let work = resources.work_restir(ctx.gpu, pipelines, ctx.targets, &inputs, limit);
                restir_pipelines.record(ctx.encoder, &group1, &group2, pipelines, &work)
            };
            if recorded {
                resources.commit();
                restir.commit();
            }
            return Ok(());
        }
        let recorded = {
            let work: AreaShadowWork<'_> =
                resources.work(ctx.gpu, pipelines, ctx.targets, &inputs, limit);
            pipelines.record(ctx.encoder, &group1, &work)
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

    /// Allocates 8x8 resources for 2 groups plus the ReSTIR set and publishes into a fresh table:
    /// the helper `prepare` calls after (re)allocation. No shared input is read.
    #[test]
    fn taps_publish_the_array_ports_and_the_atlas() {
        let test = "taps_publish_the_array_ports_and_the_atlas";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let resources = AreaShadowResources::new(&gpu, (8, 8), 2);
        let restir = RestirResources::new(&gpu, (8, 8));
        let table = ResourceTable::new();
        publish_taps(&table, &resources, Some(&restir));
        let entries = table.entries();
        let desc = |port: &str| {
            let found = entries
                .iter()
                .find(|e| e.0 == AREA_SHADOW_PASS && e.1 == port);
            found.unwrap_or_else(|| panic!("missing {port}")).2
        };
        for port in ["areaShadow", "mask", "raw"] {
            assert_eq!(desc(port).dim, ImageDim::D2Array, "{port}");
            assert_eq!(desc(port).layers, 2, "{port}");
        }
        assert_eq!(desc("depthHistory").format, wgpu::TextureFormat::R32Float);
        assert_eq!(desc("restirAtlas").format, wgpu::TextureFormat::Rgba16Float);
        assert_eq!(entries.len(), 5);
    }

    #[test]
    fn constants_match_the_s12_table() {
        assert_eq!(AREA_SHADOW_PASS, "gi.area-shadow");
        assert_eq!(AREA_SHADOW_ORDER, 43);
    }

    #[test]
    fn area_shadow_bytes_matches_the_hand_derivation() {
        assert_eq!(area_shadow_bytes((1280, 720), 2), 66_355_349);
    }

    #[test]
    fn area_restir_bytes_matches_the_contract() {
        assert_eq!(area_restir_bytes((1280, 720)), 81_297_621);
        assert_eq!(area_restir_bytes((320, 180)), 5_265_621);
    }
}
