//! The GPU resources of the ReSTIR ultra tier (contract 'area-restir-resources'): the current
//! and history reservoir buffers, the 64-B `RestirParamsGpu` uniform, the LTC atlas, the
//! reservoir history validity and the cached group-2 bind group on `RestirPipelines.layout`.
#![allow(dead_code)]

use raa_core::GpuContext;
use raa_pipeline::Targets;
use raa_taa::HistoryState;

use super::resources::{AreaShadowInputs, AreaSharedViews};
use super::restir::{RESERVOIR_BYTES, RESTIR_PARAMS_SIZE, RestirPipelines};
use super::restir_cpu::{
    RESTIR_BOILING_FACTOR, RESTIR_CANDIDATES, RESTIR_M_CAP, RESTIR_NORMAL_COS,
    RESTIR_SPATIAL_RADIUS, RESTIR_SPATIAL_SAMPLES,
};

/// The bytes of the 128 x 192 Rgba16Float BRDF/LTC atlas.
const LTC_ATLAS_BYTES: u64 = 128 * 192 * 8;

/// The inputs the cached group-2 bind group was built from, and the group itself.
struct RestirBindings {
    gi_frame: wgpu::Buffer,
    lit_frame: wgpu::Buffer,
    table: wgpu::Buffer,
    pyramid: Option<wgpu::TextureView>,
    sdf_field: Option<wgpu::TextureView>,
    sdf_uniform: Option<wgpu::Buffer>,
    raw: wgpu::TextureView,
    parity: usize,
    group: wgpu::BindGroup,
}

fn storage(gpu: &GpuContext, label: &str, size: u64) -> wgpu::Buffer {
    gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    })
}

fn texture_resource(view: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::TextureView(view)
}

/// The ReSTIR reservoirs, params, LTC atlas and reservoir history validity of one viewport.
///
/// Handoff: card 'raa-gi: gi.area-shadow part runs the ReSTIR ultra tier on Restir frames' calls
/// `RestirResources::new`, `RestirResources::write_params`, `RestirResources::bind_group`,
/// `RestirResources::commit`, `RestirResources::invalidate`, `RestirResources::history_valid`,
/// `RestirResources::allocated_bytes` and the free `needs_restir_reallocation` and
/// `restir_resource_bytes`. Today `new` allocates the two 16-B-per-pixel STORAGE reservoir
/// buffers, the 64-B params uniform and the LTC atlas; `bind_group` caches the group-2 bind group
/// over `RestirPipelines.layout` (Handoff at restir.rs:199 `RestirPipelines`) and rebuilds it when
/// the shared parity or an input buffer or view differs. LIFECYCLE: RestirResources is created by
/// GiAreaShadowPass::prepare (pipeline prepare, order 43) on a frame whose
/// AreaStochasticFrame.mode is Restir when needs_restir_reallocation reports a new viewport, and
/// dropped by that prepare on a Stratified frame or whenever it clears the mask (GI inactive,
/// pass disabled, limit 0, no GiFrameGpu) and by GiPart::release; bind_group runs in
/// GiAreaShadowPass::record after gi.screen (order 42) published the pyramid, and commit runs at
/// the end of that record when it recorded.
pub(crate) struct RestirResources {
    viewport: (u32, u32),
    current: wgpu::Buffer,
    history: wgpu::Buffer,
    params: wgpu::Buffer,
    atlas: wgpu::Texture,
    atlas_view: wgpu::TextureView,
    state: HistoryState,
    bindings: Option<RestirBindings>,
}

impl RestirResources {
    /// Allocates the resources of `viewport`, its axes clamped to at least 1.
    pub(crate) fn new(gpu: &GpuContext, viewport: (u32, u32)) -> RestirResources {
        let viewport = (viewport.0.max(1), viewport.1.max(1));
        let reservoirs = u64::from(viewport.0) * u64::from(viewport.1) * RESERVOIR_BYTES;
        let atlas = raa_pipeline::create_brdf_atlas(gpu, "raa-gi.area-restir.ltc", [1.0, 0.0]);
        let atlas_view = atlas.create_view(&wgpu::TextureViewDescriptor::default());
        RestirResources {
            viewport,
            current: storage(gpu, "raa-gi.area-restir.current", reservoirs),
            history: storage(gpu, "raa-gi.area-restir.history", reservoirs),
            params: gpu.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("raa-gi.area-restir.params"),
                size: RESTIR_PARAMS_SIZE,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            atlas,
            atlas_view,
            state: HistoryState::new(viewport),
            bindings: None,
        }
    }

    /// The clamped viewport the reservoirs were allocated for.
    pub(crate) fn viewport(&self) -> (u32, u32) {
        self.viewport
    }

    /// True when a frame was committed since the last invalidation.
    pub(crate) fn history_valid(&self) -> bool {
        self.state.is_valid()
    }

    /// Marks the history reservoirs unusable.
    pub(crate) fn invalidate(&mut self) {
        self.state.invalidate();
    }

    /// Records one committed frame of reservoirs.
    pub(crate) fn commit(&mut self) {
        self.state.commit();
    }

    /// Writes the 64-B `RestirParamsGpu` little-endian by offset.
    pub(crate) fn write_params(&self, gpu: &GpuContext, frame_index: u32, remap: &[u32; 8]) {
        let mut bytes = [0u8; RESTIR_PARAMS_SIZE as usize];
        let words = [
            frame_index,
            u32::from(self.history_valid()),
            RESTIR_CANDIDATES,
            RESTIR_SPATIAL_SAMPLES,
        ];
        for (lane, value) in words.iter().chain(remap.iter()).enumerate() {
            bytes[lane * 4..lane * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        let floats = [
            RESTIR_SPATIAL_RADIUS,
            RESTIR_M_CAP,
            RESTIR_BOILING_FACTOR,
            RESTIR_NORMAL_COS,
        ];
        for (lane, value) in floats.iter().enumerate() {
            bytes[48 + lane * 4..52 + lane * 4].copy_from_slice(&value.to_le_bytes());
        }
        gpu.queue().write_buffer(&self.params, 0, &bytes);
    }

    /// The cached group-2 bind group, rebuilt when the shared parity, the raw view or any input
    /// buffer or view differs from the one it was built from.
    pub(crate) fn bind_group(
        &mut self,
        gpu: &GpuContext,
        pipelines: &RestirPipelines,
        targets: &Targets,
        inputs: &AreaShadowInputs<'_>,
        shared: &AreaSharedViews<'_>,
    ) -> &wgpu::BindGroup {
        let stale = self.bindings.as_ref().is_none_or(|bindings| {
            bindings.parity != shared.parity
                || bindings.raw != *shared.raw
                || bindings.gi_frame != *inputs.gi_frame
                || bindings.lit_frame != *inputs.lit_frame
                || bindings.table != *inputs.table
                || bindings.pyramid.as_ref() != inputs.pyramid
                || bindings.sdf_field.as_ref() != inputs.sdf.map(|sdf| sdf.0)
                || bindings.sdf_uniform.as_ref() != inputs.sdf.map(|sdf| sdf.1)
        });
        if stale {
            let group = self.build_group(gpu, pipelines, targets, inputs, shared);
            self.bindings = Some(RestirBindings {
                gi_frame: inputs.gi_frame.clone(),
                lit_frame: inputs.lit_frame.clone(),
                table: inputs.table.clone(),
                pyramid: inputs.pyramid.cloned(),
                sdf_field: inputs.sdf.map(|sdf| sdf.0.clone()),
                sdf_uniform: inputs.sdf.map(|sdf| sdf.1.clone()),
                raw: shared.raw.clone(),
                parity: shared.parity,
                group,
            });
        }
        match &self.bindings {
            Some(bindings) => &bindings.group,
            None => unreachable!("the bindings were built above"),
        }
    }

    fn build_group(
        &self,
        gpu: &GpuContext,
        pipelines: &RestirPipelines,
        targets: &Targets,
        inputs: &AreaShadowInputs<'_>,
        shared: &AreaSharedViews<'_>,
    ) -> wgpu::BindGroup {
        let pyramid = inputs.pyramid.unwrap_or(shared.dummy_pyramid);
        let (sdf_field, sdf_uniform) = inputs
            .sdf
            .unwrap_or((shared.dummy_sdf_field, shared.dummy_sdf_uniform));
        let bindings = [
            (0, inputs.gi_frame.as_entire_binding()),
            (1, texture_resource(targets.depth())),
            (2, texture_resource(targets.normal())),
            (4, texture_resource(pyramid)),
            (5, texture_resource(shared.raw)),
            (6, texture_resource(shared.depth_write)),
            (8, inputs.table.as_entire_binding()),
            (10, texture_resource(sdf_field)),
            (11, sdf_uniform.as_entire_binding()),
            (12, texture_resource(targets.motion())),
            (13, inputs.lit_frame.as_entire_binding()),
            (14, texture_resource(shared.depth_read)),
            (15, self.params.as_entire_binding()),
            (16, self.current.as_entire_binding()),
            (17, self.history.as_entire_binding()),
            (18, texture_resource(&self.atlas_view)),
            (19, texture_resource(targets.material())),
        ];
        let entries: Vec<wgpu::BindGroupEntry<'_>> = bindings
            .into_iter()
            .map(|(binding, resource)| wgpu::BindGroupEntry { binding, resource })
            .collect();
        gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa-gi.area-restir.group2"),
            layout: &pipelines.layout,
            entries: &entries,
        })
    }

    /// The GPU bytes this set owns: `restir_resource_bytes(viewport)`.
    pub(crate) fn allocated_bytes(&self) -> u64 {
        restir_resource_bytes(self.viewport)
    }
}

/// The GPU bytes of one `RestirResources` of `viewport` (axes clamped to at least 1): two
/// 16-B-per-pixel reservoir buffers, the 64-B params and the 196,608-B LTC atlas.
pub(crate) fn restir_resource_bytes(viewport: (u32, u32)) -> u64 {
    let texels = u64::from(viewport.0.max(1)) * u64::from(viewport.1.max(1));
    2 * RESERVOIR_BYTES * texels + RESTIR_PARAMS_SIZE + LTC_ATLAS_BYTES
}

/// True when `current` is not Some of the clamped `viewport`.
pub(crate) fn needs_restir_reallocation(current: Option<(u32, u32)>, viewport: (u32, u32)) -> bool {
    current != Some((viewport.0.max(1), viewport.1.max(1)))
}

#[cfg(test)]
mod tests {
    use super::super::resources::AreaShadowResources;
    use super::*;
    use crate::parts::GiShared;
    use raa_core::testing::{block_on, skip, software_gpu};

    fn uniform(gpu: &GpuContext, label: &str, size: u64) -> wgpu::Buffer {
        gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    #[test]
    fn restir_resource_bytes_matches_the_hand_derivation() {
        assert_eq!(restir_resource_bytes((1280, 720)), 29_687_872);
        assert_eq!(restir_resource_bytes((320, 180)), 2_039_872);
        assert_eq!(restir_resource_bytes((0, 0)), 196_704);
    }

    #[test]
    fn needs_restir_reallocation_tracks_the_clamped_viewport() {
        assert!(needs_restir_reallocation(None, (64, 32)));
        assert!(!needs_restir_reallocation(Some((64, 32)), (64, 32)));
        assert!(needs_restir_reallocation(Some((64, 32)), (128, 32)));
        assert!(!needs_restir_reallocation(Some((1, 1)), (0, 0)));
    }

    #[test]
    fn restir_resources_allocate_and_bind_in_a_validation_scope() {
        let test = "restir_resources_allocate_and_bind_in_a_validation_scope";
        let Some(gpu) = software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        // No pass runs: the targets stay as created (cleared), and the bind group only needs
        // their views; nothing is dispatched.
        let shared = GiShared::new(&gpu);
        let targets = Targets::new(&gpu, (64, 32));
        let area = AreaShadowResources::new(&gpu, (64, 32), 1);
        let pipelines = RestirPipelines::new(&gpu, &shared.compute_group1_layout());
        let gi_frame = uniform(&gpu, "raa-gi.area-restir.test.frame", 432);
        let lit_frame = uniform(&gpu, "raa-gi.area-restir.test.lit-frame", 768);
        let table = uniform(&gpu, "raa-gi.area-restir.test.table", 80);
        let sampler = gpu.device().create_sampler(&wgpu::SamplerDescriptor {
            label: Some("raa-gi.area-restir.test.sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let inputs = AreaShadowInputs {
            gi_frame: &gi_frame,
            lit_frame: &lit_frame,
            table: &table,
            sampler: &sampler,
            pyramid: None,
            sdf: None,
        };

        let scope = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
        let mut resources = RestirResources::new(&gpu, (64, 32));
        resources.write_params(&gpu, 1, &[0, 1, 2, 3, 4, 5, 6, 7]);
        resources.bind_group(&gpu, &pipelines, &targets, &inputs, &area.shared_views());
        assert!(
            block_on(scope.pop()).is_none(),
            "the ReSTIR resources and group-2 bind group create without a validation error"
        );
        assert_eq!(resources.viewport(), (64, 32));
        assert_eq!(resources.allocated_bytes(), restir_resource_bytes((64, 32)));

        assert!(!resources.history_valid());
        resources.commit();
        assert!(resources.history_valid());
        resources.invalidate();
        assert!(!resources.history_valid());
    }
}
