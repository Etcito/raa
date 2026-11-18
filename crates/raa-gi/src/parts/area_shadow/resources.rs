//! The GPU resources of the `gi.area-shadow` pass (S44, contract 'area-shadow-resources'): the
//! raw history layers the trace writes, one `raa_taa::HistoryPair` per history group, the depth
//! history the trace writes and the temporal filter reprojects, the visibility mask, the trace
//! and dispatch uniforms, the 1x1 dummies of the optional pyramid and SDF volume, and the cached
//! bind groups of contracts 'area-shadow-trace-wgsl', 'area-shadow-sdf-wgsl' and
//! 'area-shadow-denoise-wgsl' on the layouts of `AreaShadowPipelines`.
#![allow(dead_code)]

use raa_core::GpuContext;
use raa_taa::HistoryPair;

use super::cpu;
use super::passes::{
    AREA_DISPATCH_SIZE, AREA_PARAMS_SIZE, AreaShadowPipelines, AreaShadowWork, SDF_VOLUME_SIZE,
};
use crate::parts::GBufferViews;

/// The ray-trace inputs of one frame: the two frame uniforms, the `RaaAreaStochasticGpu` slot
/// table, the shared clamp sampler, the all-mip depth pyramid `gi.screen` published this frame
/// (or `None`) and the attached raa-sdf-trace volume (field and uniform, or `None`).
pub(crate) struct AreaShadowInputs<'a> {
    pub(crate) gi_frame: &'a wgpu::Buffer,
    pub(crate) lit_frame: &'a wgpu::Buffer,
    pub(crate) table: &'a wgpu::Buffer,
    pub(crate) sampler: &'a wgpu::Sampler,
    pub(crate) pyramid: Option<&'a wgpu::TextureView>,
    pub(crate) sdf: Option<(&'a wgpu::TextureView, &'a wgpu::Buffer)>,
}

/// The inputs one set of cached bind groups was built from, and the groups themselves. `work`
/// rebuilds them when the parity, any of these buffers and views, or the G-buffer `views`
/// differs.
struct AreaShadowBindings {
    gi_frame: wgpu::Buffer,
    lit_frame: wgpu::Buffer,
    table: wgpu::Buffer,
    sampler: wgpu::Sampler,
    pyramid: Option<wgpu::TextureView>,
    sdf_field: Option<wgpu::TextureView>,
    sdf_uniform: Option<wgpu::Buffer>,
    views: GBufferViews,
    parity: usize,
    trace: wgpu::BindGroup,
    temporal: Vec<wgpu::BindGroup>,
    spatial_h: Vec<wgpu::BindGroup>,
    spatial_v: Vec<wgpu::BindGroup>,
    spatial_v_broadcast: Vec<wgpu::BindGroup>,
}

/// The shared views the ReSTIR tier writes and binds: the D2Array raw history view, the R32Float
/// depth pair's write and read views, the 1x1 pyramid, 1x1x1 SDF-field and 64-B SDF-uniform
/// dummies, and `parity`, the depth pair's `state().write_index()`.
pub(crate) struct AreaSharedViews<'a> {
    pub(crate) raw: &'a wgpu::TextureView,
    pub(crate) depth_write: &'a wgpu::TextureView,
    pub(crate) depth_read: &'a wgpu::TextureView,
    pub(crate) dummy_pyramid: &'a wgpu::TextureView,
    pub(crate) dummy_sdf_field: &'a wgpu::TextureView,
    pub(crate) dummy_sdf_uniform: &'a wgpu::Buffer,
    pub(crate) parity: usize,
}

/// A create-texture helper of the resource set: one mip, one sample, `layers` array layers (or
/// the depth of a D3 texture).
fn texture(
    gpu: &GpuContext,
    label: &str,
    size: (u32, u32),
    layers: u32,
    dimension: wgpu::TextureDimension,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: layers,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension,
        format,
        usage,
        view_formats: &[],
    })
}

/// A `UNIFORM | COPY_DST` buffer of `size` bytes (the params, dispatch and SDF-uniform layout).
fn uniform(gpu: &GpuContext, label: &str, size: u64) -> wgpu::Buffer {
    gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// The default view of a texture: D2Array for an array texture, D2 for one layer, D3 for a
/// volume.
fn full_view(texture: &wgpu::Texture) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// The D2 view of one array layer of a texture.
fn layer_view(texture: &wgpu::Texture, layer: u32) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        base_array_layer: layer,
        array_layer_count: Some(1),
        dimension: Some(wgpu::TextureViewDimension::D2),
        ..Default::default()
    })
}

fn texture_resource(view: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::TextureView(view)
}

/// Creates a bind group from (binding, resource) pairs, as `ScreenResources` does.
fn group(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    bindings: &[(u32, wgpu::BindingResource<'_>)],
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry<'_>> = bindings
        .iter()
        .map(|(binding, resource)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: resource.clone(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &entries,
    })
}

/// Handoff: card 'raa-gi: AreaShadowPart and the gi.area-shadow feature pass (order 43)
/// publishing the mask' takes over `AreaShadowResources`: it calls `AreaShadowResources::new`
/// for one (viewport, groups), `AreaShadowResources::work` per frame to obtain the
/// `AreaShadowWork` it records through `AreaShadowPipelines::record`,
/// `AreaShadowResources::write_params` before the record, `AreaShadowResources::mask_view` for
/// `LightingResources::set_area_shadow`, `AreaShadowResources::invalidate_group`,
/// `AreaShadowResources::invalidate_all` and `AreaShadowResources::history_mask` for the slot
/// limit and history changes, `AreaShadowResources::commit` at the end of the record,
/// `AreaShadowResources::invalidate_bindings` on the pass resize hook and
/// `AreaShadowResources::allocated_bytes` for the stats, and the free `needs_reallocation` and
/// `resource_bytes` for the key and the byte count. Today `AreaShadowResources::new` allocates
/// the Rgba16Float raw array of max(2, groups) layers (a D2Array and one D2 view per layer), one
/// Rgba16Float history pair per history group, the R32Float depth pair, the two-layer Rgba16Float
/// mask (a D2Array and two D2 views), the 16-B params, the four 16-B dispatch buffers and the
/// 1x1 pyramid, 1x1x1 SDF field and 64-B zero SDF-uniform dummies; `work` caches the trace, the
/// per-history-group temporal and horizontal spatial and the per-mask-layer vertical spatial bind
/// groups and rebuilds them after `invalidate_bindings`, a parity change or a changed input
/// buffer or view; `commit` flips every pair and the depth pair together. LIFECYCLE:
/// AreaShadowResources is created by GiAreaShadowPass::prepare (pipeline prepare, order 43)
/// when needs_reallocation reports a new (viewport, groups), and dropped by GiPart::release (run
/// by release_resources in crates/raa-gi/src/feature.rs on every inactive frame and before a
/// reallocation) and when the slot limit drops to 0. Its cached bind groups are dropped by
/// invalidate_bindings from the pass's resize hook. Per enabled pass in order, the pipeline calls
/// resize (only when the targets generation changed) and then prepare
/// (crates/raa-pipeline/src/pipeline.rs:881-887), then records every enabled pass in order
/// (pipeline.rs:917-919). work runs in record after gi.screen (42) published the pyramid. commit
/// runs at the end of the same record, the pattern of raa-taa's TaaPass record. It extends the
/// `AreaShadowPipelines` Handoff at
/// crates/raa-gi/src/parts/area_shadow/passes.rs:200 `AreaShadowPipelines` and does not restate
/// it.
pub(crate) struct AreaShadowResources {
    viewport: (u32, u32),
    groups: u32,
    raw_view: wgpu::TextureView,
    raw_layers: Vec<wgpu::TextureView>,
    histories: Vec<HistoryPair>,
    depth: HistoryPair,
    mask_view: wgpu::TextureView,
    mask_layers: Vec<wgpu::TextureView>,
    params: wgpu::Buffer,
    dispatch: [wgpu::Buffer; 4],
    dummy_pyramid: wgpu::TextureView,
    dummy_sdf_field: wgpu::TextureView,
    dummy_sdf_uniform: wgpu::Buffer,
    bindings: Option<AreaShadowBindings>,
}

impl AreaShadowResources {
    /// Allocates the resources of one (viewport, groups): the raw history array, the history
    /// pairs, the depth pair, the mask, the uniforms and the dummies. The viewport axes are
    /// clamped to at least 1 and the history groups to 1..=4.
    pub(crate) fn new(gpu: &GpuContext, viewport: (u32, u32), groups: u32) -> AreaShadowResources {
        let viewport = (viewport.0.max(1), viewport.1.max(1));
        let groups = groups.clamp(1, 4);
        let textured = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING;

        let raw = texture(
            gpu,
            "raa-gi.area-shadow.raw",
            viewport,
            groups.max(2),
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::Rgba16Float,
            textured,
        );
        let raw_view = full_view(&raw);
        let raw_layers = (0..groups.max(2))
            .map(|layer| layer_view(&raw, layer))
            .collect();

        let histories = (0..groups)
            .map(|_| {
                HistoryPair::new(
                    gpu,
                    "raa-gi.area-shadow.history",
                    wgpu::TextureFormat::Rgba16Float,
                    viewport,
                )
            })
            .collect();
        let depth = HistoryPair::new(
            gpu,
            "raa-gi.area-shadow.depth",
            wgpu::TextureFormat::R32Float,
            viewport,
        );

        let mask = texture(
            gpu,
            "raa-gi.area-shadow.mask",
            viewport,
            2,
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::Rgba16Float,
            textured,
        );
        let mask_view = full_view(&mask);
        let mask_layers = (0..2).map(|layer| layer_view(&mask, layer)).collect();

        let params = uniform(gpu, "raa-gi.area-shadow.params", AREA_PARAMS_SIZE);
        let dispatch = std::array::from_fn(|_| {
            uniform(gpu, "raa-gi.area-shadow.dispatch", AREA_DISPATCH_SIZE)
        });

        let dummy_pyramid_texture = texture(
            gpu,
            "raa-gi.area-shadow.pyramid-dummy",
            (1, 1),
            1,
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::R32Float,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        gpu.queue().write_texture(
            dummy_pyramid_texture.as_image_copy(),
            &1.0f32.to_le_bytes(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let dummy_pyramid = full_view(&dummy_pyramid_texture);
        let dummy_sdf_field = full_view(&texture(
            gpu,
            "raa-gi.area-shadow.sdf-dummy",
            (1, 1),
            1,
            wgpu::TextureDimension::D3,
            wgpu::TextureFormat::R8Snorm,
            wgpu::TextureUsages::TEXTURE_BINDING,
        ));
        let dummy_sdf_uniform = uniform(gpu, "raa-gi.area-shadow.sdf-uniform", SDF_VOLUME_SIZE);

        AreaShadowResources {
            viewport,
            groups,
            raw_view,
            raw_layers,
            histories,
            depth,
            mask_view,
            mask_layers,
            params,
            dispatch,
            dummy_pyramid,
            dummy_sdf_field,
            dummy_sdf_uniform,
            bindings: None,
        }
    }

    /// The viewport this resource set was allocated for.
    pub(crate) fn viewport(&self) -> (u32, u32) {
        self.viewport
    }

    /// The history groups this resource set was allocated for.
    pub(crate) fn groups(&self) -> u32 {
        self.groups
    }

    /// A clone of the two-layer D2Array mask view the forward pass publishes.
    pub(crate) fn mask_view(&self) -> wgpu::TextureView {
        self.mask_view.clone()
    }

    /// Invalidates history group `group`; out-of-range groups are ignored.
    pub(crate) fn invalidate_group(&mut self, group: u32) {
        if let Some(pair) = self.histories.get_mut(group as usize) {
            pair.state_mut().invalidate();
        }
    }

    /// Invalidates every history pair and the depth pair.
    pub(crate) fn invalidate_all(&mut self) {
        for pair in &mut self.histories {
            pair.state_mut().invalidate();
        }
        self.depth.state_mut().invalidate();
    }

    /// Bit `g` is set when history pair `g` holds a committed frame.
    pub(crate) fn history_mask(&self) -> u32 {
        self.histories
            .iter()
            .enumerate()
            .fold(0, |mask, (group, pair)| {
                mask | (u32::from(pair.state().is_valid()) << group)
            })
    }

    /// Writes the 16 B of `AreaShadowParamsGpu` as (frame_index, 0, 0, 0) and every 16 B
    /// `AreaDispatchGpu` as (index, history bit, pairs of this group, min(limit, 8)).
    pub(crate) fn write_params(&self, gpu: &GpuContext, frame_index: u32, limit: u32) {
        let mut params = [0u8; AREA_PARAMS_SIZE as usize];
        params[0..4].copy_from_slice(&frame_index.to_le_bytes());
        gpu.queue().write_buffer(&self.params, 0, &params);

        let mask = self.history_mask();
        let limit = limit.min(8);
        for (index, dispatch) in self.dispatch.iter().enumerate() {
            let index = index as u32;
            let pairs = if 2 * index < self.groups {
                (self.groups - 2 * index).min(2)
            } else {
                0
            };
            let values = [index, (mask >> index) & 1, pairs, limit];
            let mut bytes = [0u8; AREA_DISPATCH_SIZE as usize];
            for (lane, value) in values.into_iter().enumerate() {
                bytes[lane * 4..lane * 4 + 4].copy_from_slice(&value.to_le_bytes());
            }
            gpu.queue().write_buffer(dispatch, 0, &bytes);
        }
    }

    /// The GPU bytes of this resource set, `resource_bytes` of its viewport and groups.
    pub(crate) fn allocated_bytes(&self) -> u64 {
        resource_bytes(self.viewport, self.groups)
    }

    /// Builds the bind groups on first use and caches them; rebuilds them after
    /// `invalidate_bindings`, a change of the depth pair's write index (the shared parity of
    /// every pair) or a changed G-buffer view or input buffer or view. The returned
    /// `AreaShadowWork` carries the trace group, `cpu::groups_for(limit).min(groups)` temporal
    /// and horizontal spatial groups, `limit.min(8).div_ceil(4)` vertical spatial groups and the
    /// viewport.
    pub(crate) fn work<'a>(
        &'a mut self,
        gpu: &GpuContext,
        pipelines: &AreaShadowPipelines,
        views: &GBufferViews,
        inputs: &AreaShadowInputs<'_>,
        limit: u32,
    ) -> AreaShadowWork<'a> {
        self.refresh(gpu, pipelines, views, inputs);
        let bindings = self.bindings.as_ref().expect("bindings built by refresh");
        let groups = cpu::groups_for(limit).min(self.groups) as usize;
        let layers = (limit.min(8).div_ceil(4) as usize).min(bindings.spatial_v.len());
        AreaShadowWork {
            trace: &bindings.trace,
            temporal: &bindings.temporal[..groups],
            spatial_h: &bindings.spatial_h[..groups],
            spatial_v: &bindings.spatial_v[..layers],
            viewport: self.viewport,
        }
    }

    /// The `AreaShadowWork` of the ReSTIR tier, under the cache rules of `work`: the trace group,
    /// `temporal[..1]`, `spatial_h[..1]`, the broadcast vertical groups
    /// `[..limit.min(8).div_ceil(4)]` (none when `limit` is 0) and the viewport. Broadcast group
    /// l binds raw layer 0 at 3 and 4, mask layer l at 5 and dispatch[l] at 7, so the vertical
    /// spatial pass writes the ratio of history group 0 into every channel of mask layer l. It
    /// is meant for allocations with groups = 1.
    ///
    /// Handoff: card 'raa-gi: gi.area-shadow part runs the ReSTIR ultra tier on Restir frames'
    /// takes over `AreaShadowResources::work_restir`: it calls `AreaShadowResources::work_restir`
    /// and `AreaShadowResources::shared_views` on Restir frames, and card 'raa-gi:
    /// RestirResources (reservoir buffers, params, LTC atlas, bind group)' reads
    /// `AreaSharedViews` from `AreaShadowResources::shared_views` to bind the raw layer, the
    /// depth pair and the dummies; today both return the views and groups described above. It
    /// cites the Handoffs at crates/raa-gi/src/parts/area_shadow/passes.rs:200
    /// `AreaShadowPipelines` and crates/raa-gi/src/parts/area_shadow/resources.rs:169
    /// `AreaShadowResources` and does not restate them.
    pub(crate) fn work_restir<'a>(
        &'a mut self,
        gpu: &GpuContext,
        pipelines: &AreaShadowPipelines,
        views: &GBufferViews,
        inputs: &AreaShadowInputs<'_>,
        limit: u32,
    ) -> AreaShadowWork<'a> {
        self.refresh(gpu, pipelines, views, inputs);
        let bindings = self.bindings.as_ref().expect("bindings built by refresh");
        let layers = (limit.min(8).div_ceil(4) as usize).min(bindings.spatial_v_broadcast.len());
        AreaShadowWork {
            trace: &bindings.trace,
            temporal: &bindings.temporal[..1],
            spatial_h: &bindings.spatial_h[..1],
            spatial_v: &bindings.spatial_v_broadcast[..layers],
            viewport: self.viewport,
        }
    }

    /// The shared views of `AreaSharedViews`; `parity` is the depth pair's write index.
    pub(crate) fn shared_views(&self) -> AreaSharedViews<'_> {
        AreaSharedViews {
            raw: &self.raw_view,
            depth_write: self.depth.write_view(),
            depth_read: self.depth.read_view(),
            dummy_pyramid: &self.dummy_pyramid,
            dummy_sdf_field: &self.dummy_sdf_field,
            dummy_sdf_uniform: &self.dummy_sdf_uniform,
            parity: self.depth.state().write_index(),
        }
    }

    /// Rebuilds the cached bind groups after `invalidate_bindings`, a parity change or a changed
    /// G-buffer view or input buffer or view.
    fn refresh(
        &mut self,
        gpu: &GpuContext,
        pipelines: &AreaShadowPipelines,
        views: &GBufferViews,
        inputs: &AreaShadowInputs<'_>,
    ) {
        let parity = self.depth.state().write_index();
        let stale = self.bindings.as_ref().is_none_or(|bindings| {
            bindings.parity != parity
                || bindings.views != *views
                || bindings.gi_frame != *inputs.gi_frame
                || bindings.lit_frame != *inputs.lit_frame
                || bindings.table != *inputs.table
                || bindings.sampler != *inputs.sampler
                || bindings.pyramid.as_ref() != inputs.pyramid
                || bindings.sdf_field.as_ref() != inputs.sdf.map(|sdf| sdf.0)
                || bindings.sdf_uniform.as_ref() != inputs.sdf.map(|sdf| sdf.1)
        });
        if stale {
            self.bindings = Some(self.build_bindings(gpu, pipelines, views, inputs, parity));
        }
    }

    /// Drops the cached bind groups, so the next `work` builds them again.
    pub(crate) fn invalidate_bindings(&mut self) {
        self.bindings = None;
    }

    /// Commits every history pair and the depth pair: one frame of history is now readable.
    pub(crate) fn commit(&mut self) {
        for pair in &mut self.histories {
            pair.state_mut().commit();
        }
        self.depth.state_mut().commit();
    }

    fn build_bindings(
        &self,
        gpu: &GpuContext,
        pipelines: &AreaShadowPipelines,
        views: &GBufferViews,
        inputs: &AreaShadowInputs<'_>,
        parity: usize,
    ) -> AreaShadowBindings {
        let device = gpu.device();
        let pyramid = inputs.pyramid.unwrap_or(&self.dummy_pyramid);
        let sdf_field = inputs.sdf.map_or(&self.dummy_sdf_field, |sdf| sdf.0);
        let sdf_uniform = inputs.sdf.map_or(&self.dummy_sdf_uniform, |sdf| sdf.1);
        let trace = group(
            device,
            "raa-gi.area-shadow.trace",
            &pipelines.trace_layout,
            &[
                (0, inputs.gi_frame.as_entire_binding()),
                (1, texture_resource(&views.depth)),
                (2, texture_resource(&views.normal)),
                (4, texture_resource(pyramid)),
                (5, texture_resource(&self.raw_view)),
                (6, texture_resource(self.depth.write_view())),
                (8, inputs.table.as_entire_binding()),
                (9, self.params.as_entire_binding()),
                (10, texture_resource(sdf_field)),
                (11, sdf_uniform.as_entire_binding()),
            ],
        );
        let temporal = self
            .histories
            .iter()
            .enumerate()
            .map(|(index, pair)| {
                group(
                    device,
                    "raa-gi.area-shadow.temporal",
                    &pipelines.temporal_layout,
                    &[
                        (0, inputs.lit_frame.as_entire_binding()),
                        (1, texture_resource(self.depth.write_view())),
                        (2, texture_resource(&views.motion)),
                        (3, texture_resource(&self.raw_layers[index])),
                        (4, texture_resource(pair.read_view())),
                        (5, texture_resource(self.depth.read_view())),
                        (6, texture_resource(pair.write_view())),
                        (7, self.dispatch[index].as_entire_binding()),
                        (8, wgpu::BindingResource::Sampler(inputs.sampler)),
                    ],
                )
            })
            .collect();
        let spatial_h = self
            .histories
            .iter()
            .enumerate()
            .map(|(index, pair)| {
                group(
                    device,
                    "raa-gi.area-shadow.spatial-h",
                    &pipelines.spatial_layout,
                    &[
                        (1, texture_resource(self.depth.write_view())),
                        (2, texture_resource(&views.normal)),
                        (3, texture_resource(pair.write_view())),
                        (4, texture_resource(pair.write_view())),
                        (5, texture_resource(&self.raw_layers[index])),
                        (7, self.dispatch[index].as_entire_binding()),
                    ],
                )
            })
            .collect();
        let spatial_v = (0..self
            .raw_layers
            .len()
            .div_ceil(2)
            .min(self.mask_layers.len()))
            .map(|layer| {
                let first = layer * 2;
                let second = if first + 1 < self.groups as usize {
                    first + 1
                } else {
                    first
                };
                group(
                    device,
                    "raa-gi.area-shadow.spatial-v",
                    &pipelines.spatial_layout,
                    &[
                        (1, texture_resource(self.depth.write_view())),
                        (2, texture_resource(&views.normal)),
                        (3, texture_resource(&self.raw_layers[first])),
                        (4, texture_resource(&self.raw_layers[second])),
                        (5, texture_resource(&self.mask_layers[layer])),
                        (7, self.dispatch[layer].as_entire_binding()),
                    ],
                )
            })
            .collect();
        let spatial_v_broadcast = self
            .mask_layers
            .iter()
            .enumerate()
            .map(|(layer, mask)| {
                group(
                    device,
                    "raa-gi.area-shadow.spatial-v-broadcast",
                    &pipelines.spatial_layout,
                    &[
                        (1, texture_resource(self.depth.write_view())),
                        (2, texture_resource(&views.normal)),
                        (3, texture_resource(&self.raw_layers[0])),
                        (4, texture_resource(&self.raw_layers[0])),
                        (5, texture_resource(mask)),
                        (7, self.dispatch[layer].as_entire_binding()),
                    ],
                )
            })
            .collect();
        AreaShadowBindings {
            gi_frame: inputs.gi_frame.clone(),
            lit_frame: inputs.lit_frame.clone(),
            table: inputs.table.clone(),
            sampler: inputs.sampler.clone(),
            pyramid: inputs.pyramid.cloned(),
            sdf_field: inputs.sdf.map(|sdf| sdf.0.clone()),
            sdf_uniform: inputs.sdf.map(|sdf| sdf.1.clone()),
            views: views.clone(),
            parity,
            trace,
            temporal,
            spatial_h,
            spatial_v,
            spatial_v_broadcast,
        }
    }
}

/// The GPU bytes of one `AreaShadowResources` of `viewport` and `groups`: the raw array at 8 B
/// per texel and max(2, groups) layers, the history pairs at 16 B per texel and group, the
/// R32Float depth pair at 8 B, the two-layer mask at 16 B, the 16-B params, the four 16-B
/// dispatch buffers, the 4-B pyramid dummy, the 1-B SDF dummy and the 64-B SDF uniform. For
/// 1280x720 with 2 groups it is 66,355,349.
pub(crate) fn resource_bytes(viewport: (u32, u32), groups: u32) -> u64 {
    let width = u64::from(viewport.0.max(1));
    let height = u64::from(viewport.1.max(1));
    let groups = u64::from(groups.clamp(1, 4));
    let texels = width * height;
    texels * (8 * groups.max(2) + 16 * groups + 8 + 16)
        + AREA_PARAMS_SIZE
        + 4 * AREA_DISPATCH_SIZE
        + 4
        + 1
        + SDF_VOLUME_SIZE
}

/// True when `current` is absent or is not (viewport, groups); the viewport axes are clamped to
/// at least 1 and the groups to 1..=4 as `AreaShadowResources::new` does.
pub(crate) fn needs_reallocation(
    current: Option<((u32, u32), u32)>,
    viewport: (u32, u32),
    groups: u32,
) -> bool {
    current != Some(((viewport.0.max(1), viewport.1.max(1)), groups.clamp(1, 4)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parts::GiShared;
    use crate::parts::area_shadow::AREA_SHADOW_PASS;
    use raa_core::testing::{block_on, skip};
    use raa_pipeline::{ResourceTable, Targets};

    #[test]
    fn resource_bytes_matches_the_hand_derivation() {
        assert_eq!(resource_bytes((1280, 720), 2), 66_355_349);
        assert_eq!(resource_bytes((0, 0), 0), 16 + 16 + 8 + 16 + 149);
        assert_eq!(resource_bytes((0, 0), 0), 205);
    }

    #[test]
    fn needs_reallocation_tracks_the_clamped_key() {
        assert!(needs_reallocation(None, (1280, 720), 2));
        let current = Some(((1280, 720), 2));
        assert!(!needs_reallocation(current, (1280, 720), 2));
        assert!(needs_reallocation(current, (1920, 1080), 2));
        assert!(needs_reallocation(current, (1280, 720), 3));
        assert!(!needs_reallocation(Some(((1, 1), 1)), (0, 0), 0));
    }

    #[test]
    fn resources_allocate_and_bind_in_a_validation_scope() {
        let test = "resources_allocate_and_bind_in_a_validation_scope";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let shared = GiShared::new(&gpu);
        let pipelines = AreaShadowPipelines::new(&gpu, &shared.compute_group1_layout());
        let targets = Targets::new(&gpu, (64, 32));
        let views = GBufferViews::resolve(&targets, &ResourceTable::new(), AREA_SHADOW_PASS);
        let gi_frame = uniform(&gpu, "raa-gi.area-shadow.test.frame", 432);
        let lit_frame = uniform(&gpu, "raa-gi.area-shadow.test.lit-frame", 768);
        let table = uniform(&gpu, "raa-gi.area-shadow.test.table", 80);
        let sampler = gpu.device().create_sampler(&wgpu::SamplerDescriptor {
            label: Some("raa-gi.area-shadow.test.sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let scope = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
        let mut resources = AreaShadowResources::new(&gpu, (64, 32), 2);
        assert_eq!(resources.viewport(), (64, 32));
        assert_eq!(resources.groups(), 2);
        assert_eq!(resources.allocated_bytes(), resource_bytes((64, 32), 2));
        let work = resources.work(
            &gpu,
            &pipelines,
            &views,
            &AreaShadowInputs {
                gi_frame: &gi_frame,
                lit_frame: &lit_frame,
                table: &table,
                sampler: &sampler,
                pyramid: None,
                sdf: None,
            },
            3,
        );
        assert_eq!(work.temporal.len(), 2);
        assert_eq!(work.spatial_h.len(), 2);
        assert_eq!(work.spatial_v.len(), 1);
        assert_eq!(work.viewport, (64, 32));
        assert!(
            block_on(scope.pop()).is_none(),
            "the resources and bind groups create without a validation error"
        );

        assert_eq!(resources.history_mask(), 0);
        resources.commit();
        assert_eq!(resources.history_mask(), 0b11);
        resources.invalidate_group(1);
        assert_eq!(resources.history_mask(), 0b01);
    }

    #[test]
    fn work_restir_slices_the_broadcast_groups_in_a_validation_scope() {
        let test = "work_restir_slices_the_broadcast_groups_in_a_validation_scope";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let shared = GiShared::new(&gpu);
        let pipelines = AreaShadowPipelines::new(&gpu, &shared.compute_group1_layout());
        let targets = Targets::new(&gpu, (64, 32));
        let views = GBufferViews::resolve(&targets, &ResourceTable::new(), AREA_SHADOW_PASS);
        let gi_frame = uniform(&gpu, "raa-gi.area-shadow.test.frame", 432);
        let lit_frame = uniform(&gpu, "raa-gi.area-shadow.test.lit-frame", 768);
        let table = uniform(&gpu, "raa-gi.area-shadow.test.table", 80);
        let sampler = gpu.device().create_sampler(&wgpu::SamplerDescriptor {
            label: Some("raa-gi.area-shadow.test.sampler"),
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
        let mut resources = AreaShadowResources::new(&gpu, (64, 32), 1);
        let work = resources.work_restir(&gpu, &pipelines, &views, &inputs, 8);
        assert_eq!(work.temporal.len(), 1);
        assert_eq!(work.spatial_h.len(), 1);
        assert_eq!(work.spatial_v.len(), 2);
        assert_eq!(work.viewport, (64, 32));
        let work = resources.work_restir(&gpu, &pipelines, &views, &inputs, 3);
        assert_eq!(work.spatial_v.len(), 1);
        let work = resources.work_restir(&gpu, &pipelines, &views, &inputs, 0);
        assert_eq!(work.spatial_v.len(), 0);
        assert!(
            block_on(scope.pop()).is_none(),
            "the broadcast bind groups create without a validation error"
        );

        let before = resources.shared_views().parity;
        assert_eq!(before, resources.depth.state().write_index());
        resources.commit();
        let after = resources.shared_views().parity;
        assert_eq!(after, resources.depth.state().write_index());
        assert_ne!(before, after);
    }

    /// The G-buffer views are part of the cache key: `AreaShadowResources::work` keeps the cached
    /// trace and temporal groups for equal views and rebuilds them when the `normal` view differs.
    #[test]
    fn work_rebuilds_the_cached_groups_when_the_normal_view_changes() {
        let test = "work_rebuilds_the_cached_groups_when_the_normal_view_changes";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let shared = GiShared::new(&gpu);
        let pipelines = AreaShadowPipelines::new(&gpu, &shared.compute_group1_layout());
        let targets = Targets::new(&gpu, (64, 32));
        let gi_frame = uniform(&gpu, "raa-gi.area-shadow.test.frame", 432);
        let lit_frame = uniform(&gpu, "raa-gi.area-shadow.test.lit-frame", 768);
        let table = uniform(&gpu, "raa-gi.area-shadow.test.table", 80);
        let sampler = gpu.device().create_sampler(&wgpu::SamplerDescriptor {
            label: Some("raa-gi.area-shadow.test.sampler"),
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
        let plain = GBufferViews::resolve(&targets, &ResourceTable::new(), AREA_SHADOW_PASS);
        let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("raa-gi.area-shadow.test-bound-normal"),
            size: wgpu::Extent3d {
                width: 64,
                height: 32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let mut bound = plain.clone();
        bound.normal = texture.create_view(&wgpu::TextureViewDescriptor::default());
        assert_ne!(bound.normal, plain.normal);

        let mut resources = AreaShadowResources::new(&gpu, (64, 32), 2);
        let first = {
            let work = resources.work(&gpu, &pipelines, &plain, &inputs, 3);
            (work.trace.clone(), work.temporal[0].clone())
        };
        let kept = {
            let work = resources.work(&gpu, &pipelines, &plain, &inputs, 3);
            (work.trace.clone(), work.temporal[0].clone())
        };
        assert_eq!(first, kept);
        let rebuilt = {
            let work = resources.work(&gpu, &pipelines, &bound, &inputs, 3);
            (work.trace.clone(), work.temporal[0].clone())
        };
        assert_ne!(first.0, rebuilt.0);
        assert_ne!(first.1, rebuilt.1);
        let cached = resources.bindings.as_ref().expect("bindings built above");
        assert!(cached.views == bound);
        assert_eq!(cached.views.normal, bound.normal);
        assert_eq!(cached.views.depth, plain.depth);
    }
}
