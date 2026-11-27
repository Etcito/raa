//! The GPU resources of `gi.ssgi` (contract 'ssgi-resources'): the raw Rgba16Float trace texture,
//! the `raa_taa::HistoryPair` read by the resolve, the Rgba16Float output, the 32-B params uniform,
//! two 1x1 (0, 0, 0, 1) dummies and the cached trace and resolve bind groups. The bind groups use
//! the layouts of `SsgiPipelines` (crates/raa-gi/src/parts/ssgi/passes.rs:116); the helpers follow
//! `ScreenResources` (crates/raa-gi/src/parts/screen.rs:87).
#![allow(dead_code, reason = "the SsgiPart card is the first caller")]

use raa_core::GpuContext;
use raa_taa::HistoryPair;

use super::cpu;
use super::passes::{SSGI_PARAMS_SIZE, SsgiPipelines, SsgiWork};
use super::ssgi_bytes;
use crate::parts::GBufferViews;

/// Format of the raw, history, output and dummy textures.
const SSGI_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Resolve groups kept at once: the probe parity times the pair parity.
const RESOLVE_SLOTS: usize = 4;

/// The 1x1 Rgba16Float texel (0, 0, 0, 1) as little-endian f16 values.
const DUMMY_TEXEL: [u8; 8] = [0, 0, 0, 0, 0, 0, 0x00, 0x3C];

/// The trace group 2 cached for one `gi_frame` buffer, one previous lit HDR view and one set of
/// G-buffer views.
struct TraceBindings {
    gi_frame: wgpu::Buffer,
    history: Option<wgpu::TextureView>,
    /// The voxel-buffer generation binding 6 was built with (None: the dummy).
    voxels: Option<u64>,
    /// The `depth`, `normal` and `motion` views bindings 1, 2 and the resolve bindings were built
    /// with.
    views: GBufferViews,
    group: wgpu::BindGroup,
}

/// One cached resolve group 0, keyed by its probe view (None for the dummy) and the pair's write
/// index.
struct ResolveBindings {
    probe: Option<wgpu::TextureView>,
    write_index: usize,
    group: wgpu::BindGroup,
}

/// The per-frame inputs `SsgiResources::work` binds: the GiFrameGpu buffer, the lit frame buffer
/// (`LitHandle::frame_buffer`), the previous lit HDR (`GiShared::history_view`) and the probe
/// output (`GiShared::probe_output_view`). A None history or probe binds its dummy.
pub(crate) struct SsgiInputs<'a> {
    pub(crate) gi_frame: &'a wgpu::Buffer,
    pub(crate) lit_frame: &'a wgpu::Buffer,
    pub(crate) history: Option<&'a wgpu::TextureView>,
    pub(crate) probe: Option<&'a wgpu::TextureView>,
}

/// `SsgiResources` holds the GPU resources of `gi.ssgi` for one viewport: the raw texture, one
/// `raa_taa::HistoryPair` (label "raa-gi.ssgi.history"), the output, the 32-B params uniform, two
/// 1x1 (0, 0, 0, 1) dummies (lit-HDR history and probe output) and the cached bind groups.
///
/// Handoff: `SsgiResources` is taken over by card 'raa-gi ssgi: SsgiPart, the gi.ssgi pass (order
/// 44) and the binding-14 pick', which calls `SsgiResources::new` per viewport, `write_params` and
/// `work` per frame (its `SsgiWork` is dispatched by `SsgiPipelines::record`),
/// `commit` and `invalidate` for the history, `invalidate_bindings` from the resize hook,
/// `output_view` for the binding-14 output and `allocated_bytes` for the stats. Today nothing
/// outside the tests calls those methods, and `new` allocates the resources listed above.
/// LIFECYCLE (contract 'ssgi-resources', realised by that card): SsgiPart::prepare_frame (a GiPart
/// hook run by GiHandle::prepare_frame before the pipeline executes) creates SsgiResources on an
/// active frame whose viewport key changed, and drops them on an inactive frame. GiPart::release
/// (release_resources in crates/raa-gi/src/feature.rs) drops them. The bind groups are built lazily
/// by `work` from the gi.ssgi record and dropped by `invalidate_bindings` from the gi.ssgi resize
/// hook. Per enabled pass in order, the pipeline calls resize (only when the targets generation
/// changed) and then prepare (crates/raa-pipeline/src/pipeline.rs:881-887), then records every
/// enabled pass in order (pipeline.rs:917-919). So gi.volume (25) prepares binding 14, gi.screen
/// (42) records the probe output and gi.ssgi (44) records after it.
pub(crate) struct SsgiResources {
    viewport: (u32, u32),
    raw: wgpu::TextureView,
    history: HistoryPair,
    output: wgpu::TextureView,
    params: wgpu::Buffer,
    history_dummy: wgpu::TextureView,
    probe_dummy: wgpu::TextureView,
    /// The 16-B zeroed storage buffer trace binding 6 reads while `GiShared::voxel_buffer` is None.
    voxel_dummy: wgpu::Buffer,
    lit_frame: Option<wgpu::Buffer>,
    trace: Option<TraceBindings>,
    resolve: Vec<ResolveBindings>,
}

/// A clamped-size Rgba16Float texture with the given usages.
fn texture(
    gpu: &GpuContext,
    label: &str,
    size: (u32, u32),
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: SSGI_FORMAT,
        usage,
        view_formats: &[],
    })
}

fn full_view(texture: &wgpu::Texture) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

fn buffer(gpu: &GpuContext, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

/// A 1x1 Rgba16Float view holding (0, 0, 0, 1), written with `write_texture`.
fn dummy(gpu: &GpuContext, label: &str) -> wgpu::TextureView {
    let image = texture(
        gpu,
        label,
        (1, 1),
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    );
    gpu.queue().write_texture(
        image.as_image_copy(),
        &DUMMY_TEXEL,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(8),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    full_view(&image)
}

fn texture_resource(view: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::TextureView(view)
}

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

impl SsgiResources {
    /// Allocates the resources of `viewport`, each axis clamped to at least 1. No bind group is
    /// built here: `work` builds them on first use.
    pub(crate) fn new(gpu: &GpuContext, viewport: (u32, u32)) -> SsgiResources {
        let viewport = (viewport.0.max(1), viewport.1.max(1));
        let textured = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING;
        let raw = texture(gpu, "raa-gi.ssgi.raw", viewport, textured);
        let output = texture(gpu, "raa-gi.ssgi.output", viewport, textured);
        SsgiResources {
            viewport,
            raw: full_view(&raw),
            history: HistoryPair::new(gpu, "raa-gi.ssgi.history", SSGI_FORMAT, viewport),
            output: full_view(&output),
            params: buffer(
                gpu,
                "raa-gi.ssgi.params",
                SSGI_PARAMS_SIZE,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            history_dummy: dummy(gpu, "raa-gi.ssgi.history-dummy"),
            probe_dummy: dummy(gpu, "raa-gi.ssgi.probe-dummy"),
            voxel_dummy: buffer(
                gpu,
                "raa-gi.ssgi.voxel-dummy",
                16,
                wgpu::BufferUsages::STORAGE,
            ),
            lit_frame: None,
            trace: None,
            resolve: Vec::new(),
        }
    }

    /// The viewport this resource set was allocated for, each axis at least 1.
    pub(crate) fn viewport(&self) -> (u32, u32) {
        self.viewport
    }

    /// A clone of the output view: the Rgba16Float texture the resolve writes at group 0 binding 7.
    pub(crate) fn output_view(&self) -> wgpu::TextureView {
        self.output.clone()
    }

    /// The raw trace view the gi.ssgi tap `raw` publishes.
    pub(crate) fn raw_view(&self) -> &wgpu::TextureView {
        &self.raw
    }

    /// The history half this frame's pass writes: the pair's current write view.
    pub(crate) fn history_write_view(&self) -> &wgpu::TextureView {
        self.history.write_view()
    }

    /// Writes the 32 B of `SsgiParamsGpu` for this frame; `temporal_valid` is read from the pair.
    pub(crate) fn write_params(
        &self,
        gpu: &GpuContext,
        slices: u32,
        steps: u32,
        history_valid: bool,
        frame: u32,
    ) {
        let temporal_valid = self.history.state().is_valid();
        let bytes = params_bytes(slices, steps, history_valid, frame, temporal_valid);
        gpu.queue().write_buffer(&self.params, 0, &bytes);
    }

    /// `ssgi_bytes` of this resource set's viewport.
    pub(crate) fn allocated_bytes(&self) -> u64 {
        ssgi_bytes(self.viewport)
    }

    /// Commits the recorded frame to the pair: its write index flips and one frame is counted.
    pub(crate) fn commit(&mut self) {
        self.history.state_mut().commit();
    }

    /// Drops the pair's accumulated frames, so the next resolve reads no history.
    pub(crate) fn invalidate(&mut self) {
        self.history.state_mut().invalidate();
    }

    /// Drops the cached trace and resolve bind groups; the next `work` builds them again.
    pub(crate) fn invalidate_bindings(&mut self) {
        self.trace = None;
        self.resolve.clear();
    }

    /// The trace group 2 and the resolve group 0 of the current pair parity, built on first use.
    /// The trace group is rebuilt only when `inputs.gi_frame`, `inputs.history` or `views` differs
    /// from the cached one, or the voxel-buffer generation (`SsgiPipelines.voxels`) differs. The
    /// resolve groups (at most `RESOLVE_SLOTS`, keyed by the probe view or none and the pair's
    /// write index) are dropped whenever the trace group is rebuilt or `inputs.lit_frame` changes.
    pub(crate) fn work<'a>(
        &'a mut self,
        gpu: &GpuContext,
        pipelines: &SsgiPipelines,
        views: &GBufferViews,
        inputs: &SsgiInputs<'_>,
    ) -> SsgiWork<'a> {
        let voxels = pipelines.voxels.as_ref();
        let stale = self.trace.as_ref().is_none_or(|cached| {
            cached.gi_frame != *inputs.gi_frame
                || cached.history.as_ref() != inputs.history
                || cached.voxels != voxels.map(|v| v.1)
                || cached.views != *views
        });
        if stale {
            self.trace = None;
            self.resolve.clear();
        }
        if self.lit_frame.as_ref() != Some(inputs.lit_frame) {
            self.lit_frame = Some(inputs.lit_frame.clone());
            self.resolve.clear();
        }
        let trace: &'a TraceBindings = self.trace.get_or_insert_with(|| TraceBindings {
            gi_frame: inputs.gi_frame.clone(),
            history: inputs.history.cloned(),
            voxels: voxels.map(|v| v.1),
            views: views.clone(),
            group: group(
                gpu.device(),
                "raa-gi.ssgi.trace",
                &pipelines.trace_layout,
                &[
                    (0, inputs.gi_frame.as_entire_binding()),
                    (1, texture_resource(&views.depth)),
                    (2, texture_resource(&views.normal)),
                    (
                        3,
                        texture_resource(inputs.history.unwrap_or(&self.history_dummy)),
                    ),
                    (5, texture_resource(&self.raw)),
                    (
                        6,
                        voxels
                            .map_or(&self.voxel_dummy, |v| &v.0)
                            .as_entire_binding(),
                    ),
                    (8, self.params.as_entire_binding()),
                ],
            ),
        });
        let write_index = self.history.state().write_index();
        let probe = inputs.probe;
        let found = self
            .resolve
            .iter()
            .position(|slot| slot.probe.as_ref() == probe && slot.write_index == write_index);
        let index = match found {
            Some(index) => index,
            None => {
                if self.resolve.len() == RESOLVE_SLOTS {
                    self.resolve.remove(0);
                }
                let resolve = group(
                    gpu.device(),
                    "raa-gi.ssgi.resolve",
                    &pipelines.resolve_layout,
                    &[
                        (0, inputs.lit_frame.as_entire_binding()),
                        (1, texture_resource(&views.depth)),
                        (2, texture_resource(&views.motion)),
                        (3, texture_resource(&self.raw)),
                        (4, texture_resource(self.history.read_view())),
                        (5, texture_resource(probe.unwrap_or(&self.probe_dummy))),
                        (6, texture_resource(self.history.write_view())),
                        (7, texture_resource(&self.output)),
                        (8, self.params.as_entire_binding()),
                    ],
                );
                self.resolve.push(ResolveBindings {
                    probe: probe.cloned(),
                    write_index,
                    group: resolve,
                });
                self.resolve.len() - 1
            }
        };
        SsgiWork {
            trace: &trace.group,
            resolve: &self.resolve[index].group,
            viewport: self.viewport,
        }
    }
}

/// The 32 B of `SsgiParamsGpu`, little-endian: slices, steps, history_valid and frame (u32),
/// radius_m and thickness_m (f32, `cpu::RADIUS_METRES` and `cpu::THICKNESS_METRES`), temporal_valid
/// (u32) and a zero pad (u32). `write_params` uploads exactly these bytes.
fn params_bytes(
    slices: u32,
    steps: u32,
    history_valid: bool,
    frame: u32,
    temporal_valid: bool,
) -> [u8; 32] {
    let words = [
        slices.to_le_bytes(),
        steps.to_le_bytes(),
        u32::from(history_valid).to_le_bytes(),
        frame.to_le_bytes(),
        cpu::RADIUS_METRES.to_le_bytes(),
        cpu::THICKNESS_METRES.to_le_bytes(),
        u32::from(temporal_valid).to_le_bytes(),
        0u32.to_le_bytes(),
    ];
    let mut bytes = [0u8; 32];
    for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
        chunk.copy_from_slice(&word);
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parts::GiShared;
    use raa_core::testing::skip;
    use raa_pipeline::Targets;

    #[test]
    fn allocated_bytes_matches_ssgi_bytes() {
        let test = "allocated_bytes_matches_ssgi_bytes";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let resources = SsgiResources::new(&gpu, (320, 180));
        assert_eq!(resources.allocated_bytes(), 1_843_248);
        let clamped = SsgiResources::new(&gpu, (0, 0));
        assert_eq!(clamped.viewport(), (1, 1));
        assert_eq!(clamped.allocated_bytes(), 80);
    }

    /// `work` keeps its cached groups while the `GBufferViews` stay the same and rebuilds them
    /// when one of the views (here `motion`) differs. The trace group binds `depth` and `normal`,
    /// so only the cache key carries the `motion` view; the resolve group binds it.
    #[test]
    fn work_rebuilds_when_the_gbuffer_views_change() {
        let test = "work_rebuilds_when_the_gbuffer_views_change";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let shared = GiShared::new(&gpu);
        let pipelines = SsgiPipelines::new(&gpu, &shared.compute_group1_layout());
        let mut resources = SsgiResources::new(&gpu, (8, 8));
        let gi_frame = buffer(
            &gpu,
            "raa-gi.ssgi.test.gi-frame",
            1024,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let lit_frame = buffer(
            &gpu,
            "raa-gi.ssgi.test.lit-frame",
            1024,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let inputs = SsgiInputs {
            gi_frame: &gi_frame,
            lit_frame: &lit_frame,
            history: None,
            probe: None,
        };
        let targets = Targets::new(&gpu, (8, 8));
        let views = GBufferViews {
            depth: targets.depth().clone(),
            normal: targets.normal().clone(),
            material: targets.material().clone(),
            motion: targets.motion().clone(),
        };
        let first = resources
            .work(&gpu, &pipelines, &views, &inputs)
            .trace
            .clone();
        let first_resolve = resources
            .work(&gpu, &pipelines, &views, &inputs)
            .resolve
            .clone();
        let same = resources
            .work(&gpu, &pipelines, &views, &inputs)
            .trace
            .clone();
        assert_eq!(first, same, "the same views must keep the cached groups");
        let changed = GBufferViews {
            motion: dummy(&gpu, "raa-gi.ssgi.test.other-motion"),
            ..views.clone()
        };
        let rebuilt = resources
            .work(&gpu, &pipelines, &changed, &inputs)
            .trace
            .clone();
        assert_ne!(
            first, rebuilt,
            "a changed motion must rebuild the trace group"
        );
        let rebuilt_resolve = resources
            .work(&gpu, &pipelines, &changed, &inputs)
            .resolve
            .clone();
        assert_ne!(
            first_resolve, rebuilt_resolve,
            "a changed motion must rebuild the resolve group"
        );
        let kept = resources
            .work(&gpu, &pipelines, &changed, &inputs)
            .trace
            .clone();
        assert_eq!(rebuilt, kept, "the new views must stay cached");
    }

    #[test]
    fn params_bytes_follow_the_layout() {
        let bytes = params_bytes(4, 8, true, 7, false);
        let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        let float = |at: usize| f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        assert_eq!([word(0), word(4), word(8), word(12)], [4, 8, 1, 7]);
        assert_eq!(float(16), 2.0);
        assert_eq!(float(20), 0.5);
        assert_eq!([word(24), word(28)], [0, 0]);
    }
}
