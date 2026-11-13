//! The S32 hierarchical depth pyramid of RAA's virtualized geometry (module `vg-hzb`) and its
//! two WGSL module builders. Adapted from solstice (MIT), Copyright (c) 2026 SamG-Coder:
//! `crates/solstice-render/src/vg/hzb.rs:1-127`,
//! `crates/solstice-render/shaders/vg/hzb_seed.wgsl` and `hzb_down.wgsl`, revision 7e465b5.
//! Solstice reduces reversed-Z depths (farthest = `min`); RAA reduces standard-Z depths to
//! the farthest through `raa_depth_farthest`, starting from the nearest possible depth 0.0, so
//! a texel whose pixels are all empty ends at the far plane 1.0.

use crate::culling::PASS_VIEW_WGSL;
use raa_core::{GpuContext, wgpu};
use std::sync::Mutex;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
const HZB_SEED_WGSL: &str = include_str!("wgsl/hzb_seed.wgsl");
const HZB_DOWN_WGSL: &str = include_str!("wgsl/hzb_down.wgsl");

/// The seed module: `raa_shader::DEPTH_WGSL` + [`PASS_VIEW_WGSL`] + `hzb_seed.wgsl`, with the
/// entry points `seed_from_keys` (group 0: 0 params uniform, 1 `vis_depth` read-only storage,
/// 2 output r32float storage texture) and `seed_from_depth` (group 0: 0 params, 3
/// `texture_depth_2d`, 2 output).
pub fn hzb_seed_module_wgsl() -> String {
    [raa_shader::DEPTH_WGSL, PASS_VIEW_WGSL, HZB_SEED_WGSL].concat()
}

/// The downsample module: `raa_shader::DEPTH_WGSL` + [`PASS_VIEW_WGSL`] + `hzb_down.wgsl`,
/// with the entry point `main` (group 0: 0 source mip `texture_2d<f32>`, 1 output storage
/// texture).
pub fn hzb_down_module_wgsl() -> String {
    [raa_shader::DEPTH_WGSL, PASS_VIEW_WGSL, HZB_DOWN_WGSL].concat()
}

/// The pyramid size: the previous power of two of each viewport axis
/// (solstice@7e465b5 `hzb.rs:118-120`).
pub fn hzb_size(viewport: (u32, u32)) -> (u32, u32) {
    (prev_power_of_two(viewport.0), prev_power_of_two(viewport.1))
}

/// The largest power of two at most `value`, and at least one.
fn prev_power_of_two(value: u32) -> u32 {
    1 << (31 - value.max(1).leading_zeros())
}

fn buffer(gpu: &GpuContext, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

fn bind_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

/// A cached `seed_from_depth` bind group with the view it was built for.
struct DepthSeed {
    view: wgpu::TextureView,
    group: wgpu::BindGroup,
}

/// A previous-power-of-two R32Float depth pyramid holding the farthest depth of each area,
/// rebuilt from the visibility depth keys or from a Depth32Float depth texture. The seed bind
/// group depends on its source, so both sources cache theirs.
pub struct Hzb {
    device: wgpu::Device,
    /// All mips, for the occlusion test.
    view: wgpu::TextureView,
    /// Mip 0 alone, the seed's storage texture output.
    first: wgpu::TextureView,
    size: (u32, u32),
    allocated_bytes: u64,
    params: wgpu::Buffer,
    seed_keys: wgpu::ComputePipeline,
    seed_depth: wgpu::ComputePipeline,
    down: wgpu::ComputePipeline,
    /// One bind group per mip after the first, with that mip's size.
    down_groups: Vec<(wgpu::BindGroup, (u32, u32))>,
    keys: Mutex<Option<(wgpu::Buffer, wgpu::BindGroup)>>,
    depth: Mutex<Option<DepthSeed>>,
}

impl Hzb {
    /// An R32Float pyramid of [`hzb_size`]`(size)`, down to 1x1, and the seed and downsample
    /// pipelines. `size` is the viewport the depth source is read at.
    pub fn new(gpu: &GpuContext, size: (u32, u32)) -> Hzb {
        let device = gpu.device();
        let target = hzb_size(size);
        let mip_count = 32 - target.0.max(target.1).leading_zeros();
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("raa-vg hzb"),
            size: wgpu::Extent3d {
                width: target.0,
                height: target.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let mip_view = |mip: u32| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("raa-vg hzb mip"),
                base_mip_level: mip,
                mip_level_count: Some(1),
                ..Default::default()
            })
        };

        let params = buffer(
            gpu,
            "raa-vg hzb params",
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let mut params_bytes = Vec::with_capacity(16);
        for value in [size.0, size.1, target.0, target.1] {
            params_bytes.extend_from_slice(&value.to_le_bytes());
        }
        gpu.queue().write_buffer(&params, 0, &params_bytes);

        let seed_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa-vg hzb seed"),
            source: wgpu::ShaderSource::Wgsl(hzb_seed_module_wgsl().into()),
        });
        let down_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa-vg hzb down"),
            source: wgpu::ShaderSource::Wgsl(hzb_down_module_wgsl().into()),
        });
        let pipeline = |label: &str, module: &wgpu::ShaderModule, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let seed_keys = pipeline("raa-vg hzb seed keys", &seed_module, "seed_from_keys");
        let seed_depth = pipeline("raa-vg hzb seed depth", &seed_module, "seed_from_depth");
        let down = pipeline("raa-vg hzb down", &down_module, "main");

        let down_groups: Vec<(wgpu::BindGroup, (u32, u32))> = (1..mip_count)
            .map(|mip| {
                let source = mip_view(mip - 1);
                let output = mip_view(mip);
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("raa-vg hzb down"),
                    layout: &down.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&source),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&output),
                        },
                    ],
                });
                (group, ((target.0 >> mip).max(1), (target.1 >> mip).max(1)))
            })
            .collect();
        let allocated_bytes = params.size()
            + (0..mip_count)
                .map(|mip| {
                    u64::from((target.0 >> mip).max(1)) * u64::from((target.1 >> mip).max(1)) * 4
                })
                .sum::<u64>();

        Hzb {
            device: device.clone(),
            view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
            first: mip_view(0),
            size: target,
            allocated_bytes,
            params,
            seed_keys,
            seed_depth,
            down,
            down_groups,
            keys: Mutex::new(None),
            depth: Mutex::new(None),
        }
    }

    /// All mips of the pyramid, for the occlusion test.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// The pyramid size: the previous power of two of the viewport passed to [`Hzb::new`].
    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Seeds every texel from the visibility depth keys through `raa_depth_from_key`, then
    /// downsamples. Key 0 is far or empty, so a texel whose pixels are all empty ends at 1.0.
    pub fn record_from_keys(&self, encoder: &mut wgpu::CommandEncoder, vis_depth: &wgpu::Buffer) {
        let mut cached = self.keys.lock().unwrap_or_else(|error| error.into_inner());
        if cached
            .as_ref()
            .is_none_or(|(buffer, _)| buffer != vis_depth)
        {
            // The bind group only names the buffer, never its contents, so a buffer's own
            // handle identity is the cheap test the caller cannot spoof.
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("raa-vg hzb seed keys"),
                layout: &self.seed_keys.get_bind_group_layout(0),
                entries: &[
                    bind_entry(0, &self.params),
                    bind_entry(1, vis_depth),
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.first),
                    },
                ],
            });
            *cached = Some((vis_depth.clone(), group));
        }
        let group = &cached.as_ref().expect("the keys group is cached").1;
        self.record(encoder, &self.seed_keys, group);
    }

    /// Seeds every texel from a Depth32Float texture, then downsamples.
    pub fn record_from_depth(&self, encoder: &mut wgpu::CommandEncoder, depth: &wgpu::TextureView) {
        let mut cached = self.depth.lock().unwrap_or_else(|error| error.into_inner());
        if cached.as_ref().is_none_or(|cache| cache.view != *depth) {
            // The seed bind group depends on the depth view. `TextureView` compares and
            // clones by the resource's own identifier, a generation the caller cannot
            // change: a new view always rebuilds, the same view reuses the cached group.
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("raa-vg hzb seed depth"),
                layout: &self.seed_depth.get_bind_group_layout(0),
                entries: &[
                    bind_entry(0, &self.params),
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.first),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(depth),
                    },
                ],
            });
            *cached = Some(DepthSeed {
                view: depth.clone(),
                group,
            });
        }
        let group = &cached.as_ref().expect("the depth group is cached").group;
        self.record(encoder, &self.seed_depth, group);
    }

    /// The GPU memory the pyramid owns: its 16-byte params buffer and four bytes per texel
    /// over every mip.
    pub fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes
    }

    /// One seed dispatch, then one dispatch per mip after the first.
    fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        seed: &wgpu::ComputePipeline,
        group: &wgpu::BindGroup,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("raa-vg hzb"),
            timestamp_writes: None,
        });
        pass.set_pipeline(seed);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(self.size.0.div_ceil(8), self.size.1.div_ceil(8), 1);
        pass.set_pipeline(&self.down);
        for (group, size) in &self.down_groups {
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(size.0.div_ceil(8), size.1.div_ceil(8), 1);
        }
    }
}
