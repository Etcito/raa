//! Virtual shadow map (VSM), GPU side: the page writes (`vsm_copy.wgsl`: rendered window
//! pages and page tables into the pool layers of the shadow array), the page marking
//! (`vsm_mark.wgsl`: the request bit set from the prepass depth) and its read-back ring.
//!
//! The page writes run in the owned pass `shadow.csm#vsm-pages` (order 24: after the window
//! renders of `shadow.csm` and of the virtualized-geometry casters, before every shadow reader),
//! the marking in `shadow.csm#vsm-mark` (order 34: after the prepass and the virtualized-geometry
//! G-buffer). The read-back is asynchronous (`map_buffer_on_submit`): `shadow.csm` applies the
//! newest completed request set a few frames later; until then pixels use coarser resident
//! levels or the scene-wide coarse map.

use std::num::NonZeroU64;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};

use raa_core::GpuContext;
use raa_math::Mat4;

use crate::vsm::{MAX_LEVELS, PAGE, REQUEST_WORDS, TABLE_WORDS};

/// Bytes of one `VsmJob`.
const JOB_SIZE: u64 = 32;
/// Page writes per frame at most: the copies (budget <= 768) and the page tables.
pub(crate) const MAX_JOBS: usize = 768 + MAX_LEVELS as usize;
/// Bytes of `VsmMark`.
const MARK_SIZE: usize = 64 + 64 + 64 + 16 + 16 + 16;
/// Bytes of the request bit set.
const REQUEST_BYTES: u64 = REQUEST_WORDS as u64 * 4;
/// Read-back buffers in flight at most.
const READBACKS: usize = 3;

static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// A completed read-back: the frame it was marked in, the window origins it was marked with
/// and the request words.
pub(crate) type Requests = (u64, [[i32; 2]; MAX_LEVELS as usize], Vec<u32>);

/// One page write: a window page copy (kind 0) or a level's page table (kind 1).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct Job {
    pub src: (u32, u32),
    pub dst: (u32, u32),
    pub src_layer: u32,
    pub kind: u32,
    pub level: u32,
}

/// What the marking pass needs for one frame.
pub(crate) struct MarkInput {
    pub inv_view_proj: Mat4,
    pub light: Mat4,
    pub origins: [[i32; 2]; MAX_LEVELS as usize],
    pub view_z: [f32; 4],
    pub texel0: f32,
    pub levels: u32,
    pub pixel_angle: f32,
    /// Constant pixel footprint (m) of an orthographic camera (0 for a perspective one).
    pub footprint: f32,
    pub bias: f32,
    pub margin: f32,
}

pub(crate) struct VsmGpu {
    page_pipeline: wgpu::RenderPipeline,
    page_layout: wgpu::BindGroupLayout,
    jobs: wgpu::Buffer,
    tables: wgpu::Buffer,
    mark_pipeline: wgpu::ComputePipeline,
    mark_layout: wgpu::BindGroupLayout,
    mark_uniform: wgpu::Buffer,
    requests: wgpu::Buffer,
    readbacks: Vec<(wgpu::Buffer, Arc<AtomicBool>)>,
    send: Sender<Requests>,
    recv: Receiver<Requests>,
    /// The page bind group over the scratch layers of the current caster array.
    page_group: Option<wgpu::BindGroup>,
    /// Per pool layer: the job range drawn into it this frame.
    pub(crate) draws: Vec<(u32, Range<u32>)>,
    /// This frame marks (frame index and the origins of this frame's uniform).
    pub(crate) mark: Option<(u64, [[i32; 2]; MAX_LEVELS as usize])>,
    /// Unique per instance (the marking bind group of the pass is rebuilt on a new one).
    id: u64,
    job_bytes: Vec<u8>,
}

fn storage_entry(
    binding: u32,
    read_only: bool,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl VsmGpu {
    pub(crate) fn new(gpu: &GpuContext) -> VsmGpu {
        let device = gpu.device();
        let page_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa.shadow.vsm.pages"),
            source: wgpu::ShaderSource::Wgsl(include_str!("wgsl/vsm_copy.wgsl").into()),
        });
        let page_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa.shadow.vsm.pages"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                storage_entry(1, true, wgpu::ShaderStages::VERTEX_FRAGMENT),
                storage_entry(2, true, wgpu::ShaderStages::FRAGMENT),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raa.shadow.vsm.pages"),
            bind_group_layouts: &[Some(&page_layout)],
            immediate_size: 0,
        });
        let page_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("raa.shadow.vsm.pages"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &page_module,
                entry_point: Some("vs_page"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &page_module,
                entry_point: Some("fs_page"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[],
            }),
            multiview_mask: None,
            cache: None,
        });
        let mark_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa.shadow.vsm.mark"),
            source: wgpu::ShaderSource::Wgsl(include_str!("wgsl/vsm_mark.wgsl").into()),
        });
        let mark_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa.shadow.vsm.mark"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(MARK_SIZE as u64),
                    },
                    count: None,
                },
                storage_entry(2, false, wgpu::ShaderStages::COMPUTE),
            ],
        });
        let mark_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raa.shadow.vsm.mark"),
            bind_group_layouts: &[Some(&mark_layout)],
            immediate_size: 0,
        });
        let mark_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("raa.shadow.vsm.mark"),
            layout: Some(&mark_pipeline_layout),
            module: &mark_module,
            entry_point: Some("cs_mark"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let buffer = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let jobs = buffer(
            "raa.shadow.vsm.jobs",
            JOB_SIZE * MAX_JOBS as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let tables = buffer(
            "raa.shadow.vsm.tables",
            (TABLE_WORDS * MAX_LEVELS as usize * 4) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let mark_uniform = buffer(
            "raa.shadow.vsm.mark",
            MARK_SIZE as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let requests = buffer(
            "raa.shadow.vsm.requests",
            REQUEST_BYTES,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let readbacks = (0..READBACKS)
            .map(|_| {
                (
                    buffer(
                        "raa.shadow.vsm.readback",
                        REQUEST_BYTES,
                        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    ),
                    Arc::new(AtomicBool::new(false)),
                )
            })
            .collect();
        let (send, recv) = channel();
        VsmGpu {
            page_pipeline,
            page_layout,
            jobs,
            tables,
            mark_pipeline,
            mark_layout,
            mark_uniform,
            requests,
            readbacks,
            send,
            recv,
            page_group: None,
            draws: Vec::new(),
            mark: None,
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            job_bytes: Vec::with_capacity(MAX_JOBS * JOB_SIZE as usize),
        }
    }

    /// Unique id of this instance.
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// Bytes of the GPU buffers (diagnostics).
    pub(crate) fn allocated_bytes(&self) -> u64 {
        JOB_SIZE * MAX_JOBS as u64
            + (TABLE_WORDS * MAX_LEVELS as usize * 4) as u64
            + MARK_SIZE as u64
            + REQUEST_BYTES * (1 + READBACKS as u64)
    }

    /// Drops the page bind group (a new caster array).
    pub(crate) fn invalidate_page_group(&mut self) {
        self.page_group = None;
    }

    /// The newest completed read-back, if any arrived since the last call.
    pub(crate) fn take_requests(&mut self) -> Option<Requests> {
        let mut newest: Option<Requests> = None;
        while let Ok(r) = self.recv.try_recv() {
            if newest.as_ref().is_none_or(|n| r.0 > n.0) {
                newest = Some(r);
            }
        }
        newest
    }

    /// Uploads `jobs` (already sorted by pool layer, `layers[i]` the pool layer of job `i`)
    /// and records the per-layer draw ranges.
    pub(crate) fn write_jobs(&mut self, gpu: &GpuContext, jobs: &[(u32, Job)]) {
        self.draws.clear();
        self.job_bytes.clear();
        for (i, (layer, j)) in jobs.iter().take(MAX_JOBS).enumerate() {
            for w in [
                j.src.0,
                j.src.1,
                j.dst.0,
                j.dst.1,
                j.src_layer,
                j.kind,
                j.level,
                0,
            ] {
                self.job_bytes.extend_from_slice(&w.to_le_bytes());
            }
            let i = i as u32;
            match self.draws.last_mut() {
                Some((l, range)) if *l == *layer => range.end = i + 1,
                _ => self.draws.push((*layer, i..i + 1)),
            }
        }
        if !self.job_bytes.is_empty() {
            gpu.queue().write_buffer(&self.jobs, 0, &self.job_bytes);
        }
    }

    /// Uploads the page table of `level`.
    pub(crate) fn write_table(&self, gpu: &GpuContext, level: u32, table: &[u32]) {
        let mut bytes = Vec::with_capacity(table.len() * 4);
        for w in table {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        gpu.queue().write_buffer(
            &self.tables,
            (level as usize * TABLE_WORDS * 4) as u64,
            &bytes,
        );
    }

    /// Writes the marking uniform of this frame.
    pub(crate) fn write_mark(&self, gpu: &GpuContext, m: &MarkInput) {
        let mut b = [0u8; MARK_SIZE];
        m.inv_view_proj.write_le_bytes(&mut b[0..64]);
        m.light.write_le_bytes(&mut b[64..128]);
        let mut f = |offset: usize, v: f32| b[offset..offset + 4].copy_from_slice(&v.to_le_bytes());
        for (l, o) in m.origins.iter().enumerate() {
            f(128 + l * 8, o[0] as f32);
            f(128 + l * 8 + 4, o[1] as f32);
        }
        for (i, v) in m.view_z.iter().enumerate() {
            f(192 + i * 4, *v);
        }
        f(208, m.texel0);
        f(212, m.levels as f32);
        f(216, m.pixel_angle);
        f(220, m.bias);
        f(224, m.margin);
        f(228, m.footprint);
        gpu.queue().write_buffer(&self.mark_uniform, 0, &b);
    }

    /// Records this frame's page writes into the pool layers (`pool_view(i)` is the D2 view of
    /// pool layer `i`); `scratch` is the D2Array view over the 4 cascade layers.
    pub(crate) fn record_pages<'v>(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        scratch: &wgpu::TextureView,
        pool_view: impl Fn(u32) -> Option<&'v wgpu::TextureView>,
    ) {
        if self.draws.is_empty() {
            return;
        }
        if self.page_group.is_none() {
            self.page_group = Some(gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("raa.shadow.vsm.pages"),
                layout: &self.page_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(scratch),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.jobs.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.tables.as_entire_binding(),
                    },
                ],
            }));
        }
        let Some(group) = self.page_group.as_ref() else {
            return;
        };
        for (layer, range) in &self.draws {
            let Some(view) = pool_view(*layer) else {
                continue;
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("raa.shadow.vsm.pages"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.page_pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..6, range.clone());
        }
        self.draws.clear();
    }

    /// The marking bind group over `depth`.
    pub(crate) fn mark_group(
        &self,
        gpu: &GpuContext,
        depth: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa.shadow.vsm.mark"),
            layout: &self.mark_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.mark_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.requests.as_entire_binding(),
                },
            ],
        })
    }

    /// Records the marking of this frame (when `mark` is set) over a `size` depth, and its
    /// read-back when a read-back buffer is free.
    pub(crate) fn record_mark(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        group: &wgpu::BindGroup,
        size: (u32, u32),
    ) {
        let Some((frame, origins)) = self.mark.take() else {
            return;
        };
        let Some((readback, busy)) = self
            .readbacks
            .iter()
            .find(|(_, busy)| !busy.load(Ordering::Acquire))
            .cloned()
        else {
            return;
        };
        encoder.clear_buffer(&self.requests, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("raa.shadow.vsm.mark"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.mark_pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(size.0.div_ceil(8), size.1.div_ceil(8), 1);
        }
        encoder.copy_buffer_to_buffer(&self.requests, 0, &readback, 0, REQUEST_BYTES);
        busy.store(true, Ordering::Release);
        let send = self.send.clone();
        let mapped = readback.clone();
        encoder.map_buffer_on_submit(
            &readback,
            wgpu::MapMode::Read,
            0..REQUEST_BYTES,
            move |result| {
                if result.is_ok() {
                    let words =
                        mapped
                            .slice(0..REQUEST_BYTES)
                            .get_mapped_range()
                            .ok()
                            .map(|data| {
                                data.chunks_exact(4)
                                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                    .collect::<Vec<u32>>()
                            });
                    mapped.unmap();
                    if let Some(words) = words {
                        let _ = send.send((frame, origins, words));
                    }
                }
                busy.store(false, Ordering::Release);
            },
        );
    }
}

/// Texel origin (x, y) and pool layer index of pool slot `slot` with `pps` pages per layer side.
pub(crate) fn slot_texel(slot: u32, pps: u32) -> (u32, (u32, u32)) {
    let per_layer = pps * pps;
    let local = slot % per_layer;
    (
        slot / per_layer,
        ((local % pps) * PAGE, (local / pps) * PAGE),
    )
}
