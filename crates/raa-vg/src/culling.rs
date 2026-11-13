//! The S32 `PassView` of RAA's virtualized geometry and the view-slot table, plus the plane
//! extraction and builders shared by the cull, raster and shadow paths. Adapted from solstice
//! (MIT), Copyright (c) 2026 SamG-Coder: `crates/solstice-render/src/vg/culling.rs:11-199`,
//! revision 7e465b5. The reversed-Z near/far roles are swapped for RAA's standard-Z convention.

use raa_core::{GpuContext, wgpu};
use raa_math::{Mat4, Vec3, Vec4};

/// S32 `pass_view.wgsl`: the WGSL twin of [`PassView`], the cull state and its read view, the
/// instance-filter and instance-flag constants, and the scene twins (`VgInstance`, `VgNode`,
/// `VgCluster`, `PackedVertex`, `VisibleCluster`, `QueueEntry`) with their decode helpers. It
/// declares no binding and no entry point; the cull, raster and resolve builders concatenate
/// it after `raa_shader::DEPTH_WGSL`.
pub const PASS_VIEW_WGSL: &str = include_str!("wgsl/pass_view.wgsl");

const CULL_CONTROL_WGSL: &str = include_str!("wgsl/cull_control.wgsl");

const CULL_WGSL: &str = include_str!("wgsl/cull.wgsl");

/// The cull-control module: `raa_shader::DEPTH_WGSL` + [`PASS_VIEW_WGSL`] + `cull_control.wgsl`,
/// with entry points `advance`, `finish` and `begin_second_phase` at group 2.
pub fn cull_control_module_wgsl() -> String {
    [raa_shader::DEPTH_WGSL, PASS_VIEW_WGSL, CULL_CONTROL_WGSL].concat()
}

/// The hierarchy-traversal module: `raa_shader::DEPTH_WGSL` + [`PASS_VIEW_WGSL`] +
/// `cull.wgsl`, with entry points `seed` and `process` (group 2 scene bindings 0-4, group 3
/// work bindings 5-8).
pub fn cull_module_wgsl() -> String {
    [raa_shader::DEPTH_WGSL, PASS_VIEW_WGSL, CULL_WGSL].concat()
}

/// Bytes of one `PassView` record (zero padding included).
pub const PASS_VIEW_SIZE: usize = 432;
/// Number of view slots in [`ViewSlots`].
pub const VG_VIEW_SLOTS: u32 = 11;
/// The main view's first-phase slot.
pub const MAIN_FIRST: u32 = 0;
/// The main view's second-phase slot.
pub const MAIN_SECOND: u32 = 1;
/// The first shadow cascade's slot (cascades 0..3 at 2..5).
pub const CASCADE_BASE: u32 = 2;
/// The first GI slot (6..10, reserved and never written by P3).
pub const GI_BASE: u32 = 6;
/// The number of GI slots.
pub const GI_SLOTS: u32 = 5;
/// Bytes between view slots.
pub const SLOT_STRIDE: u32 = 512;

/// Twin of `PassView` in `pass_view.wgsl`. The 432-byte layout is the shared state.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PassView {
    pub raster_view_proj: Mat4,
    pub occlusion_view: Mat4,
    pub planes: [Vec4; 6],
    pub camera_position: Vec3,
    pub lod_threshold: f32,
    pub viewport: [f32; 2],
    pub proj_scale: [f32; 2],
    pub hzb_size: [f32; 2],
    pub near: f32,
    pub lod_scale: f32,
    pub is_ortho: u32,
    pub phase: u32,
    pub occlusion_enabled: u32,
    pub allow_software: u32,
    pub eye_position: Vec3,
    pub eye_lod_scale: f32,
    pub eye_near: f32,
    pub receiver_plane_count: u32,
    pub instance_filter: u32,
    pub stamp: u32,
    pub receiver_planes: [Vec4; 6],
    pub eye_light: Vec3,
}

impl PassView {
    /// The six planes of `view_proj`, in the order left, right, bottom, top, near, far.
    /// Standard-Z: near is row 2 and far is row 3 minus row 2. Each plane is normalised by
    /// its xyz length; a degenerate plane becomes `(0, 0, 0, 1)`. Inside is positive.
    pub fn frustum_planes(view_proj: Mat4) -> [Vec4; 6] {
        let row = |r: usize| {
            Vec4::new(
                view_proj.cols[0][r],
                view_proj.cols[1][r],
                view_proj.cols[2][r],
                view_proj.cols[3][r],
            )
        };
        let [x, y, z, w] = [row(0), row(1), row(2), row(3)];
        [w + x, w - x, w + y, w - y, z, w - z].map(|plane| {
            let length = plane.xyz().length();
            if length > 1.0e-6 {
                plane * (1.0 / length)
            } else {
                Vec4::new(0.0, 0.0, 0.0, 1.0)
            }
        })
    }

    /// A perspective view: no occlusion testing and no software rasterization, drawing all
    /// instances. `lod_scale` is pixels per metre at unit distance.
    pub fn perspective(
        raster_view_proj: Mat4,
        occlusion_view: Mat4,
        camera_position: Vec3,
        viewport: [f32; 2],
        proj_scale: [f32; 2],
        near: f32,
        lod_threshold: f32,
    ) -> PassView {
        PassView {
            raster_view_proj,
            occlusion_view,
            planes: Self::frustum_planes(raster_view_proj),
            camera_position,
            lod_threshold,
            viewport,
            proj_scale,
            hzb_size: [1.0, 1.0],
            near,
            lod_scale: 0.5 * viewport[1] * proj_scale[1],
            is_ortho: 0,
            phase: 0,
            occlusion_enabled: 0,
            allow_software: 0,
            eye_position: Vec3::ZERO,
            eye_lod_scale: 0.0,
            eye_near: 0.0,
            receiver_plane_count: 0,
            instance_filter: InstanceFilter::All as u32,
            stamp: 0,
            receiver_planes: [Vec4::new(0.0, 0.0, 0.0, 0.0); 6],
            eye_light: Vec3::Y,
        }
    }

    /// An orthographic view for volumes such as shadow cascades. `pixels_per_unit` converts
    /// LOD error to pixels.
    pub fn orthographic(
        view_proj: Mat4,
        center: Vec3,
        viewport: [f32; 2],
        pixels_per_unit: f32,
        lod_threshold: f32,
    ) -> PassView {
        PassView {
            raster_view_proj: view_proj,
            occlusion_view: Mat4::IDENTITY,
            planes: Self::frustum_planes(view_proj),
            camera_position: center,
            lod_threshold,
            viewport,
            proj_scale: [1.0, 1.0],
            hzb_size: [1.0, 1.0],
            near: 0.0,
            lod_scale: pixels_per_unit,
            is_ortho: 1,
            phase: 0,
            occlusion_enabled: 0,
            allow_software: 0,
            eye_position: Vec3::ZERO,
            eye_lod_scale: 0.0,
            eye_near: 0.0,
            receiver_plane_count: 0,
            instance_filter: InstanceFilter::All as u32,
            stamp: 0,
            receiver_planes: [Vec4::new(0.0, 0.0, 0.0, 0.0); 6],
            eye_light: Vec3::Y,
        }
    }

    /// Keeps only the planes facing the light (`light` points at the light).
    pub fn shadowing(mut self, planes: &[Vec4; 6], light: Vec3) -> PassView {
        self.receiver_planes = [Vec4::new(0.0, 0.0, 0.0, 0.0); 6];
        let mut count = 0;
        for plane in planes {
            if plane.xyz().dot(light) >= 0.0 {
                self.receiver_planes[count] = *plane;
                count += 1;
            }
        }
        self.receiver_plane_count = count as u32;
        self
    }

    /// Bounds a light view's LOD by what a camera at `eye` sees of the shadows it draws.
    pub fn seen_from(mut self, eye: Vec3, light: Vec3, lod_scale: f32, nearest: f32) -> PassView {
        self.eye_position = eye;
        let unit = light.normalize_or_zero();
        self.eye_light = if unit == Vec3::ZERO { Vec3::Y } else { unit };
        self.eye_lod_scale = lod_scale;
        self.eye_near = nearest.max(1.0e-3);
        self
    }

    /// The same view, drawing other instances.
    pub fn drawing(mut self, filter: InstanceFilter) -> PassView {
        self.instance_filter = filter as u32;
        self
    }

    /// Writes the S32 layout little-endian. A shorter `out` gets the prefix that fits.
    pub fn write_le_bytes(&self, out: &mut [u8]) {
        put_mat4(out, 0, &self.raster_view_proj);
        put_mat4(out, 64, &self.occlusion_view);
        let mut at = 128;
        for value in self.planes.iter().flat_map(|p| p.to_array()) {
            put_f32(out, at, value);
            at += 4;
        }
        for value in self.camera_position.to_array() {
            put_f32(out, at, value);
            at += 4;
        }
        put_f32(out, at, self.lod_threshold);
        at += 4;
        for value in self
            .viewport
            .into_iter()
            .chain(self.proj_scale)
            .chain(self.hzb_size)
        {
            put_f32(out, at, value);
            at += 4;
        }
        put_f32(out, at, self.near);
        put_f32(out, at + 4, self.lod_scale);
        put_u32(out, at + 8, self.is_ortho);
        put_u32(out, at + 12, self.phase);
        put_u32(out, at + 16, self.occlusion_enabled);
        put_u32(out, at + 20, self.allow_software);
        at += 24;
        for value in self.eye_position.to_array() {
            put_f32(out, at, value);
            at += 4;
        }
        put_f32(out, at, self.eye_lod_scale);
        put_f32(out, at + 4, self.eye_near);
        put_u32(out, at + 8, self.receiver_plane_count);
        put_u32(out, at + 12, self.instance_filter);
        put_u32(out, at + 16, self.stamp);
        at += 20;
        for value in self.receiver_planes.iter().flat_map(|p| p.to_array()) {
            put_f32(out, at, value);
            at += 4;
        }
        for value in self.eye_light.to_array() {
            put_f32(out, at, value);
            at += 4;
        }
        put_u32(out, at, 0);
    }
}

fn put_mat4(out: &mut [u8], at: usize, matrix: &Mat4) {
    for (i, value) in matrix.cols.iter().flatten().enumerate() {
        put_f32(out, at + i * 4, *value);
    }
}

fn put_f32(out: &mut [u8], at: usize, value: f32) {
    if let Some(chunk) = out.get_mut(at..at + 4) {
        chunk.copy_from_slice(&value.to_le_bytes());
    }
}

fn put_u32(out: &mut [u8], at: usize, value: u32) {
    if let Some(chunk) = out.get_mut(at..at + 4) {
        chunk.copy_from_slice(&value.to_le_bytes());
    }
}

/// Which instances a view draws. Twin of the `INSTANCES_` constants in `pass_view.wgsl`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum InstanceFilter {
    All = 0,
    Static = 1,
    Movable = 2,
    Gi = 3,
}

const _: () = assert!(PASS_VIEW_SIZE <= SLOT_STRIDE as usize);
const _: () = assert!(std::mem::size_of::<PassView>() <= PASS_VIEW_SIZE);

/// A uniform buffer holding [`VG_VIEW_SLOTS`] [`PassView`]s, selected by dynamic offset.
pub struct ViewSlots {
    buffer: wgpu::Buffer,
}

impl ViewSlots {
    /// One UNIFORM | COPY_DST buffer of `VG_VIEW_SLOTS * SLOT_STRIDE` bytes.
    pub fn new(gpu: &GpuContext) -> ViewSlots {
        let buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa-vg view slots"),
            size: u64::from(VG_VIEW_SLOTS * SLOT_STRIDE),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ViewSlots { buffer }
    }

    /// Writes `view` into `slot`; an out-of-range slot writes nothing.
    pub fn write(&self, gpu: &GpuContext, slot: u32, view: &PassView) {
        if slot >= VG_VIEW_SLOTS {
            return;
        }
        let mut bytes = [0u8; PASS_VIEW_SIZE];
        view.write_le_bytes(&mut bytes);
        gpu.queue()
            .write_buffer(&self.buffer, u64::from(Self::offset(slot)), &bytes);
    }

    pub fn offset(slot: u32) -> u32 {
        slot * SLOT_STRIDE
    }

    /// The binding at offset 0, sized to one `PassView`; pair it with `offset(slot)`.
    pub fn binding(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: 0,
            size: wgpu::BufferSize::new(PASS_VIEW_SIZE as u64),
        })
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn allocated_bytes(&self) -> u64 {
        self.buffer.size()
    }
}

/// Bytes of one `CullState` record (`CullState` in `pass_view.wgsl`).
const STATE_SIZE: u64 = 48;
/// Bytes of one indirect-arguments record (`Args` in `cull_control.wgsl`).
const ARGS_SIZE: u64 = 56;
/// Offset of the software-rasterization dispatch arguments.
const ARGS_SOFTWARE_OFFSET: u64 = 12;
/// Offset of the hardware-rasterization draw arguments.
const ARGS_DRAW_OFFSET: u64 = 24;
/// Offset of the software-depth merge draw arguments.
const ARGS_MERGE_OFFSET: u64 = 40;

/// Which occlusion phase a cull runs.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CullPhase {
    /// Tests the previous frame's depth and collects retries. Views without occlusion run
    /// this phase alone.
    First,
    /// Re-tests the retries against this frame's depth.
    Second,
}

/// One buffer with `size` bytes and `usage`, labelled with `label`.
fn buffer(gpu: &GpuContext, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

/// One compute-stage bind group layout entry.
fn entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}

/// A read-only or read-write storage buffer binding.
fn storage(read_only: bool) -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    }
}

/// A bind group entry naming the whole of `buffer`.
fn bind_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

/// A labelled compute pipeline over `layouts` for the module's `entry` point.
fn compute(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    layouts: &[Option<&wgpu::BindGroupLayout>],
    entry: &str,
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(entry),
        bind_group_layouts: layouts,
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: Some(&pipeline_layout),
        module,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

/// The compute pipelines and bind group layouts of culling, shared by every view. The
/// placeholder buffer and pyramid stand in for a scene that does not exist yet.
#[allow(dead_code)]
pub struct CullPipelines {
    pub(crate) scene_layout: wgpu::BindGroupLayout,
    pub(crate) work_layout: wgpu::BindGroupLayout,
    pub(crate) control_layout: wgpu::BindGroupLayout,
    pub(crate) seed: wgpu::ComputePipeline,
    pub(crate) process: wgpu::ComputePipeline,
    pub(crate) advance: wgpu::ComputePipeline,
    pub(crate) finish: wgpu::ComputePipeline,
    pub(crate) begin_second_phase: wgpu::ComputePipeline,
    pub(crate) placeholder: wgpu::Buffer,
    pub(crate) placeholder_pyramid: wgpu::TextureView,
}

impl CullPipelines {
    /// Builds the three layouts, the five compute pipelines and the 1x1 placeholders.
    pub fn new(gpu: &GpuContext) -> CullPipelines {
        let device = gpu.device();
        let scene_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa-vg cull scene"),
            entries: &[
                entry(
                    0,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(PASS_VIEW_SIZE as u64),
                    },
                ),
                entry(1, storage(true)),
                entry(2, storage(true)),
                entry(3, storage(true)),
                entry(
                    4,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
            ],
        });
        let work_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa-vg cull work"),
            entries: &[
                entry(5, storage(false)),
                entry(6, storage(false)),
                entry(7, storage(false)),
                entry(8, storage(false)),
            ],
        });
        let control_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa-vg cull control"),
            entries: &[entry(0, storage(false)), entry(1, storage(false))],
        });

        let cull = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa-vg cull module"),
            source: wgpu::ShaderSource::Wgsl(cull_module_wgsl().into()),
        });
        let control = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa-vg cull control module"),
            source: wgpu::ShaderSource::Wgsl(cull_control_module_wgsl().into()),
        });
        let cull_layouts = [None, None, Some(&scene_layout), Some(&work_layout)];
        let control_layouts = [None, None, Some(&control_layout)];

        let placeholder = buffer(
            gpu,
            "raa-vg cull placeholder instances",
            16,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let pyramid = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("raa-vg cull placeholder pyramid"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        gpu.queue().write_texture(
            pyramid.as_image_copy(),
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
        let placeholder_pyramid = pyramid.create_view(&wgpu::TextureViewDescriptor::default());

        CullPipelines {
            seed: compute(device, &cull, &cull_layouts, "seed"),
            process: compute(device, &cull, &cull_layouts, "process"),
            advance: compute(device, &control, &control_layouts, "advance"),
            finish: compute(device, &control, &control_layouts, "finish"),
            begin_second_phase: compute(device, &control, &control_layouts, "begin_second_phase"),
            scene_layout,
            work_layout,
            control_layout,
            placeholder,
            placeholder_pyramid,
        }
    }

    /// The GPU memory culling owns while no scene is bound: the placeholder buffer and texel.
    pub fn allocated_bytes(&self) -> u64 {
        self.placeholder.size() + 4
    }
}

/// The transient buffers of one culling run, with one bind group per occlusion phase.
#[allow(dead_code)]
pub struct CullWork {
    state: wgpu::Buffer,
    args: wgpu::Buffer,
    visible: wgpu::Buffer,
    queue_capacity: u64,
    visible_capacity: u64,
    first_phase: wgpu::BindGroup,
    second_phase: wgpu::BindGroup,
    control: wgpu::BindGroup,
}

impl CullWork {
    /// One state buffer, two queues of `queue_capacity * 8` bytes, one visible list of
    /// `visible_capacity * 8` bytes and the 56-byte arguments buffer. The first-phase bind
    /// group reads queue A and retries into B; the second phase swaps them.
    pub fn new(
        gpu: &GpuContext,
        pipelines: &CullPipelines,
        label: &str,
        queue_capacity: u64,
        visible_capacity: u64,
    ) -> CullWork {
        let state = buffer(
            gpu,
            &format!("{label} cull state"),
            STATE_SIZE,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        let queue_a = buffer(
            gpu,
            &format!("{label} cull queue a"),
            queue_capacity * 8,
            wgpu::BufferUsages::STORAGE,
        );
        let queue_b = buffer(
            gpu,
            &format!("{label} cull queue b"),
            queue_capacity * 8,
            wgpu::BufferUsages::STORAGE,
        );
        let visible = buffer(
            gpu,
            &format!("{label} cull visible"),
            visible_capacity * 8,
            wgpu::BufferUsages::STORAGE,
        );
        let args = buffer(
            gpu,
            &format!("{label} cull args"),
            ARGS_SIZE,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC,
        );
        let bind = |queue: &wgpu::Buffer, retry: &wgpu::Buffer| {
            gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &pipelines.work_layout,
                entries: &[
                    bind_entry(5, &state),
                    bind_entry(6, queue),
                    bind_entry(7, retry),
                    bind_entry(8, &visible),
                ],
            })
        };
        let first_phase = bind(&queue_a, &queue_b);
        let second_phase = bind(&queue_b, &queue_a);
        let control = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &pipelines.control_layout,
            entries: &[bind_entry(0, &state), bind_entry(1, &args)],
        });
        CullWork {
            state,
            args,
            visible,
            queue_capacity,
            visible_capacity,
            first_phase,
            second_phase,
            control,
        }
    }

    pub fn state(&self) -> &wgpu::Buffer {
        &self.state
    }

    pub fn args(&self) -> &wgpu::Buffer {
        &self.args
    }

    pub fn visible(&self) -> &wgpu::Buffer {
        &self.visible
    }

    pub fn visible_capacity(&self) -> u64 {
        self.visible_capacity
    }

    pub const fn software_args_offset() -> u64 {
        ARGS_SOFTWARE_OFFSET
    }

    pub const fn draw_args_offset() -> u64 {
        ARGS_DRAW_OFFSET
    }

    pub const fn merge_args_offset() -> u64 {
        ARGS_MERGE_OFFSET
    }

    /// The GPU memory this run owns.
    pub fn allocated_bytes(&self) -> u64 {
        self.state.size() + self.args.size() + self.visible.size() + 2 * self.queue_capacity * 8
    }

    #[allow(dead_code)]
    pub(crate) fn work_group(&self, phase: CullPhase) -> &wgpu::BindGroup {
        match phase {
            CullPhase::First => &self.first_phase,
            CullPhase::Second => &self.second_phase,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn control_group(&self) -> &wgpu::BindGroup {
        &self.control
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raa_core::testing::{block_on, skip, software_gpu};

    #[test]
    fn args_offsets_and_size() {
        assert_eq!(CullWork::software_args_offset(), 12);
        assert_eq!(CullWork::draw_args_offset(), 24);
        assert_eq!(CullWork::merge_args_offset(), 40);
        assert_eq!(ARGS_SIZE, 56);
    }

    #[test]
    fn cull_pipelines_and_work_create_without_validation_error() {
        let test = "cull_pipelines_and_work_create_without_validation_error";
        let Some(gpu) = software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
        let pipelines = CullPipelines::new(&gpu);
        let work = CullWork::new(&gpu, &pipelines, "cull test", 1024, 1024);
        let error = block_on(errors.pop());
        assert!(error.is_none(), "cull pipeline validation: {error:?}");
        assert_eq!(pipelines.allocated_bytes(), 20);
        assert_eq!(work.allocated_bytes(), 48 + 2 * 8192 + 8192 + 56);
        assert_eq!(work.visible_capacity(), 1024);
        assert!(!std::ptr::eq(
            work.work_group(CullPhase::First),
            work.work_group(CullPhase::Second)
        ));
        let _ = work.control_group();
    }
}
