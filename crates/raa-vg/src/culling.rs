//! The S32 `PassView` of RAA's virtualized geometry and the view-slot table, plus the plane
//! extraction and builders shared by the cull, raster and shadow paths. Adapted from solstice
//! (MIT), Copyright (c) 2026 SamG-Coder: `crates/solstice-render/src/vg/culling.rs:11-199`,
//! revision 7e465b5. The reversed-Z near/far roles are swapped for RAA's standard-Z convention.

use raa_core::{GpuContext, wgpu};
use raa_math::{Mat4, Vec3, Vec4};

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
