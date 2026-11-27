//! Cached cube shadow maps of shadow-casting local lights (`shadow.local`, order 36).
//!
//! A point or spot light with `shadows` gets a stable slot from raa-lit (`LocalShadowSlot`,
//! published through `LightingResources::local_shadow_slots`; its GPU record carries `slot + 1`).
//! This pass keeps an R32Float array of 6 faces per slot and fills it by tracing the shared
//! world-space mesh BVH (the one DDGI traces) from each light: texel = distance to the first
//! occluder, rays starting `near` (the light's emitter radius) from the light so its own housing
//! casts nothing. The forward samples it with 2x2 PCF (raa-shader `raa_local_shadow`).
//!
//! Static caching: a face is traced once, then only again when its light moves (position, range
//! or radius), the BVH changes inside the light's sphere (the geometry's dirty bounds; a full
//! rebuild re-traces everything) or the array is reallocated. At most `faces_per_frame` faces are
//! traced per frame, nearest lights first, so a scene load or an edit never costs one long frame;
//! until a face is traced its texels are 0 and read as unoccluded. Without trace geometry (no
//! BVH-consuming GI in the graph) nothing is traced and every light stays unshadowed.

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::{GpuContext, RaaError};
use raa_pipeline::{
    FeaturePass, LightingResources, LocalShadowSlot, MAX_LOCAL_SHADOWS, Pipeline, PrepareCtx,
    RecordCtx,
};
use wgpu::{Buffer, BufferUsages as Usage};

use super::MeshTraceHandle;
use super::pass::{buffer, buffer_entry};

pub const LOCAL_SHADOW_NAME: &str = "shadow.local";
const ORDER: u32 = 36;
const FACES: u32 = 6;
/// Faces one dispatch can bake (the params uniform holds this many jobs).
pub const LOCAL_SHADOW_MAX_JOBS: u32 = 24;
/// Slots are allocated in groups of this many, so a few added lights do not reallocate.
const SLOT_GROUP: u32 = 4;
const PARAMS_BYTES: u64 = 16 + 32 * LOCAL_SHADOW_MAX_JOBS as u64;
const OPTIONS_BYTES: u64 = 64;
const WORKGROUP: u32 = 8;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalShadowSettings {
    pub enabled: bool,
    /// Texels per cube-face side (multiple of 8, 64..=1024).
    pub resolution: u32,
    /// Faces traced per frame at most (1..=24).
    pub faces_per_frame: u32,
}

impl Default for LocalShadowSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            resolution: 128,
            faces_per_frame: 2,
        }
    }
}

impl LocalShadowSettings {
    pub fn validate(self) -> Result<Self, RaaError> {
        let invalid = |message: String| RaaError::FrameGraph {
            code: "shadow.local_invalid",
            message,
        };
        if !(64..=1024).contains(&self.resolution) || !self.resolution.is_multiple_of(WORKGROUP) {
            return Err(invalid(format!(
                "resolution must be a multiple of 8 in 64..=1024 (got {})",
                self.resolution
            )));
        }
        if !(1..=LOCAL_SHADOW_MAX_JOBS).contains(&self.faces_per_frame) {
            return Err(invalid(format!(
                "facesPerFrame must be in 1..={LOCAL_SHADOW_MAX_JOBS} (got {})",
                self.faces_per_frame
            )));
        }
        Ok(self)
    }
}

/// What the last prepare saw (diagnostics, `localShadowStats` in JS).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LocalShadowStats {
    /// Shadow-casting lights holding a slot.
    pub lights: u32,
    /// Slots the array holds (a multiple of 4).
    pub capacity: u32,
    pub resolution: u32,
    /// Faces of the current lights traced and still valid.
    pub ready_faces: u32,
    /// Faces still to trace.
    pub pending_faces: u32,
    /// Faces traced this frame.
    pub traced_faces: u32,
    /// Faces traced since the pass was installed.
    pub traced_total: u64,
    /// Triangles of the traced BVH (0: no trace geometry, nothing is traced).
    pub triangles: u32,
    /// Bytes of the face array.
    pub bytes: u64,
}

#[derive(Default)]
pub struct LocalShadowState {
    pub settings: LocalShadowSettings,
    eye: Option<[f32; 3]>,
    stats: LocalShadowStats,
    /// Forces every face to be traced again at the next prepare.
    reset: bool,
}

impl LocalShadowState {
    pub fn configure(&mut self, settings: LocalShadowSettings) -> Result<(), RaaError> {
        self.settings = settings.validate()?;
        Ok(())
    }

    pub fn stats(&self) -> LocalShadowStats {
        self.stats
    }

    /// The camera position of the frame: the nearest lights are traced first.
    pub fn set_eye(&mut self, eye: [f32; 3]) {
        self.eye = Some(eye);
    }

    /// Re-traces every face (diagnostics and A/B captures).
    pub fn invalidate(&mut self) {
        self.reset = true;
    }
}

pub type LocalShadowHandle = Rc<RefCell<LocalShadowState>>;

pub struct LocalShadowFeature;

impl LocalShadowFeature {
    pub fn install(
        pipeline: &mut Pipeline,
        mesh: MeshTraceHandle,
    ) -> Result<LocalShadowHandle, RaaError> {
        let state = LocalShadowHandle::default();
        pipeline.add(Box::new(LocalShadowPass {
            state: state.clone(),
            mesh,
            array: None,
            cache: [SlotCache::default(); MAX_LOCAL_SHADOWS],
            geometry_revision: None,
            kernel: None,
            jobs: Vec::new(),
            last_frame: None,
            published: false,
        }))?;
        Ok(state)
    }
}

/// What a slot's faces were traced for: the light (position, range, near as bits) and the
/// faces done (bit f = face f).
#[derive(Clone, Copy, Default, PartialEq)]
struct SlotCache {
    key: Option<[u32; 5]>,
    done: u8,
}

const ALL_FACES: u8 = 0b11_1111;

fn slot_key(slot: &LocalShadowSlot) -> [u32; 5] {
    [
        slot.position[0].to_bits(),
        slot.position[1].to_bits(),
        slot.position[2].to_bits(),
        slot.range.to_bits(),
        slot.near.to_bits(),
    ]
}

struct FaceArray {
    texture: wgpu::Texture,
    sampled: wgpu::TextureView,
    storage: wgpu::TextureView,
    capacity: u32,
    resolution: u32,
}

struct Kernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    params: Buffer,
    options: Buffer,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Job {
    slot: u32,
    face: u32,
    position: [f32; 3],
    near: f32,
    range: f32,
}

struct LocalShadowPass {
    state: LocalShadowHandle,
    mesh: MeshTraceHandle,
    array: Option<FaceArray>,
    cache: [SlotCache; MAX_LOCAL_SHADOWS],
    geometry_revision: Option<u64>,
    kernel: Option<Kernel>,
    jobs: Vec<Job>,
    last_frame: Option<u64>,
    published: bool,
}

/// Whether a light's sphere overlaps an axis-aligned box.
fn sphere_overlaps(centre: [f32; 3], radius: f32, min: [f32; 3], max: [f32; 3]) -> bool {
    let mut d2 = 0.0;
    for axis in 0..3 {
        let c = centre[axis].clamp(min[axis], max[axis]);
        d2 += (centre[axis] - c).powi(2);
    }
    d2 <= radius * radius
}

/// The next faces to trace: slots nearest the eye first, at most `budget` faces.
fn pick_jobs(
    slots: &[LocalShadowSlot],
    cache: &[SlotCache; MAX_LOCAL_SHADOWS],
    eye: Option<[f32; 3]>,
    budget: u32,
) -> Vec<Job> {
    let mut order: Vec<&LocalShadowSlot> = slots
        .iter()
        .filter(|s| {
            (s.slot as usize) < MAX_LOCAL_SHADOWS && cache[s.slot as usize].done != ALL_FACES
        })
        .collect();
    let distance = |s: &LocalShadowSlot| {
        eye.map_or(0.0, |e| {
            (0..3).map(|i| (s.position[i] - e[i]).powi(2)).sum::<f32>()
        })
    };
    order.sort_by(|a, b| {
        distance(a)
            .total_cmp(&distance(b))
            .then(a.slot.cmp(&b.slot))
    });
    let mut jobs = Vec::new();
    for slot in order {
        let done = cache[slot.slot as usize].done;
        for face in 0..FACES {
            if jobs.len() as u32 >= budget {
                return jobs;
            }
            if done & (1 << face) == 0 {
                jobs.push(Job {
                    slot: slot.slot,
                    face,
                    position: slot.position,
                    near: slot.near,
                    range: slot.range,
                });
            }
        }
    }
    jobs
}

impl LocalShadowPass {
    fn unpublish(&mut self, lighting: &mut LightingResources) {
        if self.published {
            lighting.clear_local_shadow_map();
            self.published = false;
        }
    }

    fn kernel(&mut self, device: &wgpu::Device) -> &Kernel {
        self.kernel.get_or_insert_with(|| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("shadow.local.bake"),
                source: wgpu::ShaderSource::Wgsl(local_shadow_module().into()),
            });
            let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty,
                count: None,
            };
            let buf = |ty| wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            };
            let read = buf(wgpu::BufferBindingType::Storage { read_only: true });
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shadow.local.bake"),
                entries: &[
                    entry(0, buf(wgpu::BufferBindingType::Uniform)),
                    entry(1, buf(wgpu::BufferBindingType::Uniform)),
                    entry(2, read),
                    entry(3, read),
                    entry(
                        5,
                        wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                        },
                    ),
                ],
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow.local.bake"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("shadow.local.bake"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("bake_faces"),
                compilation_options: Default::default(),
                cache: None,
            });
            Kernel {
                pipeline,
                layout,
                params: buffer(
                    device,
                    "shadow.local.params",
                    PARAMS_BYTES,
                    Usage::UNIFORM | Usage::COPY_DST,
                ),
                options: buffer(
                    device,
                    "shadow.local.options",
                    OPTIONS_BYTES,
                    Usage::UNIFORM | Usage::COPY_DST,
                ),
            }
        })
    }

    fn ensure_array(&mut self, gpu: &GpuContext, needed: u32, resolution: u32) {
        if self
            .array
            .as_ref()
            .is_some_and(|a| a.capacity >= needed && a.resolution == resolution)
        {
            return;
        }
        let capacity = needed.div_ceil(SLOT_GROUP).max(1) * SLOT_GROUP;
        let capacity = capacity.min(MAX_LOCAL_SHADOWS as u32);
        // A new array is zero: every face reads as not traced (unoccluded) until baked.
        let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow.local.faces"),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: capacity * FACES,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let view = |label| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(label),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })
        };
        let sampled = view("shadow.local.faces.sampled");
        let storage = view("shadow.local.faces.storage");
        if let Some(old) = self.array.take() {
            old.texture.destroy();
        }
        self.array = Some(FaceArray {
            texture,
            sampled,
            storage,
            capacity,
            resolution,
        });
        for cache in &mut self.cache {
            cache.done = 0;
        }
    }

    fn release(&mut self, lighting: &mut LightingResources) {
        self.unpublish(lighting);
        if let Some(old) = self.array.take() {
            old.texture.destroy();
        }
        self.cache = [SlotCache::default(); MAX_LOCAL_SHADOWS];
        self.jobs.clear();
    }
}

impl FeaturePass for LocalShadowPass {
    fn name(&self) -> &str {
        LOCAL_SHADOW_NAME
    }

    fn order(&self) -> u32 {
        ORDER
    }

    fn set_enabled(&mut self, enabled: bool, lighting: &mut LightingResources) {
        if !enabled {
            self.release(lighting);
        }
    }

    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        // Several viewports prepare the pass once each: trace in the first only.
        let first = self.last_frame != Some(ctx.frame);
        self.last_frame = Some(ctx.frame);
        if first {
            self.jobs.clear();
        }
        let (settings, eye, reset) = {
            let mut state = self.state.borrow_mut();
            (state.settings, state.eye, std::mem::take(&mut state.reset))
        };
        let slots: Vec<LocalShadowSlot> = ctx.lighting.local_shadow_slots().to_vec();
        if !settings.enabled || slots.is_empty() {
            self.release(ctx.lighting);
            self.state.borrow_mut().stats = LocalShadowStats::default();
            return Ok(());
        }
        let (triangles, revision, dirty) = {
            let mesh = self.mesh.borrow();
            let revision = mesh.geometry.revision;
            let dirty: Vec<([f32; 3], [f32; 3])> = if self
                .geometry_revision
                .is_some_and(|old| old.wrapping_add(1) == revision)
            {
                mesh.geometry
                    .dirty_bounds
                    .iter()
                    .map(|b| (b.min, b.max))
                    .collect()
            } else {
                Vec::new()
            };
            (mesh.geometry.count(), revision, dirty)
        };
        let needed = slots.iter().map(|s| s.slot + 1).max().unwrap_or(0);
        self.ensure_array(ctx.gpu, needed, settings.resolution);
        // Free the slots no light holds any more; re-trace moved lights.
        let mut held = [false; MAX_LOCAL_SHADOWS];
        for slot in &slots {
            let Some(cache) = self.cache.get_mut(slot.slot as usize) else {
                continue;
            };
            held[slot.slot as usize] = true;
            let key = slot_key(slot);
            if cache.key != Some(key) || reset {
                *cache = SlotCache {
                    key: Some(key),
                    done: 0,
                };
            }
        }
        for (cache, held) in self.cache.iter_mut().zip(held) {
            if !held {
                *cache = SlotCache::default();
            }
        }
        // Geometry edits: the lights whose sphere meets a dirty box, or all after a rebuild.
        if let Some(old) = self.geometry_revision.replace(revision)
            && old != revision
        {
            let delta = old.wrapping_add(1) == revision && !dirty.is_empty();
            for slot in &slots {
                let Some(cache) = self.cache.get_mut(slot.slot as usize) else {
                    continue;
                };
                if !delta
                    || dirty
                        .iter()
                        .any(|(min, max)| sphere_overlaps(slot.position, slot.range, *min, *max))
                {
                    cache.done = 0;
                }
            }
        }
        if first && triangles > 0 {
            self.jobs = pick_jobs(&slots, &self.cache, eye, settings.faces_per_frame);
        }
        let array = self.array.as_ref().expect("face array");
        ctx.lighting.set_local_shadow_map(array.sampled.clone());
        self.published = true;
        let ready_faces: u32 = slots
            .iter()
            .filter_map(|s| self.cache.get(s.slot as usize))
            .map(|c| c.done.count_ones())
            .sum();
        let mut state = self.state.borrow_mut();
        let traced_total = state.stats.traced_total;
        state.stats = LocalShadowStats {
            lights: slots.len() as u32,
            capacity: array.capacity,
            resolution: array.resolution,
            ready_faces,
            pending_faces: FACES * slots.len() as u32 - ready_faces,
            traced_faces: self.jobs.len() as u32,
            traced_total: traced_total + self.jobs.len() as u64,
            triangles,
            bytes: u64::from(array.resolution).pow(2) * 4 * u64::from(array.capacity * FACES),
        };
        Ok(())
    }

    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        if self.jobs.is_empty() || self.array.is_none() {
            return Ok(());
        }
        let jobs = std::mem::take(&mut self.jobs);
        let mut mesh = self.mesh.borrow_mut();
        let triangles = mesh.geometry.count();
        if triangles == 0 {
            return Ok(());
        }
        let geometry = mesh.geometry_buffers(ctx.gpu);
        drop(mesh);
        let device = ctx.gpu.device();
        self.kernel(device);
        let kernel = self.kernel.as_ref().expect("kernel");
        let array = self.array.as_ref().expect("face array");
        let mut params = Vec::with_capacity(PARAMS_BYTES as usize);
        for word in [array.resolution, jobs.len() as u32, 0, 0] {
            params.extend_from_slice(&word.to_le_bytes());
        }
        for job in &jobs {
            for value in [job.position[0], job.position[1], job.position[2], job.near] {
                params.extend_from_slice(&value.to_le_bytes());
            }
            for word in [
                job.slot * FACES + job.face,
                job.face,
                job.range.to_bits(),
                0,
            ] {
                params.extend_from_slice(&word.to_le_bytes());
            }
        }
        params.resize(PARAMS_BYTES as usize, 0);
        let mut options = vec![0u8; OPTIONS_BYTES as usize];
        options[..4].copy_from_slice(&triangles.to_le_bytes());
        let queue = ctx.gpu.queue();
        queue.write_buffer(&kernel.params, 0, &params);
        queue.write_buffer(&kernel.options, 0, &options);
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow.local.bake"),
            layout: &kernel.layout,
            entries: &[
                buffer_entry(0, &kernel.params),
                buffer_entry(1, &kernel.options),
                buffer_entry(2, &geometry[0]),
                buffer_entry(3, &geometry[1]),
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&array.storage),
                },
            ],
        });
        {
            let mut pass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("shadow.local.bake"),
                    ..Default::default()
                });
            pass.set_pipeline(&kernel.pipeline);
            pass.set_bind_group(0, &group, &[]);
            let groups = array.resolution / WORKGROUP;
            pass.dispatch_workgroups(groups, groups, jobs.len() as u32);
        }
        for job in &jobs {
            if let Some(cache) = self.cache.get_mut(job.slot as usize) {
                cache.done |= 1 << job.face;
            }
        }
        Ok(())
    }
}

pub(crate) fn local_shadow_module() -> String {
    format!(
        "const LOCAL_SHADOW_MAX_JOBS: u32 = {LOCAL_SHADOW_MAX_JOBS}u;\n{}\n{}",
        include_str!("../wgsl/ray_interval.wgsl"),
        include_str!("../wgsl/mesh_local_shadow.wgsl"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot(slot: u32, x: f32) -> LocalShadowSlot {
        LocalShadowSlot {
            slot,
            index: slot,
            position: [x, 0.0, 0.0],
            range: 8.0,
            near: 0.3,
        }
    }

    #[test]
    fn local_shadow_module_validates() {
        raa_shader::Composer::validate(&local_shadow_module()).expect("local shadow WGSL");
    }

    #[test]
    fn settings_validate_resolution_and_budget() {
        assert!(LocalShadowSettings::default().validate().is_ok());
        for bad in [
            LocalShadowSettings {
                resolution: 60,
                ..Default::default()
            },
            LocalShadowSettings {
                resolution: 2048,
                ..Default::default()
            },
            LocalShadowSettings {
                faces_per_frame: 0,
                ..Default::default()
            },
            LocalShadowSettings {
                faces_per_frame: 25,
                ..Default::default()
            },
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
    }

    #[test]
    fn jobs_take_the_nearest_lights_first_and_skip_done_faces() {
        let mut cache = [SlotCache::default(); MAX_LOCAL_SHADOWS];
        let slots = [slot(0, 10.0), slot(1, 1.0), slot(2, 5.0)];
        cache[1].done = 0b00_0011;
        let jobs = pick_jobs(&slots, &cache, Some([0.0; 3]), 7);
        let picked: Vec<(u32, u32)> = jobs.iter().map(|j| (j.slot, j.face)).collect();
        assert_eq!(
            picked,
            [(1, 2), (1, 3), (1, 4), (1, 5), (2, 0), (2, 1), (2, 2)]
        );
        cache[1].done = ALL_FACES;
        cache[2].done = ALL_FACES;
        cache[0].done = ALL_FACES;
        assert!(pick_jobs(&slots, &cache, None, 7).is_empty());
    }

    #[test]
    fn sphere_box_overlap() {
        assert!(sphere_overlaps(
            [0.0; 3],
            1.0,
            [0.5, -1.0, -1.0],
            [2.0, 1.0, 1.0]
        ));
        assert!(!sphere_overlaps(
            [0.0; 3],
            1.0,
            [1.5, -1.0, -1.0],
            [2.0, 1.0, 1.0]
        ));
    }
}
