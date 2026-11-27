//! Native foliage reference views and convex-hull bake layers (M7, contract
//! m7-reference-views).
//!
//! [`render_foliage_references`] renders a retained [`FoliageSource`] from deterministic
//! spherical cameras in source-root space: a world normal/coverage target, a standard-Z
//! depth target and a linear unlit albedo/coverage target, one D2-array layer per view.
//! The ordered camera table stores each view's column-major root-to-clip matrix and
//! near/far metres.
//!
//! [`pack_foliage_layers`] converts those references into the packed layers of the native
//! resource generation: it copies the unlit albedo/coverage unchanged and runs
//! `foliage_pack.wgsl` to rotate the world normals back into source-root space and to keep
//! the standard-Z depth in W. Both functions return submitted GPU resources; no CPU texel
//! is read back and no live scene camera or instance visibility is touched.

use std::collections::HashMap;

use raa_core::{GpuContext, MaterialId, RaaError};
use raa_math::{Mat4, Vec3};
use raa_scene::{Camera, Scene};
use raa_shader::{Composer, MaterialFnId, PassKind, VariantKey};

use crate::extras::{INSTANCE_EXTRA_GPU_SIZE, InstanceExtra};
use crate::foliage_resources::{FoliageReferences, FoliageSource, PackedFoliageLayers};
use crate::frame::{FRAME_GPU_SIZE, FrameParams, write_frame_gpu};
use crate::handle::LitHandle;
use crate::instance_store::write_instance_record;
use crate::material::FLAG_DOUBLE_SIDED;
use crate::prep::INSTANCE_GPU_SIZE;
use crate::renderer::{REFERENCE_TARGET_FORMATS, SourceDrawBuffers, reference_pipeline};

/// Golden angle of the deterministic spherical view distribution.
const GOLDEN_ANGLE: f32 = std::f32::consts::PI * (3.0 - 2.236_068);
/// Vertical field of view of every reference camera.
const REFERENCE_FOV_Y: f32 = std::f32::consts::FRAC_PI_4;
/// The reference camera distance as a multiple of the bounding-sphere radius.
const DISTANCE_FACTOR: f32 = 3.0;
/// Workgroup edge of `foliage_pack.wgsl`.
const PACK_WORKGROUP: u32 = 8;

fn invalid(message: String) -> RaaError {
    RaaError::FrameGraph {
        code: "foliage.invalid",
        message,
    }
}

fn shader_error(e: raa_shader::ShaderError) -> RaaError {
    RaaError::FrameGraph {
        code: e.code,
        message: e.message,
    }
}

/// Refuses a requested view count: only 8 (impostor preparation) and 64..=128 (native
/// optimization) are supported. Every other count returns `foliage.invalid` before any
/// GPU work.
pub fn validate_view_count(views: u32) -> Result<(), RaaError> {
    if views == 8 || (64..=128).contains(&views) {
        Ok(())
    } else {
        Err(invalid(format!(
            "foliage reference views {views} must be 8 or within 64..=128"
        )))
    }
}

/// Refuses a texture resolution other than 64, 128, 256 or 512 with `foliage.invalid`.
pub fn validate_resolution(resolution: u32) -> Result<(), RaaError> {
    if matches!(resolution, 64 | 128 | 256 | 512) {
        Ok(())
    } else {
        Err(invalid(format!(
            "foliage reference resolution {resolution} must be 64, 128, 256 or 512"
        )))
    }
}

/// Checks that the three reference arrays share one square extent and layer count and
/// that the camera table has exactly one record per layer. Each triple is
/// `[width, height, layers]`. Mismatched extents return `foliage.invalid`.
pub fn validate_reference_extents(
    albedo: [u32; 3],
    normal: [u32; 3],
    depth: [u32; 3],
    views: usize,
) -> Result<(), RaaError> {
    for (name, extent) in [
        ("albedo_coverage", albedo),
        ("normal_coverage", normal),
        ("depth", depth),
    ] {
        if extent[0] == 0 || extent[1] == 0 || extent[2] == 0 {
            return Err(invalid(format!("foliage {name} reference array is empty")));
        }
        if extent[0] != extent[1] {
            return Err(invalid(format!(
                "foliage {name} reference array is not square ({}x{})",
                extent[0], extent[1]
            )));
        }
    }
    if albedo != normal || albedo != depth {
        return Err(invalid(format!(
            "foliage reference arrays disagree: albedo {albedo:?}, normal {normal:?}, depth {depth:?}"
        )));
    }
    if albedo[2] as usize != views {
        return Err(invalid(format!(
            "foliage reference table has {views} view(s), expected {}",
            albedo[2]
        )));
    }
    Ok(())
}

/// One deterministic spherical reference camera. `root` is the source-root camera used to
/// write the frame, `root_to_clip` is its column-major view-projection (the table record),
/// and `world_to_clip` / `world_view` render the same view with world-space instances.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ReferenceCamera {
    pub root: Camera,
    pub root_to_clip: Mat4,
    pub world_to_clip: Mat4,
    pub world_view: Mat4,
    pub eye_world: Vec3,
    pub near: f32,
    pub far: f32,
}

fn fibonacci_direction(index: u32, views: u32) -> Vec3 {
    let k = index as f32 + 0.5;
    let y = 1.0 - 2.0 * k / views as f32;
    let r = (1.0 - y * y).max(0.0).sqrt();
    let phi = k * GOLDEN_ANGLE;
    Vec3::new(r * phi.cos(), y, r * phi.sin())
}

/// Generates `views` deterministic spherical cameras around the source-root bounds: a
/// Fibonacci-sphere eye at three bounding-sphere radii from the center, looking at the
/// center, with `near = radius`, `far = 5 * radius` (standard-Z) and a 45 degree
/// field of view. The tuple order is the view table order. `root_world` maps source-root
/// positions to world positions and produces the independent world-to-root matrices.
pub fn reference_cameras(
    min: [f32; 3],
    max: [f32; 3],
    root_world: Mat4,
    views: u32,
) -> Result<Vec<ReferenceCamera>, RaaError> {
    validate_view_count(views)?;
    if !min.iter().chain(max.iter()).all(|v| v.is_finite()) {
        return Err(invalid("foliage reference bounds must be finite".into()));
    }
    for axis in 0..3 {
        if min[axis] >= max[axis] {
            return Err(invalid(format!(
                "foliage reference bounds are degenerate on axis {axis}: {} >= {}",
                min[axis], max[axis]
            )));
        }
    }
    let root_inverse = root_world
        .inverse()
        .ok_or_else(|| invalid("the foliage source root frame is not invertible".into()))?;
    let center = Vec3::from_array([0, 1, 2].map(|i| 0.5 * (min[i] + max[i])));
    let extent = Vec3::from_array([0, 1, 2].map(|i| max[i] - min[i]));
    let radius = (0.5 * extent.length()).max(1e-4);
    let distance = DISTANCE_FACTOR * radius;
    let near = (distance - 2.0 * radius).max(1e-4);
    let far = distance + 2.0 * radius;
    let mut cameras = Vec::with_capacity(views as usize);
    for index in 0..views {
        let dir = fibonacci_direction(index, views);
        let eye = center + dir * distance;
        let forward = -dir;
        let hint = if forward.dot(Vec3::Y).abs() > 0.99 {
            Vec3::X
        } else {
            Vec3::Y
        };
        let up = (hint - forward * forward.dot(hint)).normalize_or_zero();
        let root = Camera {
            eye,
            target: center,
            up,
            fov_y_rad: REFERENCE_FOV_Y,
            near,
            far,
            aspect: 1.0,
            ortho_height: None,
        };
        let root_to_clip = root.view_proj();
        let world_view = root.view() * root_inverse;
        cameras.push(ReferenceCamera {
            root,
            root_to_clip,
            world_to_clip: root_to_clip * root_inverse,
            world_view,
            eye_world: root_world.transform_point(eye),
            near,
            far,
        });
    }
    Ok(cameras)
}

/// The world-to-root normal matrix of `root_world` as three vec4 columns (the w lanes are
/// zero). These are the values `foliage_pack.wgsl` applies to the reference world normals;
/// a singular source frame is `foliage.invalid`.
pub fn world_to_root_normal(root_world: Mat4) -> Result<[[f32; 4]; 3], RaaError> {
    let inverse = root_world
        .inverse()
        .ok_or_else(|| invalid("the foliage source root frame is not invertible".into()))?;
    Ok(inverse.normal_matrix())
}

fn reference_texture(
    gpu: &GpuContext,
    label: &str,
    resolution: u32,
    views: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: resolution,
            height: resolution,
            depth_or_array_layers: views,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

fn layer_view(texture: &wgpu::Texture, layer: u32) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("raa.lit.foliage_reference.layer"),
        format: None,
        dimension: Some(wgpu::TextureViewDimension::D2),
        usage: None,
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: None,
        base_array_layer: layer,
        array_layer_count: Some(1),
    })
}

fn clear_color(view: &wgpu::TextureView) -> Option<wgpu::RenderPassColorAttachment<'_>> {
    Some(wgpu::RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            store: wgpu::StoreOp::Store,
        },
    })
}

/// The byte size `count x per_item`, refused with a message naming the request when it
/// overflows `usize`.
fn buffer_bytes(what: &str, count: usize, per_item: usize) -> Result<usize, RaaError> {
    count.checked_mul(per_item).ok_or_else(|| {
        invalid(format!(
            "foliage reference {what} buffer of {count} x {per_item} bytes overflows"
        ))
    })
}

/// An empty buffer with exactly `bytes` reserved, or `foliage.invalid` naming the byte
/// count when the allocation is refused (never an allocation-failure abort).
fn reserve_bytes(what: &str, bytes: usize) -> Result<Vec<u8>, RaaError> {
    let mut buffer = Vec::new();
    buffer.try_reserve_exact(bytes).map_err(|_| {
        invalid(format!(
            "foliage reference {what} buffer could not allocate {bytes} bytes"
        ))
    })?;
    Ok(buffer)
}

/// Bytes of stream 0 (`position`, `normal`) per source corner in the S10 lit layout.
pub const REFERENCE_STREAM0_CORNER_BYTES: usize = 24;
/// Bytes of stream 1 (`uv`, `tangent`, `colour`) per source corner in the S10 lit layout.
pub const REFERENCE_STREAM1_CORNER_BYTES: usize = 40;
/// Source triangles encoded per vertex-stream upload window: the heap holds one window of
/// both streams (`4096 * 3 * 64` bytes), never the streams of the whole source.
pub const REFERENCE_STREAM_CHUNK_TRIANGLES: usize = 4096;

/// One reference draw: `count` consecutive source triangles from `first` that share one
/// compact material slot and one source tag, drawn with the one instance record `index`
/// of its run (`draw(3 * first..3 * (first + count), index..index + 1)`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ReferenceRun {
    pub first: u32,
    pub count: u32,
    pub slot: u32,
    pub source_tag: u32,
}

/// The host-side draw plan of the reference views: the distinct material table, the runs
/// and one instance record, extra and id per run (never per triangle). Every array is
/// sized from the source's distinct materials and runs and reserved fallibly.
pub struct ReferenceDrawPlan {
    pub material_ids: Vec<u32>,
    pub runs: Vec<ReferenceRun>,
    pub instances: Vec<u8>,
    pub extras: Vec<u8>,
    pub ids: Vec<u8>,
}

impl ReferenceDrawPlan {
    /// The heap bytes the plan holds.
    pub fn heap_bytes(&self) -> usize {
        self.material_ids.capacity() * 4
            + self.runs.capacity() * std::mem::size_of::<ReferenceRun>()
            + self.instances.capacity()
            + self.extras.capacity()
            + self.ids.capacity()
    }
}

fn push_fallible<T>(list: &mut Vec<T>, value: T, what: &str) -> Result<(), RaaError> {
    list.try_reserve(1).map_err(|_| {
        invalid(format!(
            "foliage reference {what} could not grow past {} entries",
            list.len()
        ))
    })?;
    list.push(value);
    Ok(())
}

/// Builds the [`ReferenceDrawPlan`] of `source`. The material table is the ascending
/// distinct ids (as [`material_slots`] orders them), found without a per-triangle array;
/// consecutive triangles with the same slot and tag share one run and its record. A
/// refused allocation is `foliage.invalid`, never an allocation-failure abort.
pub fn reference_draw_plan(source: &FoliageSource) -> Result<ReferenceDrawPlan, RaaError> {
    let mut material_ids: Vec<u32> = Vec::new();
    for triangle in &source.triangles {
        if let Err(at) = material_ids.binary_search(&triangle.material) {
            material_ids
                .try_reserve(1)
                .map_err(|_| invalid("foliage material table could not grow".to_string()))?;
            material_ids.insert(at, triangle.material);
        }
    }
    let mut runs: Vec<ReferenceRun> = Vec::new();
    for (index, triangle) in source.triangles.iter().enumerate() {
        let index = u32::try_from(index)
            .map_err(|_| invalid(format!("foliage triangle index {index} exceeds u32")))?;
        let slot = material_ids
            .binary_search(&triangle.material)
            .map_err(|_| {
                invalid(format!(
                    "foliage material {} has no slot",
                    triangle.material
                ))
            })? as u32;
        match runs.last_mut() {
            Some(run)
                if run.slot == slot
                    && run.source_tag == triangle.source_tag
                    && run.first + run.count == index =>
            {
                run.count += 1;
            }
            _ => push_fallible(
                &mut runs,
                ReferenceRun {
                    first: index,
                    count: 1,
                    slot,
                    source_tag: triangle.source_tag,
                },
                "run list",
            )?,
        }
    }
    let mut instances = reserve_bytes(
        "instance",
        buffer_bytes("instance", runs.len(), INSTANCE_GPU_SIZE)?,
    )?;
    let mut extras = reserve_bytes(
        "extras",
        buffer_bytes("extras", runs.len(), INSTANCE_EXTRA_GPU_SIZE)?,
    )?;
    let mut ids = reserve_bytes("id", buffer_bytes("id", runs.len(), 4)?)?;
    let mut extra = Vec::with_capacity(INSTANCE_EXTRA_GPU_SIZE);
    InstanceExtra::default().write_gpu(&mut extra);
    for (index, run) in runs.iter().enumerate() {
        let mut record = [0u8; INSTANCE_GPU_SIZE];
        write_instance_record(&mut record, &source.root_world, run.slot, run.source_tag, 0);
        instances.extend_from_slice(&record);
        extras.extend_from_slice(&extra);
        ids.extend_from_slice(&(index as u32).to_le_bytes());
    }
    Ok(ReferenceDrawPlan {
        material_ids,
        runs,
        instances,
        extras,
        ids,
    })
}

/// Encodes stream 0 and stream 1 of the source triangles `first..first + count` into the
/// cleared `stream0` and `stream1`, reserving their exact window size fallibly. The bytes
/// are those of the S10 lit layouts: position and normal, then uv, the unit tangent
/// `(1, 0, 0, 1)` and the white colour per corner.
pub fn encode_reference_streams(
    source: &FoliageSource,
    first: usize,
    count: usize,
    stream0: &mut Vec<u8>,
    stream1: &mut Vec<u8>,
) -> Result<(), RaaError> {
    let triangles = source
        .triangles
        .get(first..first.saturating_add(count))
        .ok_or_else(|| {
            invalid(format!(
                "foliage stream window {first}+{count} leaves the source"
            ))
        })?;
    stream0.clear();
    stream1.clear();
    let corners = buffer_bytes("corner", count, 3)?;
    for (stream, per_corner, what) in [
        (&mut *stream0, REFERENCE_STREAM0_CORNER_BYTES, "stream0"),
        (&mut *stream1, REFERENCE_STREAM1_CORNER_BYTES, "stream1"),
    ] {
        let bytes = buffer_bytes(what, corners, per_corner)?;
        stream.try_reserve_exact(bytes).map_err(|_| {
            invalid(format!(
                "foliage reference {what} window could not allocate {bytes} bytes"
            ))
        })?;
    }
    for triangle in triangles {
        for corner in 0..3 {
            for value in triangle.positions[corner] {
                stream0.extend_from_slice(&value.to_le_bytes());
            }
            for value in triangle.normals[corner] {
                stream0.extend_from_slice(&value.to_le_bytes());
            }
            for value in triangle.uv[corner] {
                stream1.extend_from_slice(&value.to_le_bytes());
            }
            for value in [1.0f32, 0.0, 0.0, 1.0] {
                stream1.extend_from_slice(&value.to_le_bytes());
            }
            for value in [1.0f32, 1.0, 1.0, 1.0] {
                stream1.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    Ok(())
}

/// The two source vertex buffers of `source`, uploaded window by window through
/// [`encode_reference_streams`] so the heap holds one window, never the whole streams.
fn upload_reference_streams(
    gpu: &GpuContext,
    source: &FoliageSource,
    stream0_bytes: u64,
    stream1_bytes: u64,
) -> Result<(wgpu::Buffer, wgpu::Buffer), RaaError> {
    let buffer = |label: &str, size: u64| {
        gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(4),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let mut window0 = Vec::new();
    let mut window1 = Vec::new();
    let total = source.triangles.len();
    if total <= REFERENCE_STREAM_CHUNK_TRIANGLES {
        // One window holds the whole source: the shared source upload path takes it.
        encode_reference_streams(source, 0, total, &mut window0, &mut window1)?;
        return Ok((
            crate::renderer::source_vertex_buffer(gpu, "stream0", &window0),
            crate::renderer::source_vertex_buffer(gpu, "stream1", &window1),
        ));
    }
    let stream0 = buffer("raa.lit.source.stream0", stream0_bytes);
    let stream1 = buffer("raa.lit.source.stream1", stream1_bytes);
    let mut first = 0usize;
    while first < total {
        let count = REFERENCE_STREAM_CHUNK_TRIANGLES.min(total - first);
        encode_reference_streams(source, first, count, &mut window0, &mut window1)?;
        let corner = first as u64 * 3;
        gpu.queue().write_buffer(
            &stream0,
            corner * REFERENCE_STREAM0_CORNER_BYTES as u64,
            &window0,
        );
        gpu.queue().write_buffer(
            &stream1,
            corner * REFERENCE_STREAM1_CORNER_BYTES as u64,
            &window1,
        );
        first += count;
    }
    Ok((stream0, stream1))
}

/// The distinct material ids the triangles use, in ascending id order, and each
/// triangle's slot in that list. Nothing is sized or looped by the numeric id value; the
/// reference draws index the material table by slot. A refused allocation is
/// `foliage.invalid` naming the byte count.
pub fn material_slots(source: &FoliageSource) -> Result<(Vec<u32>, Vec<u32>), RaaError> {
    let count = source.triangles.len();
    let mut ids: Vec<u32> = Vec::new();
    ids.try_reserve_exact(count).map_err(|_| {
        invalid(format!(
            "foliage material table could not allocate {} bytes",
            count.saturating_mul(4)
        ))
    })?;
    ids.extend(source.triangles.iter().map(|t| t.material));
    ids.sort_unstable();
    ids.dedup();
    let mut slots: Vec<u32> = Vec::new();
    slots.try_reserve_exact(count).map_err(|_| {
        invalid(format!(
            "foliage material slots could not allocate {} bytes",
            count.saturating_mul(4)
        ))
    })?;
    for triangle in &source.triangles {
        let slot = ids.binary_search(&triangle.material).map_err(|_| {
            invalid(format!(
                "foliage material {} has no slot",
                triangle.material
            ))
        })?;
        slots.push(
            u32::try_from(slot)
                .map_err(|_| invalid("foliage material table exceeds u32".into()))?,
        );
    }
    Ok((ids, slots))
}

/// Refuses with a message, instead of a `RefCell` panic, when the renderer state is
/// mutably borrowed. Every `LitHandle` accessor this module calls borrows it immutably and
/// returns before the next call, so a free state here stays free for those calls.
fn ensure_lit_free(lit: &LitHandle) -> Result<(), RaaError> {
    lit.shared.try_borrow().map(|_| ()).map_err(|_| {
        invalid("the renderer state is borrowed while the foliage references render".into())
    })
}

fn compose_reference(lit: &LitHandle, key: &VariantKey) -> Result<String, RaaError> {
    let shared = lit.shared.try_borrow().map_err(|_| {
        invalid("the renderer state is borrowed while the foliage reference composes".into())
    })?;
    shared
        .composer()
        .compose_foliage_reference(&shared.fns, key)
        .map_err(shader_error)
}

fn reference_pipelines(
    gpu: &GpuContext,
    lit: &LitHandle,
    classes: &[(MaterialFnId, u32)],
) -> Result<HashMap<(u32, bool), wgpu::RenderPipeline>, RaaError> {
    ensure_lit_free(lit)?;
    let frame_layout = lit.frame_layout();
    let mut pipelines = HashMap::new();
    for &(fn_id, flags) in classes {
        let double_sided = flags & FLAG_DOUBLE_SIDED != 0;
        if pipelines.contains_key(&(fn_id.0, double_sided)) {
            continue;
        }
        let key = VariantKey {
            material_fn: fn_id,
            pass: PassKind::Prepass,
            msaa: 1,
            flags: if double_sided { FLAG_DOUBLE_SIDED } else { 0 },
        };
        let wgsl = compose_reference(lit, &key)?;
        Composer::validate(&wgsl).map_err(shader_error)?;
        let module = gpu
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("raa.lit.foliage_reference"),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
        pipelines.insert(
            (fn_id.0, double_sided),
            reference_pipeline(gpu, &module, &frame_layout, double_sided),
        );
    }
    Ok(pipelines)
}

fn write_view_frame(
    out: &mut [u8; FRAME_GPU_SIZE],
    camera: &ReferenceCamera,
    resolution: u32,
) -> Result<(), RaaError> {
    let mut scene = Scene::new();
    scene.camera = camera.root;
    write_frame_gpu(
        out,
        &scene,
        &FrameParams {
            viewport: (resolution, resolution),
            msaa: 1,
            ..FrameParams::default()
        },
    )?;
    camera.world_to_clip.write_le_bytes(&mut out[0..64]);
    camera.world_view.write_le_bytes(&mut out[64..128]);
    if let Some(inverse) = camera.world_view.inverse() {
        inverse.write_le_bytes(&mut out[256..320]);
    }
    let eye = camera.eye_world;
    for (slot, value) in [eye.x, eye.y, eye.z, 0.0].iter().enumerate() {
        out[320 + 4 * slot..324 + 4 * slot].copy_from_slice(&value.to_le_bytes());
    }
    camera.world_to_clip.write_le_bytes(&mut out[576..640]);
    Ok(())
}

/// Renders `source` from `views` deterministic spherical cameras at `resolution` into the
/// three native reference arrays and returns them with the ordered root-to-clip camera
/// table and the submission index.
///
/// The draws are the retained source triangles, submitted with the same private group-0
/// buffers as the M6 billboard bake ([`SourceDrawBuffers`]); every instance uses the
/// source-root object frame, so positions are world-space while the depth written is the
/// root-to-clip standard-Z value. The live scene camera and every instance's visibility
/// are untouched. `foliage.invalid` is returned for an empty source, an unsupported view
/// count (see [`validate_view_count`]) or resolution, or a singular source frame.
///
/// Handoff: 'Run differentiable foliage optimization' calls [`render_foliage_references`];
/// today it renders the reference arrays and the ordered camera table and returns them
/// with the submitted copy index.
pub fn render_foliage_references(
    gpu: &GpuContext,
    lit: &LitHandle,
    source: &FoliageSource,
    views: u32,
    resolution: u32,
) -> Result<FoliageReferences, RaaError> {
    validate_view_count(views)?;
    validate_resolution(resolution)?;
    if source.triangles.is_empty() {
        return Err(invalid("the foliage source is empty".into()));
    }
    gpu.ensure_live()?;
    let cameras = reference_cameras(source.min, source.max, source.root_world, views)?;

    ensure_lit_free(lit)?;
    let plan = reference_draw_plan(source)?;
    let material_ids = &plan.material_ids;
    let mut material_bytes = reserve_bytes(
        "material",
        buffer_bytes(
            "material",
            material_ids.len(),
            crate::SURFACE_MATERIAL_GPU_SIZE,
        )?,
    )?;
    let mut classes = Vec::new();
    classes.try_reserve_exact(material_ids.len()).map_err(|_| {
        invalid(format!(
            "foliage material classes could not allocate {} entries",
            material_ids.len()
        ))
    })?;
    for &id in material_ids {
        let record = lit.material_record(MaterialId(id)).unwrap_or_default();
        classes.push((lit.material_fn_of(MaterialId(id)), record.flags));
        record.write_gpu(&mut material_bytes);
    }
    let pipelines = reference_pipelines(gpu, lit, &classes)?;
    let corners = buffer_bytes("corner", source.triangles.len(), 3)?;
    let stream0_bytes = buffer_bytes("stream0", corners, REFERENCE_STREAM0_CORNER_BYTES)?;
    let stream1_bytes = buffer_bytes("stream1", corners, REFERENCE_STREAM1_CORNER_BYTES)?;
    let max_buffer = gpu.device().limits().max_buffer_size;
    for (what, len) in [
        ("stream0", stream0_bytes),
        ("stream1", stream1_bytes),
        ("instance", plan.instances.len()),
    ] {
        if len as u64 > max_buffer {
            return Err(invalid(format!(
                "foliage reference {what} buffer of {len} bytes exceeds the device limit of {max_buffer} bytes"
            )));
        }
    }
    // The previous-affine buffer `SourceDrawBuffers::new` zero-fills holds one record per
    // run: proven allocatable here so it is an error, not an allocation abort.
    drop(reserve_bytes(
        "previous",
        buffer_bytes("previous", plan.runs.len(), 64)?,
    )?);
    ensure_lit_free(lit)?;
    let buffers = SourceDrawBuffers::new(
        gpu,
        lit,
        &material_bytes,
        &plan.instances,
        &plan.extras,
        &plan.ids,
    );

    let array_usage = wgpu::TextureUsages::RENDER_ATTACHMENT
        | wgpu::TextureUsages::TEXTURE_BINDING
        | wgpu::TextureUsages::COPY_SRC;
    let normal_coverage = reference_texture(
        gpu,
        "raa.lit.foliage_reference.normal_coverage",
        resolution,
        views,
        REFERENCE_TARGET_FORMATS[0],
        array_usage,
    );
    let depth = reference_texture(
        gpu,
        "raa.lit.foliage_reference.depth",
        resolution,
        views,
        REFERENCE_TARGET_FORMATS[1],
        array_usage,
    );
    let albedo_coverage = reference_texture(
        gpu,
        "raa.lit.foliage_reference.albedo_coverage",
        resolution,
        views,
        REFERENCE_TARGET_FORMATS[2],
        array_usage,
    );
    let depth_texture = reference_texture(
        gpu,
        "raa.lit.foliage_reference.depth_attachment",
        resolution,
        views,
        wgpu::TextureFormat::Depth32Float,
        wgpu::TextureUsages::RENDER_ATTACHMENT,
    );

    let (stream0, stream1) =
        upload_reference_streams(gpu, source, stream0_bytes as u64, stream1_bytes as u64)?;
    let mut submission = None;
    for (view, camera) in cameras.iter().enumerate() {
        let mut frame = [0u8; FRAME_GPU_SIZE];
        write_view_frame(&mut frame, camera, resolution)?;
        gpu.queue().write_buffer(&buffers.frame, 0, &frame);
        let normal_view = layer_view(&normal_coverage, view as u32);
        let depth_view = layer_view(&depth, view as u32);
        let albedo_view = layer_view(&albedo_coverage, view as u32);
        let attachment = layer_view(&depth_texture, view as u32);
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("raa.lit.foliage_reference"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("raa.lit.foliage_reference.render"),
                color_attachments: &[
                    clear_color(&normal_view),
                    clear_color(&depth_view),
                    clear_color(&albedo_view),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &attachment,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &buffers.bind, &[]);
            pass.set_vertex_buffer(0, stream0.slice(..));
            pass.set_vertex_buffer(1, buffers.ids.slice(..));
            pass.set_vertex_buffer(2, stream1.slice(..));
            for (index, run) in plan.runs.iter().enumerate() {
                let slot = run.slot;
                let &(fn_id, flags) = classes
                    .get(slot as usize)
                    .ok_or_else(|| invalid(format!("foliage material slot {slot} has no class")))?;
                let pipeline = pipelines
                    .get(&(fn_id.0, flags & FLAG_DOUBLE_SIDED != 0))
                    .ok_or_else(|| {
                        invalid(format!(
                            "foliage material function {} has no reference pipeline",
                            fn_id.0
                        ))
                    })?;
                pass.set_pipeline(pipeline);
                let index = index as u32;
                let range = run
                    .first
                    .checked_mul(3)
                    .zip((run.first + run.count).checked_mul(3))
                    .ok_or_else(|| invalid(format!("foliage run {index} exceeds u32 vertices")))?;
                pass.draw(range.0..range.1, index..index + 1);
            }
        }
        submission = Some(gpu.queue().submit([encoder.finish()]));
    }
    let submission =
        submission.ok_or_else(|| invalid("no foliage reference view was submitted".into()))?;
    let table = cameras
        .iter()
        .map(|c| (c.root_to_clip.to_cols_array(), [c.near, c.far]))
        .collect();
    Ok(FoliageReferences {
        normal_coverage,
        depth,
        albedo_coverage,
        views: table,
        submission,
    })
}

/// Compute group 0 of `foliage_pack.wgsl`: the world-to-root normal matrix, the
/// normal/coverage and depth reference arrays, and the writable packed normal/depth array.
fn pack_bind_group_layout(gpu: &GpuContext) -> wgpu::BindGroupLayout {
    let compute = wgpu::ShaderStages::COMPUTE;
    gpu.device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa.lit.foliage_pack.layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: compute,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(48),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: compute,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: compute,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: compute,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
            ],
        })
}

fn normal_uniform(matrix: &[[f32; 4]; 3]) -> [u8; 48] {
    let mut out = [0u8; 48];
    for (column, values) in matrix.iter().enumerate() {
        for (row, value) in values.iter().enumerate() {
            let offset = column * 16 + row * 4;
            out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
    out
}

/// The CPU mirror of one `foliage_pack.wgsl` texel: `world_to_root` (the
/// [`world_to_root_normal`] columns) rotates the reference world normal of
/// `normal_coverage` into a unit source-root normal (zero when degenerate) and the
/// reference standard-Z `depth` is clamped to 0..=1 into W. The coverage stays in the
/// copied albedo/coverage layer, so a covered texel is one whose reference wrote a depth
/// below the clear depth 1.
pub fn pack_texel(
    world_to_root: &[[f32; 4]; 3],
    normal_coverage: [f32; 4],
    depth: f32,
) -> [f32; 4] {
    let [x, y, z, _] = normal_coverage;
    let n = [0, 1, 2].map(|row| {
        world_to_root[0][row] * x + world_to_root[1][row] * y + world_to_root[2][row] * z
    });
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    let n = if len > 1e-6 {
        n.map(|v| v / len)
    } else {
        [0.0; 3]
    };
    [n[0], n[1], n[2], depth.clamp(0.0, 1.0)]
}

/// Converts the reference views into the packed layers of one native foliage generation:
/// the albedo/coverage array is copied unchanged, and `foliage_pack.wgsl` rotates the
/// world normals into source-root space and writes the standard-Z depth into W of a new
/// Rgba16Float normal/depth array. The ordered camera table is retained.
///
/// Empty or inconsistent references (mismatched extents, formats or table length) return
/// `foliage.invalid`, as does a singular source-root frame. The result holds submitted GPU
/// resources, not CPU image data.
///
/// Handoff: 'Publish completed foliage assets in the renderer' calls
/// [`pack_foliage_layers`] after [`render_foliage_references`] and passes its
/// [`PackedFoliageLayers`] to `FoliageResources::initialize`; today it returns the packed
/// arrays and the same ordered camera table with their own submission.
pub fn pack_foliage_layers(
    gpu: &GpuContext,
    source: &FoliageSource,
    references: &FoliageReferences,
) -> Result<PackedFoliageLayers, RaaError> {
    let albedo = references.albedo_coverage();
    let normal = references.normal_coverage();
    let depth = references.depth();
    for (name, texture, format) in [
        ("albedo_coverage", albedo, wgpu::TextureFormat::Rgba8Unorm),
        ("normal_coverage", normal, wgpu::TextureFormat::Rgba16Float),
        ("depth", depth, wgpu::TextureFormat::Rgba32Float),
    ] {
        if texture.format() != format {
            return Err(invalid(format!(
                "foliage {name} reference must use {format:?}"
            )));
        }
        if texture.dimension() != wgpu::TextureDimension::D2 || texture.mip_level_count() != 1 {
            return Err(invalid(format!(
                "foliage {name} reference must be a mip-1 D2 array"
            )));
        }
    }
    let extent = |t: &wgpu::Texture| [t.width(), t.height(), t.depth_or_array_layers()];
    validate_reference_extents(
        extent(albedo),
        extent(normal),
        extent(depth),
        references.views.len(),
    )?;
    gpu.ensure_live()?;
    let resolution = albedo.width();
    let views = albedo.depth_or_array_layers();
    validate_view_count(views)?;
    validate_resolution(resolution)?;
    let world_to_root = world_to_root_normal(source.root_world)?;

    let albedo_out = reference_texture(
        gpu,
        "raa.lit.foliage_pack.albedo_coverage",
        resolution,
        views,
        wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
    );
    let normal_depth = reference_texture(
        gpu,
        "raa.lit.foliage_pack.normal_depth",
        resolution,
        views,
        wgpu::TextureFormat::Rgba16Float,
        wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
    );
    let uniform = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("raa.lit.foliage_pack.params"),
        size: 48,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue()
        .write_buffer(&uniform, 0, &normal_uniform(&world_to_root));
    let layout = pack_bind_group_layout(gpu);
    let bind = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("raa.lit.foliage_pack.bind"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(
                    &normal.create_view(&Default::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(
                    &depth.create_view(&Default::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(
                    &normal_depth.create_view(&Default::default()),
                ),
            },
        ],
    });
    let module = gpu
        .device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa.lit.foliage_pack"),
            source: wgpu::ShaderSource::Wgsl(include_str!("wgsl/foliage_pack.wgsl").into()),
        });
    let pipeline_layout = gpu
        .device()
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raa.lit.foliage_pack.pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
    let pipeline = gpu
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("raa.lit.foliage_pack.pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
    let mut encoder = gpu
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("raa.lit.foliage_pack"),
        });
    encoder.copy_texture_to_texture(
        albedo.as_image_copy(),
        albedo_out.as_image_copy(),
        wgpu::Extent3d {
            width: resolution,
            height: resolution,
            depth_or_array_layers: views,
        },
    );
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("raa.lit.foliage_pack.compute"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        let groups = resolution.div_ceil(PACK_WORKGROUP);
        pass.dispatch_workgroups(groups, groups, views);
    }
    let submission = gpu.queue().submit([encoder.finish()]);
    Ok(PackedFoliageLayers {
        albedo_coverage: albedo_out,
        normal_depth,
        views: references.views.clone(),
        submission,
    })
}
