//! Section 67 evaluator and authoritative internal volume ABI (see [`VOLUME_ABI_VERSION`]).
//! [`volume_module`] supplies the primary renderer; [`trace_module`] is the analytic fallback.
//!
//! [`PROGRAM_EVAL_WGSL`] declares the two packed program buffers at group 0 bindings 0 and 1 and
//! defines the evaluator functions; it has no entry points. The trace pass and the GPU scatter pass
//! (M12) include it and add their own bindings at other slots. [`eval_module`] wraps it into a
//! complete compute module for naga validation.

use std::fmt::Write;

use crate::program::TraceProgram;

/// Packed-program evaluator: local transforms, the eight primitives, the three smooth
/// combinations, material selection, the normalized rail-bend warp and the distance scale.
/// The data offset is relative to the source's own data span, exactly as in the packed file.
pub const PROGRAM_EVAL_WGSL: &str = include_str!("wgsl/program_eval_wgsl.wgsl");

/// A complete compute module for `program`: the shared evaluator, the packed buffers it binds
/// and a `sdf_eval` entry, with the program's source, instance and bend counts as constants.
pub fn eval_module(program: &TraceProgram) -> String {
    let source = program.sources.first();
    let (code_len, data_start) = source.map_or((0, 0), |s| (s.code_len, s.data_start));
    let bends = program
        .instances
        .iter()
        .filter(|i| i.deform_kind == 1)
        .count();
    let mut out = String::from(PROGRAM_EVAL_WGSL);
    let _ = write!(
        out,
        "\n@group(0) @binding(2) var<storage, read_write> trace_result: array<vec4<f32>>;
const CODE_LEN: u32 = {code_len}u;
const DATA_START: u32 = {data_start}u;
const INSTANCE_COUNT: u32 = {}u;
const BEND_COUNT: u32 = {bends}u;
@compute @workgroup_size(64)
fn sdf_eval(@builtin(global_invocation_id) id: vec3<u32>) {{
  if (id.x >= INSTANCE_COUNT) {{ return; }}
  let value = trace_instance(0u, CODE_LEN, DATA_START,
    vec4<f32>(1.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 1.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 1.0, 0.0),
    1.0, 0u, vec4<f32>(0.0), vec4<f32>(1.0, 0.0, 0.0, 0.0), 0.0, 0.0, 0.0,
    vec3<f32>(f32(id.x), f32(id.y), 0.0));
  trace_result[id.x] = vec4<f32>(value.d, f32(trace_material(-1, value)), 0.0, 1.0);
}}
",
        program.instances.len(),
    );
    out
}

// Byte-compatible with raa-lit's 512-byte FrameGpu; light colours include intensity.
const TRACE_FRAME_WGSL: &str = include_str!("wgsl/trace_frame_wgsl.wgsl");

/// Complete `sdf_trace` compute module (8x8 workgroups), including the shared evaluator.
/// Group 0: 0 code, 1 data, 2 sources (48-byte file records), 3 instances (128-byte file
/// records), 4 normalized bends (two f32 vec4s per instance, zero for rigid instances),
/// 5 BVH nodes (min.xyz, left, max.xyz, count), 6 BVH order, 7 materials (32-byte records),
/// 8 FrameGpu, 9 rgba16float colour, 10 r32float depth. Upload dummy records for empty buffers.
/// Analytic fallback; target dimensions are unrestricted and dispatch rounds up by 8.
pub fn trace_module(program: &TraceProgram) -> String {
    trace_shader(program, TRACE_FALLBACK_WGSL)
}

fn trace_shader(program: &TraceProgram, traversal: &str) -> String {
    format!(
        "{PROGRAM_EVAL_WGSL}{TRACE_FRAME_WGSL}\nconst TRACE_COUNT: u32 = {}u;\n{TRACE_WGSL}{traversal}",
        program.instances.len()
    )
}

const TRACE_FALLBACK_WGSL: &str = include_str!("wgsl/trace_fallback_wgsl.wgsl");

const TRACE_WGSL: &str = include_str!("wgsl/trace_wgsl.wgsl");

/// Frozen internal volume ABI v1, shared by resource card 7118, scheduler 7119 and duel 7129.
///
/// Bindings (entry-point reflection may omit unused bindings): group 0 is exactly
/// [`trace_module`]'s layout. Source stride 48/alignment 4, instance stride 128/alignment
/// 16, bend stride 32/alignment 16, BVH stride 32/alignment 16, order stride 4,
/// material stride 32/alignment 16; FrameGpu is 512 bytes/alignment 16. Packed file
/// f64 bends are converted to two f32 vec4s per instance in normalized record order;
/// rigid records are zero. Never reconstruct bends from angle/chord/axes or bounds.
/// Group 1: 0 [`VolumeUniform`] (64 bytes/alignment 16); 1 sampled R8Snorm D3 with
/// exactly five mips; 2 read_write storage array<u32> + COPY_SRC staging buffer;
/// 3 sampled RGBA32Float D2 coarse results; 4 RGBA32Float write-only D2 coarse output.
/// Sampled bindings use nonfiltering float sample types. No sampler, filtering,
/// atomics or R8Snorm storage-write feature is required.
///
/// Uniform offsets: origin @0 = (world_min.xyz, isotropic base voxel spacing h);
/// shape @16 = (base dimensions.xyz, revision); encoding @32 = (positive distance
/// range R, nonnegative absolute evaluator/coordinate roundoff allowance E, 0, 0);
/// build @48 = (destination mip, bytes_per_row/4, ready_revision, 0).
/// Dimensions are positive multiples of 16, each <=1024. h and R are finite and
/// positive. World bounds are [origin, origin+h*shape.xyz], enclosing the complete
/// instance BVH, including bent envelopes. Empty scenes use a valid dummy volume.
/// Mip m has shape.xyz>>m cells, spacing h*2^m, and centers at
/// origin+(cell+0.5)*h*2^m, for m=0..4. The host must also fit texture dimensions and
/// staging buffers within negotiated device limits. The volume limit never limits
/// image size: colour, depth and FrameGpu.viewport.xy equal the full render target.
///
/// Base stamping computes a signed conservative proximity to the union of packed
/// instances. Rigid fields divide by a derived Lipschitz bound (including primitive
/// transforms and ellipsoid anisotropy). Bent inverse maps have angular singularities:
/// positive steps are limited by the conservative instance envelope, becoming zero
/// inside it. Exact bend evaluation still supplies signs; exact tracing resolves the
/// envelope. This deliberately trades acceleration for safety near bent geometry.
/// Subtract sqrt(3)*h/2 + E, then encode floor(127*clamp(d/R,-1,1)) as signed i8.
/// -128 is never emitted. Decode = max(byte/127,-1)*R, as R8Snorm specifies.
/// Unsaturated rounding error is in [-R/127,0]; positive saturation only shortens
/// steps. Negative saturation is NOT a signed lower bound: all negative samples
/// are treated as zero traversal clearance, never abs(sample). E must cover numeric
/// evaluation error for the scene's coordinate/parameter range; if unavailable, set
/// ready_revision=0 to use exact fallback. Sampling subtracts an additional R/127+E
/// to cover floating encode/decode error. Each base value bounds its ENTIRE cell.
/// Mips take the minimum signed byte of all eight children, not an average; thus
/// positive clearance bounds the entire parent cell without another footprint term.
/// TextureLoad selects one cell; there is no interpolation or filtering error.
///
/// [`volume_mip_layout`] defines every copy. One invocation owns four adjacent x
/// bytes in one u32, with x+0 in bits 0..7 (little endian). bytesPerRow=align_up(width,
/// 256), rowsPerImage=height, sliceStride=bytesPerRow*height, buffer size=sliceStride*
/// depth. Buffer offset is zero. Last word lanes beyond width are zero. Row padding
/// are not voxels: dispatch excludes padding words, texture copy extent is exactly
/// (width,height,depth), and mip reduction never reads padding. Each dispatch uses
/// a separate immutable uniform snapshot. Dispatch stamp/mip as
/// (ceil(width/32),ceil(height/8),depth); copy before reading that mip for reduction.
/// Dispatch coarse/full as (ceil(target_width/8),ceil(target_height/8),1) using each
/// entry's own target dimensions; all four entries have workgroup size (8,8,1).
/// A full sampled texture view is legal since writes target the separate buffer.
///
/// Coarse texture dimensions are ceil(full_target/8). Texel c covers full pixels
/// [8*c,min(8*(c+1),target)); its cone encloses their complete pixel footprints.
/// Coarse RGBA = (resume world-ray distance, state, revision as f32, 0).
/// State 0=invalid/uninitialized; 1=certified empty prefix [0,resume); 2=ambiguous
/// (resume must be zero). There is NO miss/discard state. Exhaustion returns a safe
/// prefix; a cone candidate always runs exact refinement. Revision is 1..=2^24-1.
/// Coarse and full rays measure t from their own near-plane origin. The cone bounds
/// both origin offsets and normalized direction differences, including orthographic
/// rays; only conventional perspective/orthographic projections with constant
/// near/far homogeneous w over the viewport are accelerated. Other projections use
/// state 2 / exact fallback. Full tracing maps pixel/8, validates state and revision,
/// and otherwise resumes at zero. Mips follow clearance and pixel-cone footprint;
/// cone termination starts exact BVH/program tracing, never accepts a baked hit.
/// Exact gradients, material lookup and projected refined position supply colour
/// and depth to [`composite_module`], preserving the mesh depth test.
///
/// Any accepted program/instance/bend/bounds/encoding change invalidates ready_revision
/// and coarse results BEFORE consumers. Rebuild mip 0, copy, reduce/copy mips 1..4,
/// then publish ready_revision=revision and run coarse before full trace. Revisions
/// must match; clear coarse to zero on resize/wrap. Reject invalid updates atomically
/// before changing revision. Target size/camera changes require a new coarse prepass
/// (clear it first); they need no restamp. Never reuse coarse from a different frame.
/// The shader checks revisions, but ordering and camera invalidation belong to 7119.
pub const VOLUME_ABI_VERSION: u32 = 1;
pub const VOLUME_MIP_LEVELS: u32 = 5;
pub const VOLUME_MAX_AXIS: u32 = 1024;
pub const VOLUME_COARSE_TILE: u32 = 8;

/// Host representation of group 1 binding 0. See [`VOLUME_ABI_VERSION`] for fields.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default)]
pub struct VolumeUniform {
    pub origin: [f32; 4],
    pub shape: [u32; 4],
    pub encoding: [f32; 4],
    pub build: [u32; 4],
}

/// Buffer-to-texture copy layout; offsets are zero and no padding is sampled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VolumeMipLayout {
    pub dimensions: [u32; 3],
    pub bytes_per_row: u32,
    pub rows_per_image: u32,
    pub slice_stride: u64,
    pub buffer_size: u64,
    pub workgroups: [u32; 3],
}

pub fn volume_mip_layout(dimensions: [u32; 3], mip: u32) -> Option<VolumeMipLayout> {
    if mip >= VOLUME_MIP_LEVELS
        || dimensions
            .iter()
            .any(|&d| d == 0 || d > VOLUME_MAX_AXIS || d % 16 != 0)
    {
        return None;
    }
    let dimensions = dimensions.map(|d| d >> mip);
    let [width, height, depth] = dimensions;
    let bytes_per_row = width.div_ceil(256) * 256;
    let slice_stride = u64::from(bytes_per_row) * u64::from(height);
    Some(VolumeMipLayout {
        dimensions,
        bytes_per_row,
        rows_per_image: height,
        slice_stride,
        buffer_size: slice_stride * u64::from(depth),
        workgroups: [width.div_ceil(32), height.div_ceil(8), depth],
    })
}

/// All four volume compute entry points, using [`VOLUME_ABI_VERSION`].
pub fn volume_module(program: &TraceProgram) -> String {
    trace_shader(program, VOLUME_WGSL)
}

const VOLUME_WGSL: &str = include_str!("wgsl/volume_wgsl.wgsl");

/// Fullscreen forward composition: `vs_main` (3 vertices), `fs_main` (HDR + frag_depth).
/// Group 0 binding 0 is FrameGpu; group 1 bindings 0/1 are the trace colour/depth textures.
/// Use RAA's Less depth test with depth writes enabled and load the existing frame attachments.
/// Misses discard, preserving mesh depth; no mesh depth texture is sampled while it is attached.
pub fn composite_module() -> String {
    format!("{TRACE_FRAME_WGSL}{COMPOSITE_WGSL}")
}

const COMPOSITE_WGSL: &str = include_str!("wgsl/composite_wgsl.wgsl");
