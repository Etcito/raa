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
pub const PROGRAM_EVAL_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> trace_code: array<u32>;
@group(0) @binding(1) var<storage, read> trace_data: array<f32>;
struct TraceSdf { d: f32, material: u32 }
fn trace_map(row0: vec4<f32>, row1: vec4<f32>, row2: vec4<f32>, p: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(dot(row0.xyz, p) + row0.w, dot(row1.xyz, p) + row1.w, dot(row2.xyz, p) + row2.w);
}
fn trace_primitive(op: u32, p: vec3<f32>, d: vec4<f32>) -> f32 {
  if (op == 1u || op == 3u) {
    let q = abs(p) - d.xyz + vec3<f32>(d.w);
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - d.w;
  }
  if (op == 2u) { return length(p) - d.x; }
  if (op == 4u) {
    let qy = p.y - clamp(p.y, -d.x, d.x);
    return sqrt(p.x * p.x + qy * qy + p.z * p.z) - d.y;
  }
  if (op == 5u) {
    let dx = sqrt(p.x * p.x + p.z * p.z) - d.x;
    let dy = abs(p.y) - d.y;
    let m = max(vec2<f32>(dx, dy), vec2<f32>(0.0));
    return min(max(dx, dy), 0.0) + length(m);
  }
  if (op == 6u) {
    let h = d.x;
    let rb = d.y;
    let rt = d.z;
    let qx = sqrt(p.x * p.x + p.z * p.z);
    let kx = rt - rb;
    let ky = 2.0 * h;
    let ax = qx - min(qx, select(rt, rb, p.y < 0.0));
    let ay = abs(p.y) - h;
    let t = clamp(((rt - qx) * kx + (h - p.y) * ky) / (kx * kx + ky * ky), 0.0, 1.0);
    let bx = qx - rt + kx * t;
    let by = p.y - h + ky * t;
    let s = select(1.0, -1.0, bx < 0.0 && ay < 0.0);
    return s * sqrt(min(ax * ax + ay * ay, bx * bx + by * by));
  }
  if (op == 7u) {
    let q = sqrt(p.x * p.x + p.z * p.z) - d.x;
    return sqrt(q * q + p.y * p.y) - d.y;
  }
  if (op == 8u) {
    let a = p / d.xyz;
    let k0 = length(a);
    if (k0 < 1.0) {
      return (k0 - 1.0) * min(d.x, min(d.y, d.z));
    }
    let k1 = length(a / d.xyz);
    return k0 * (k0 - 1.0) / k1;
  }
  return 1e30;
}
fn trace_combine(op: u32, a: f32, b: f32, k: f32) -> f32 {
  if (op == 16u) {
    if (!(k > 0.0)) { return min(a, b); }
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    if (h >= 1.0) { return a; }
    if (h <= 0.0) { return b; }
    return b + (a - b) * h - k * h * (1.0 - h);
  }
  if (op == 17u) {
    if (!(k > 0.0)) { return max(a, -b); }
    let h = clamp(0.5 - 0.5 * (b + a) / k, 0.0, 1.0);
    if (h >= 1.0) { return -b; }
    if (h <= 0.0) { return a; }
    return a + (-b - a) * h + k * h * (1.0 - h);
  }
  if (!(k > 0.0)) { return max(a, b); }
  let h = clamp(0.5 - 0.5 * (b - a) / k, 0.0, 1.0);
  if (h >= 1.0) { return a; }
  if (h <= 0.0) { return b; }
  return b + (a - b) * h + k * h * (1.0 - h);
}
fn trace_eval(code_start: u32, code_len: u32, data_start: u32, p: vec3<f32>) -> TraceSdf {
  var distances: array<f32, 64>;
  var materials: array<u32, 64>;
  var depth = 0u;
  for (var i = 0u; i < code_len; i = i + 1u) {
    let word = trace_code[code_start + i];
    let op = word & 0xffu;
    let base = data_start + (word >> 8u);
    if (op >= 1u && op <= 8u) {
      let local = trace_map(
        vec4<f32>(trace_data[base], trace_data[base + 1u], trace_data[base + 2u], trace_data[base + 3u]),
        vec4<f32>(trace_data[base + 4u], trace_data[base + 5u], trace_data[base + 6u], trace_data[base + 7u]),
        vec4<f32>(trace_data[base + 8u], trace_data[base + 9u], trace_data[base + 10u], trace_data[base + 11u]),
        p,
      );
      var params = vec4<f32>(0.0);
      params.x = trace_data[base + 14u];
      if (op != 2u) { params.y = trace_data[base + 15u]; }
      if (op == 1u || op == 3u || op == 6u || op == 8u) { params.z = trace_data[base + 16u]; }
      if (op == 1u || op == 3u) { params.w = trace_data[base + 17u]; }
      if (depth >= 64u) { return TraceSdf(1e30, 0u); }
      distances[depth] = trace_primitive(op, local, params) * trace_data[base + 12u];
      materials[depth] = u32(trace_data[base + 13u]);
      depth = depth + 1u;
    } else if (depth >= 2u) {
      let a = distances[depth - 2u];
      let b = distances[depth - 1u];
      var m = materials[depth - 2u];
      if (op == 16u && a > b) { m = materials[depth - 1u]; }
      if (op == 18u && a < b) { m = materials[depth - 1u]; }
      distances[depth - 2u] = trace_combine(op, a, b, trace_data[base]);
      materials[depth - 2u] = m;
      depth = depth - 1u;
    }
  }
  if (depth == 0u) { return TraceSdf(1e30, 0u); }
  return TraceSdf(distances[depth - 1u], materials[depth - 1u]);
}
fn trace_rest_point(bend0: vec4<f32>, bend1: vec4<f32>, p: vec3<f32>) -> vec3<f32> {
  let angle = radians(bend0.y);
  let te = abs(angle);
  if (te < 1e-6) { return p; }
  let sgn = select(1.0, -1.0, bend0.y < 0.0);
  let a = u32(bend0.x);
  let b = (a + 1u) % 3u;
  let c = (a + 2u) % 3u;
  let s = bend0.w;
  let radius = (bend1.x - s) / te;
  let dir = radians(bend0.z);
  let cd = cos(dir);
  let sd = sin(dir);
  let ch = cos(te * 0.5);
  let sh = sin(te * 0.5);
  let center = bend1.yzw;
  var u = array<f32, 3>(p.x - center.x, p.y - center.y, p.z - center.z);
  let t = u[a];
  let y = sgn * (cd * u[b] + sd * u[c]);
  let z = -sd * u[b] + cd * u[c];
  let vx = t - s;
  let vy = y - radius;
  let phi = te * 0.5 + atan2(vx * ch + vy * sh, vx * sh - vy * ch);
  let pc = clamp(phi, 0.0, te);
  let cp = cos(pc);
  let sp = sin(pc);
  let yr = sgn * (radius - (vx * sp - vy * cp));
  u[a] = s + pc * radius + vx * cp + vy * sp;
  u[b] = cd * yr - sd * z;
  u[c] = sd * yr + cd * z;
  return vec3<f32>(u[0] + center.x, u[1] + center.y, u[2] + center.z);
}
fn trace_instance(
  code_start: u32, code_len: u32, data_start: u32,
  row0: vec4<f32>, row1: vec4<f32>, row2: vec4<f32>,
  dist_scale: f32, deform_kind: u32,
  bend0: vec4<f32>, bend1: vec4<f32>,
  bend_angle: f32, bend_length: f32, extent: f32,
  p: vec3<f32>,
) -> TraceSdf {
  let local = trace_map(row0, row1, row2, p);
  var rest = local;
  var factor = 1.0;
  if (deform_kind == 1u) {
    rest = trace_rest_point(bend0, bend1, local);
    factor = max(1.0, 1.0 + abs(bend_angle) * extent / max(bend_length, 1e-30));
  }
  let value = trace_eval(code_start, code_len, data_start, rest);
  return TraceSdf(value.d * dist_scale / factor, value.material);
}
fn trace_material(override_material: i32, value: TraceSdf) -> u32 {
  if (override_material >= 0) { return u32(override_material); }
  return value.material;
}
"#;

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
const TRACE_FRAME_WGSL: &str = r#"
struct TraceFrame {
  view_proj: mat4x4<f32>, view: mat4x4<f32>, proj: mat4x4<f32>,
  inv_proj: mat4x4<f32>, inv_view: mat4x4<f32>,
  camera_pos: vec4<f32>, sun_dir: vec4<f32>, sun_color: vec4<f32>,
  ambient: vec4<f32>, viewport: vec4<f32>, clip: vec4<f32>,
  flags: vec4<u32>, exposure: vec4<f32>, reserved: array<vec4<f32>, 4>,
}
"#;

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

const TRACE_FALLBACK_WGSL: &str = r#"
fn trace_start(uv: vec2<f32>, o: vec3<f32>, d: vec3<f32>, limit: f32) -> f32 { return 0.0; }
"#;

const TRACE_WGSL: &str = r#"
struct TraceSource {
  code_start: u32, code_len: u32, data_start: u32, data_len: u32,
  lo: array<f32, 3>, hi: array<f32, 3>, lipschitz: f32, flags: u32,
}
struct TraceInstance {
  world: array<vec4<f32>, 3>, local: array<vec4<f32>, 3>,
  scale: f32, source: u32, material: i32, key: u32,
  kind: u32, angle: f32, chord: f32, axes: u32,
}
struct TraceBend { a: vec4<f32>, b: vec4<f32> }
struct TraceNode { lo: vec3<f32>, left: u32, hi: vec3<f32>, count: u32 }
struct TraceMaterial { color: vec4<f32>, rest: vec4<f32> }
@group(0) @binding(2) var<storage, read> trace_sources: array<TraceSource>;
@group(0) @binding(3) var<storage, read> trace_instances: array<TraceInstance>;
@group(0) @binding(4) var<storage, read> trace_bends: array<TraceBend>;
@group(0) @binding(5) var<storage, read> trace_nodes: array<TraceNode>;
@group(0) @binding(6) var<storage, read> trace_order: array<u32>;
@group(0) @binding(7) var<storage, read> trace_materials: array<TraceMaterial>;
@group(0) @binding(8) var<uniform> trace_frame: TraceFrame;
@group(0) @binding(9) var trace_color: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var trace_depth: texture_storage_2d<r32float, write>;
fn trace_lo(s: TraceSource) -> vec3<f32> { return vec3<f32>(s.lo[0], s.lo[1], s.lo[2]); }
fn trace_hi(s: TraceSource) -> vec3<f32> { return vec3<f32>(s.hi[0], s.hi[1], s.hi[2]); }
fn trace_value(index: u32, p: vec3<f32>) -> TraceSdf {
  let i = trace_instances[index];
  let s = trace_sources[i.source];
  let b = trace_bends[index];
  let value = trace_instance(s.code_start, s.code_len, s.data_start,
    i.local[0], i.local[1], i.local[2], i.scale, i.kind, b.a, b.b,
    i.angle, i.chord, length(trace_hi(s) - trace_lo(s)), p);
  return TraceSdf(value.d / max(s.lipschitz, 1.0), trace_material(i.material, value));
}
fn trace_interval(lo: vec3<f32>, hi: vec3<f32>, o: vec3<f32>, d: vec3<f32>, limit: f32) -> vec2<f32> {
  var near = 0.0;
  var far = limit;
  for (var a = 0u; a < 3u; a = a + 1u) {
    if (abs(d[a]) < 1e-30) {
      if (o[a] < lo[a] || o[a] > hi[a]) { return vec2<f32>(1.0, -1.0); }
    } else {
      let x = (lo[a] - o[a]) / d[a];
      let y = (hi[a] - o[a]) / d[a];
      near = max(near, min(x, y));
      far = min(far, max(x, y));
    }
  }
  return vec2<f32>(near, far);
}
// Smooth polynomial combinations preserve the maximum input Lipschitz bound.
// Frobenius norms bound affine operator norms, including nonuniform and mirrored scales.
fn trace_lipschitz(index: u32) -> f32 {
  let i = trace_instances[index];
  let s = trace_sources[i.source];
  var bound = 1.0;
  for (var k = 0u; k < s.code_len; k = k + 1u) {
    let word = trace_code[s.code_start + k];
    let op = word & 255u;
    if (op < 1u || op > 8u) { continue; }
    let b = s.data_start + (word >> 8u);
    var norm2 = 0.0;
    for (var row = 0u; row < 3u; row = row + 1u) {
      for (var col = 0u; col < 3u; col = col + 1u) {
        let v = trace_data[b + row * 4u + col];
        norm2 = norm2 + v * v;
      }
    }
    var primitive = 1.0;
    if (op == 8u) {
      let r = vec3<f32>(trace_data[b + 14u], trace_data[b + 15u], trace_data[b + 16u]);
      let ratio = max(r.x, max(r.y, r.z)) / min(r.x, min(r.y, r.z));
      primitive = 2.0 * ratio + ratio * ratio;
    }
    bound = max(bound, primitive * trace_data[b + 12u] * sqrt(norm2));
  }
  let norm = sqrt(dot(i.local[0].xyz, i.local[0].xyz) +
    dot(i.local[1].xyz, i.local[1].xyz) + dot(i.local[2].xyz, i.local[2].xyz));
  return max(1.0, bound * i.scale * norm / max(s.lipschitz, 1.0));
}
struct TraceBounds { lo: vec3<f32>, hi: vec3<f32> }
fn trace_bounds(index: u32) -> TraceBounds {
  let i = trace_instances[index];
  let s = trace_sources[i.source];
  var bounds = TraceBounds(trace_lo(s), trace_hi(s));
  if (i.kind == 1u) {
    let b = trace_bends[index];
    // Keep this identical to bvh::instance_bounds; normalized record is authoritative.
    let margin = 16.0 * max(abs(radians(b.a.y)), 1.0) *
      (length(bounds.hi - bounds.lo) + abs(b.a.w) + abs(b.b.x) + length(b.b.yzw) + 1.0);
    bounds.lo = bounds.lo - vec3<f32>(margin);
    bounds.hi = bounds.hi + vec3<f32>(margin);
  }
  return bounds;
}
fn trace_candidate(index: u32, o: vec3<f32>, d: vec3<f32>, start: f32, limit: f32) -> f32 {
  let i = trace_instances[index];
  let bounds = trace_bounds(index);
  let local_o = trace_map(i.local[0], i.local[1], i.local[2], o);
  // Keep world-distance t: the inverse-transformed direction must not be normalized.
  let local_d = vec3<f32>(dot(i.local[0].xyz, d), dot(i.local[1].xyz, d), dot(i.local[2].xyz, d));
  let interval = trace_interval(bounds.lo, bounds.hi, local_o, local_d, limit);
  var t = max(interval.x, start);
  var previous_t = t;
  var previous_d = trace_value(index, o + d * t).d;
  for (var step = 0u; step < 256u && t <= interval.y; step = step + 1u) {
    let distance = trace_value(index, o + d * t).d;
    if (distance * previous_d < 0.0) {
      // A bracket is refined on the exact program, including rays starting inside.
      var lo = previous_t;
      var hi = t;
      for (var refine = 0u; refine < 24u; refine = refine + 1u) {
        let mid = (lo + hi) * 0.5;
        let value = trace_value(index, o + d * mid).d;
        if (value * previous_d > 0.0) { lo = mid; } else { hi = mid; }
      }
      let root = (lo + hi) * 0.5;
      // A bend branch discontinuity must not be mistaken for a root.
      if (abs(trace_value(index, o + d * root).d) < max(1e-5, 1e-6 * root)) { return root; }
    }
    if (abs(distance) < max(1e-5, 1e-6 * t)) {
      for (var refine = 0u; refine < 16u; refine = refine + 1u) {
        let residual = abs(trace_value(index, o + d * t).d);
        if (residual < 1e-7) { break; }
        let next = t + residual;
        if (next > interval.y) { return limit; }
        if (next <= t) { break; }
        t = next;
      }
      if (abs(trace_value(index, o + d * t).d) < max(1e-5, 1e-6 * t)) { return t; }
    }
    previous_t = t;
    previous_d = trace_value(index, o + d * t).d;
    // Preserve the CPU twin distance estimator; the voxel footprint uses a separate bound.
    let next = t + abs(previous_d);
    if (!(next > t)) { break; }
    t = next;
  }
  return limit;
}
fn trace_unproject(ndc: vec2<f32>, z: f32) -> vec3<f32> {
  let p = trace_frame.inv_view * trace_frame.inv_proj * vec4<f32>(ndc, z, 1.0);
  return p.xyz / p.w;
}
@compute @workgroup_size(8, 8)
fn sdf_trace(@builtin(global_invocation_id) id: vec3<u32>) {
  let size = textureDimensions(trace_color);
  if (any(id.xy >= size)) { return; }
  let pixel = vec2<i32>(id.xy);
  textureStore(trace_color, pixel, vec4<f32>(0.0));
  textureStore(trace_depth, pixel, vec4<f32>(1.0));
  if (TRACE_COUNT == 0u) { return; }
  let uv = (vec2<f32>(id.xy) + vec2<f32>(0.5)) / vec2<f32>(size);
  let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let origin = trace_unproject(ndc, 0.0);
  let far = trace_unproject(ndc, 1.0);
  let direction = normalize(far - origin);
  var nearest = length(far - origin);
  let start = trace_start(uv, origin, direction, nearest);
  var hit = TRACE_COUNT;
  var stack: array<u32, 32>;
  stack[0] = 0u;
  var depth = 1u;
  loop {
    if (depth == 0u) { break; }
    depth = depth - 1u;
    let node = trace_nodes[stack[depth]];
    let span = trace_interval(node.lo, node.hi, origin, direction, nearest);
    if (span.x > span.y) { continue; }
    if (node.count == 0u) {
      stack[depth] = node.left;
      stack[depth + 1u] = node.left + 1u;
      depth = depth + 2u;
    } else {
      for (var k = 0u; k < node.count; k = k + 1u) {
        let index = trace_order[node.left + k];
        let t = trace_candidate(index, origin, direction, start, nearest);
        if (t < nearest) {
          nearest = t;
          hit = index;
        }
      }
    }
  }
  if (hit == TRACE_COUNT) { return; }
  let p = origin + direction * nearest;
  let e = 1e-3;
  let x = vec3<f32>(e, 0.0, 0.0);
  let y = vec3<f32>(0.0, e, 0.0);
  let z = vec3<f32>(0.0, 0.0, e);
  let gradient = vec3<f32>(trace_value(hit, p + x).d - trace_value(hit, p - x).d,
    trace_value(hit, p + y).d - trace_value(hit, p - y).d,
    trace_value(hit, p + z).d - trace_value(hit, p - z).d);
  var normal = vec3<f32>(0.0, 1.0, 0.0);
  if (length(gradient) > 1e-12) { normal = normalize(gradient); }
  let base = trace_materials[trace_value(hit, p).material].color.rgb;
  let light = trace_frame.ambient.rgb + trace_frame.sun_color.rgb * max(dot(normal, -trace_frame.sun_dir.xyz), 0.0);
  let clip = trace_frame.view_proj * vec4<f32>(p, 1.0);
  textureStore(trace_color, pixel, vec4<f32>(base * light, 1.0));
  textureStore(trace_depth, pixel, vec4<f32>(clamp(clip.z / clip.w, 0.0, 1.0)));
}
"#;

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

const VOLUME_WGSL: &str = r#"
struct TraceVolume {
  origin: vec4<f32>, shape: vec4<u32>, encoding: vec4<f32>, build: vec4<u32>,
}
@group(1) @binding(0) var<uniform> trace_volume: TraceVolume;
@group(1) @binding(1) var trace_field: texture_3d<f32>;
@group(1) @binding(2) var<storage, read_write> trace_voxels: array<u32>;
@group(1) @binding(3) var trace_coarse: texture_2d<f32>;
@group(1) @binding(4) var trace_coarse_out: texture_storage_2d<rgba32float, write>;
fn trace_world_distance(p: vec3<f32>, range: f32) -> f32 {
  var best = range;
  if (TRACE_COUNT == 0u) { return best; }
  var stack: array<u32, 32>;
  stack[0] = 0u;
  var depth = 1u;
  loop {
    if (depth == 0u) { break; }
    depth = depth - 1u;
    let node = trace_nodes[stack[depth]];
    let outside = max(max(node.lo - p, p - node.hi), vec3<f32>(0.0));
    if (best >= 0.0 && length(outside) > best) { continue; }
    if (node.count == 0u) {
      stack[depth] = node.left;
      stack[depth + 1u] = node.left + 1u;
      depth = depth + 2u;
    } else {
      for (var k = 0u; k < node.count; k = k + 1u) {
        let index = trace_order[node.left + k];
        let i = trace_instances[index];
        var distance = trace_value(index, p).d;
        if (i.kind == 1u) {
          let bounds = trace_bounds(index);
          let local = trace_map(i.local[0], i.local[1], i.local[2], p);
          let outside = max(max(bounds.lo - local, local - bounds.hi), vec3<f32>(0.0));
          distance = min(distance, length(outside) * i.scale);
        } else {
          distance = distance / trace_lipschitz(index);
        }
        best = min(best, distance);
      }
    }
  }
  return best;
}
fn trace_write_voxels(id: vec3<u32>, reduce: bool) {
  let level = trace_volume.build.x;
  let size = trace_volume.shape.xyz >> vec3<u32>(level);
  if (id.x * 4u >= size.x || id.y >= size.y || id.z >= size.z) { return; }
  let range = trace_volume.encoding.x;
  var packed = 0u;
  for (var b = 0u; b < 4u; b = b + 1u) {
    let cell = vec3<u32>(id.x * 4u + b, id.y, id.z);
    if (cell.x >= size.x) { continue; }
    var distance = range;
    if (reduce) {
      for (var c = 0u; c < 8u; c = c + 1u) {
        let child = cell * 2u + vec3<u32>(c & 1u, (c >> 1u) & 1u, c >> 2u);
        distance = min(distance, textureLoad(trace_field, vec3<i32>(child), i32(level) - 1).x * range);
      }
    } else {
      let p = trace_volume.origin.xyz + (vec3<f32>(cell) + vec3<f32>(0.5)) * trace_volume.origin.w;
      distance = trace_world_distance(p, range) - sqrt(3.0) * trace_volume.origin.w * 0.5 - trace_volume.encoding.y;
    }
    let normalized = clamp(distance / range, -1.0, 1.0) * 127.0;
    let code = select(floor(normalized), round(normalized), reduce);
    let byte = bitcast<u32>(i32(code)) & 255u;
    packed = packed | (byte << (b * 8u));
  }
  trace_voxels[(id.z * size.y + id.y) * trace_volume.build.y + id.x] = packed;
}
@compute @workgroup_size(8, 8)
fn sdf_stamp(@builtin(global_invocation_id) id: vec3<u32>) { trace_write_voxels(id, false); }
@compute @workgroup_size(8, 8)
fn sdf_mip(@builtin(global_invocation_id) id: vec3<u32>) { trace_write_voxels(id, true); }
fn trace_volume_distance(p: vec3<f32>, level: u32) -> f32 {
  let local = (p - trace_volume.origin.xyz) / trace_volume.origin.w;
  let size = vec3<f32>(trace_volume.shape.xyz);
  if (any(local < vec3<f32>(0.0)) || any(local >= size)) {
    return max(0.0, length(max(max(-local, local - size), vec3<f32>(0.0))) * trace_volume.origin.w
      - trace_volume.encoding.y);
  }
  let cell = min(vec3<u32>(local) >> vec3<u32>(level), textureDimensions(trace_field, level) - vec3<u32>(1u));
  return max(0.0, textureLoad(trace_field, vec3<i32>(cell), i32(level)).x * trace_volume.encoding.x
    - trace_volume.encoding.x / 127.0 - trace_volume.encoding.y);
}
struct TraceRay { origin: vec3<f32>, direction: vec3<f32> }
fn trace_ray_at(uv: vec2<f32>) -> TraceRay {
  let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let o = trace_unproject(ndc, 0.0);
  return TraceRay(o, normalize(trace_unproject(ndc, 1.0) - o));
}
fn trace_volume_ready() -> bool {
  return trace_volume.shape.w > 0u && trace_volume.shape.w < 16777216u &&
    trace_volume.build.z == trace_volume.shape.w &&
    all(trace_volume.shape.xyz == textureDimensions(trace_field, 0)) &&
    textureNumLevels(trace_field) == 5u &&
    trace_volume.origin.w > 0.0 && trace_volume.origin.w < 1e30 &&
    trace_volume.encoding.x > 0.0 && trace_volume.encoding.x < 1e30 &&
    trace_volume.encoding.y >= 0.0 && trace_volume.encoding.y < 1e30;
}
fn trace_cone(uv: vec2<f32>, half_uv: vec2<f32>) -> vec2<f32> {
  // Constant homogeneous w makes near and far world positions affine in uv.
  // General projective cameras cannot use the corner proof: fall back exactly.
  if (trace_frame.inv_proj[0].w != 0.0 || trace_frame.inv_proj[1].w != 0.0 ||
      trace_frame.inv_view[0].w != 0.0 || trace_frame.inv_view[1].w != 0.0 ||
      trace_frame.inv_view[2].w != 0.0) { return vec2<f32>(-1.0); }
  let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let origin = trace_unproject(ndc, 0.0);
  let delta = trace_unproject(ndc, 1.0) - origin;
  let direction = normalize(delta);
  var origin_radius = 0.0;
  var delta_radius = 0.0;
  var projection = length(delta);
  for (var c = 0u; c < 4u; c = c + 1u) {
    let sign = vec2<f32>(f32(c & 1u), f32(c >> 1u)) * 2.0 - vec2<f32>(1.0);
    let corner_uv = uv + sign * half_uv;
    let corner_ndc = vec2<f32>(corner_uv.x * 2.0 - 1.0, 1.0 - corner_uv.y * 2.0);
    let corner_origin = trace_unproject(corner_ndc, 0.0);
    let corner_delta = trace_unproject(corner_ndc, 1.0) - corner_origin;
    origin_radius = max(origin_radius, length(corner_origin - origin));
    delta_radius = max(delta_radius, length(corner_delta - delta));
    projection = min(projection, dot(corner_delta, direction));
  }
  if (!(projection > 0.0)) { return vec2<f32>(-1.0); }
  // ||normalize(v)-normalize(w)|| <= 2||v-w||/min(|v|,|w|).
  // Corner extrema bound the affine rectangle, including every interior pixel.
  return vec2<f32>(origin_radius, min(2.0, 2.0 * delta_radius / projection));
}
fn trace_volume_walk(o: vec3<f32>, d: vec3<f32>, limit: f32, cone: vec2<f32>, start: f32) -> f32 {
  var t = min(start, limit);
  var distance = trace_volume.origin.w;
  for (var step = 0u; step < 128u && t < limit; step = step + 1u) {
    let radius = cone.x + cone.y * t;
    let level = u32(clamp(floor(log2(max(max(radius, distance) / trace_volume.origin.w, 1.0))), 0.0, 4.0));
    distance = trace_volume_distance(o + d * t, level);
    if (distance <= radius + 1e-5) { break; }
    // The cone expands while advancing; never step across a potential surface.
    t = min(limit, t + (distance - radius) / (1.0 + cone.y));
  }
  // Exhaustion retains this potential interval for exact BVH tracing, never a miss.
  return t;
}
@compute @workgroup_size(8, 8)
fn sdf_coarse(@builtin(global_invocation_id) id: vec3<u32>) {
  if (any(id.xy >= textureDimensions(trace_coarse_out))) { return; }
  let pixel = vec2<i32>(id.xy);
  textureStore(trace_coarse_out, pixel, vec4<f32>(0.0));
  if (!trace_volume_ready()) { return; }
  let lo = vec2<f32>(id.xy * 8u) / trace_frame.viewport.xy;
  let hi = min(vec2<f32>((id.xy + vec2<u32>(1u)) * 8u) / trace_frame.viewport.xy, vec2<f32>(1.0));
  let uv = (lo + hi) * 0.5;
  let cone = trace_cone(uv, (hi - lo) * 0.5);
  if (cone.x < 0.0) {
    textureStore(trace_coarse_out, pixel, vec4<f32>(0.0, 2.0, f32(trace_volume.shape.w), 0.0));
    return;
  }
  let ray = trace_ray_at(uv);
  let far = trace_unproject(vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0), 1.0);
  let start = trace_volume_walk(ray.origin, ray.direction, length(far - ray.origin), cone, 0.0);
  textureStore(trace_coarse_out, pixel, vec4<f32>(start, 1.0, f32(trace_volume.shape.w), 0.0));
}
fn trace_start(uv: vec2<f32>, o: vec3<f32>, d: vec3<f32>, limit: f32) -> f32 {
  if (!trace_volume_ready()) { return 0.0; }
  let cone = trace_cone(uv, vec2<f32>(0.5) / trace_frame.viewport.xy);
  if (cone.x < 0.0) { return 0.0; }
  let cell = min(vec2<u32>(uv * trace_frame.viewport.xy) / 8u, textureDimensions(trace_coarse) - vec2<u32>(1u));
  let coarse = textureLoad(trace_coarse, vec2<i32>(cell), 0);
  var start = 0.0;
  if (coarse.y == 1.0 && coarse.z == f32(trace_volume.shape.w) && coarse.x >= 0.0 && coarse.x < 1e30) {
    start = min(coarse.x, limit);
  }
  return trace_volume_walk(o, d, limit, cone, start);
}

"#;

/// Fullscreen forward composition: `vs_main` (3 vertices), `fs_main` (HDR + frag_depth).
/// Group 0 binding 0 is FrameGpu; group 1 bindings 0/1 are the trace colour/depth textures.
/// Use RAA's Less depth test with depth writes enabled and load the existing frame attachments.
/// Misses discard, preserving mesh depth; no mesh depth texture is sampled while it is attached.
pub fn composite_module() -> String {
    format!("{TRACE_FRAME_WGSL}{COMPOSITE_WGSL}")
}

const COMPOSITE_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> trace_frame: TraceFrame;
@group(1) @binding(0) var trace_color: texture_2d<f32>;
@group(1) @binding(1) var trace_depth: texture_2d<f32>;
struct TraceOutput {
  @location(0) color: vec4<f32>,
  @builtin(frag_depth) depth: f32,
}
@vertex
fn vs_main(@builtin(vertex_index) vertex: u32) -> @builtin(position) vec4<f32> {
  let xy = vec2<f32>(f32((vertex << 1u) & 2u), f32(vertex & 2u)) * 2.0 - vec2<f32>(1.0);
  return vec4<f32>(xy, 0.0, 1.0);
}
@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> TraceOutput {
  let size = textureDimensions(trace_color);
  // Colour/depth and viewport have the same full target size; clamp edge coordinates.
  let uv = frag.xy / trace_frame.viewport.xy;
  let pixel = vec2<i32>(min(vec2<u32>(uv * vec2<f32>(size)), size - vec2<u32>(1u)));
  let color = textureLoad(trace_color, pixel, 0);
  if (color.a == 0.0) { discard; }
  return TraceOutput(color, textureLoad(trace_depth, pixel, 0).x);
}
"#;
