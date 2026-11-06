//! Shared WGSL evaluator of the section 67 packed SDF program.
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
/// The caller caps target dimensions to section 67's 1024 limit; dispatch rounds up by 8.
pub fn trace_module(program: &TraceProgram) -> String {
    format!(
        "{PROGRAM_EVAL_WGSL}{TRACE_FRAME_WGSL}\nconst TRACE_COUNT: u32 = {}u;\n{TRACE_WGSL}",
        program.instances.len()
    )
}

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
fn trace_candidate(index: u32, o: vec3<f32>, d: vec3<f32>, limit: f32) -> f32 {
  let i = trace_instances[index];
  let s = trace_sources[i.source];
  var lo = trace_lo(s);
  var hi = trace_hi(s);
  if (i.kind == 1u) {
    let b = trace_bends[index];
    // Same conservative deformation envelope as bvh::instance_bounds.
    let margin = 16.0 * max(abs(radians(b.a.y)), 1.0) *
      (length(hi - lo) + abs(b.a.w) + abs(b.b.x) + length(b.b.yzw) + 1.0);
    lo = lo - vec3<f32>(margin);
    hi = hi + vec3<f32>(margin);
  }
  let local_o = trace_map(i.local[0], i.local[1], i.local[2], o);
  // Keep world-distance t: the inverse-transformed direction must not be normalized.
  let local_d = vec3<f32>(dot(i.local[0].xyz, d), dot(i.local[1].xyz, d), dot(i.local[2].xyz, d));
  let interval = trace_interval(lo, hi, local_o, local_d, limit);
  var t = interval.x;
  for (var step = 0u; step < 128u && t <= interval.y; step = step + 1u) {
    let distance = trace_value(index, o + d * t).d;
    if (distance < max(1e-4, 1e-4 * t)) { return t; }
    t = t + distance;
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
        let t = trace_candidate(index, origin, direction, nearest);
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
  // Actual rounded target size, not the nominal reduction factor; clamp the last row/column.
  let uv = frag.xy / trace_frame.viewport.xy;
  let pixel = vec2<i32>(min(vec2<u32>(uv * vec2<f32>(size)), size - vec2<u32>(1u)));
  let color = textureLoad(trace_color, pixel, 0);
  if (color.a == 0.0) { discard; }
  return TraceOutput(color, textureLoad(trace_depth, pixel, 0).x);
}
"#;
