
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
    if (op < 1u || op > 9u) { continue; }
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
    if (op == 9u) { primitive = trace_data[b + 22u]; }
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
  let metric = max(trace_sources[i.source].lipschitz, 1.0);
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
      if (abs(trace_value(index, o + d * root).d) < max(1e-5, 1e-6 * root) / metric) { return root; }
    }
    if (abs(distance) < max(1e-5, 1e-6 * t) / metric) {
      for (var refine = 0u; refine < 16u; refine = refine + 1u) {
        // Signed, as in the CPU twin: a hit rounded inside must stop here; stepping by
        // |d| would march through the solid and return the unlit back face.
        let residual = trace_value(index, o + d * t).d;
        if (residual < 1e-7 / metric) { break; }
        let next = t + residual;
        if (next > interval.y) { return limit; }
        if (next <= t) { break; }
        t = next;
      }
      if (abs(trace_value(index, o + d * t).d) < max(1e-5, 1e-6 * t) / metric) { return t; }
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
