
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
struct MarchRay { origin: vec3<f32>, direction: vec3<f32> }
fn trace_ray_at(uv: vec2<f32>) -> MarchRay {
  let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let o = trace_unproject(ndc, 0.0);
  return MarchRay(o, normalize(trace_unproject(ndc, 1.0) - o));
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
