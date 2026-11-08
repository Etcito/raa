
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
fn trace_sample_value(start: u32, at: u32, format: f32) -> f32 {
  if (format == 0.0) { return trace_data[start + at]; }
  let half = (bitcast<u32>(trace_data[start + at / 2u]) >> (16u * (at & 1u))) & 65535u;
  let exponent = (half >> 10u) & 31u;
  let mantissa = half & 1023u;
  let sign = (half & 32768u) << 16u;
  if (exponent == 0u) {
    return select(1.0, -1.0, sign != 0u) * f32(mantissa) * 0.000000059604644775390625;
  }
  return bitcast<f32>(sign | ((exponent + 112u) << 23u) | (mantissa << 13u));
}
fn trace_sampled(base: u32, p: vec3<f32>) -> f32 {
  let low = vec3<f32>(trace_data[base+14u], trace_data[base+15u], trace_data[base+16u]);
  let h = trace_data[base+17u];
  let size = vec3<u32>(u32(trace_data[base+18u]), u32(trace_data[base+19u]), u32(trace_data[base+20u]));
  let start = u32(trace_data[base+21u]) * 4u;
  let q = clamp((p-low)/h, vec3<f32>(0.0), vec3<f32>(size-vec3<u32>(1u)));
  let cell = min(vec3<u32>(floor(q)), size-vec3<u32>(2u));
  let t = q-vec3<f32>(cell);
  var value = 0.0;
  for (var k=0u; k<8u; k=k+1u) {
    let bit = vec3<u32>(k&1u, (k>>1u)&1u, k>>2u);
    let w = select(vec3<f32>(1.0)-t,t,bit!=vec3<u32>(0u));
    let at = cell+bit;
    value = value+w.x*w.y*w.z*trace_sample_value(start, at.x+size.x*(at.y+size.y*at.z), trace_data[base+23u]);
  }
  return value+length(p-(low+q*h));
}
fn trace_eval(code_start: u32, code_len: u32, data_start: u32, p: vec3<f32>) -> TraceSdf {
  var distances: array<f32, 64>;
  var materials: array<u32, 64>;
  var depth = 0u;
  for (var i = 0u; i < code_len; i = i + 1u) {
    let word = trace_code[code_start + i];
    let op = word & 0xffu;
    let base = data_start + (word >> 8u);
    if (op >= 1u && op <= 9u) {
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
      var value = trace_primitive(op, local, params);
      if (op == 9u) { value = trace_sampled(base, local); }
      distances[depth] = value * trace_data[base + 12u];
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
