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
