
struct TraceFrame {
  view_proj: mat4x4<f32>, view: mat4x4<f32>, proj: mat4x4<f32>,
  inv_proj: mat4x4<f32>, inv_view: mat4x4<f32>,
  camera_pos: vec4<f32>, sun_dir: vec4<f32>, sun_color: vec4<f32>,
  ambient: vec4<f32>, viewport: vec4<f32>, clip: vec4<f32>,
  flags: vec4<u32>, exposure: vec4<f32>, reserved: array<vec4<f32>, 4>,
}
