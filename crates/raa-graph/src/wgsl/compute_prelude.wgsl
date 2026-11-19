struct RgFrame {
  size: vec2<u32>,
  inv_size: vec2<f32>,
  frame_index: u32,
  time: f32,
  view: mat4x4<f32>,
  proj: mat4x4<f32>,
  inv_proj: mat4x4<f32>,
  inv_view: mat4x4<f32>,
  previous_view_proj: mat4x4<f32>,
  history_valid: u32,
}
@group(0) @binding(0) var<uniform> rg_params: RgParams;
@group(0) @binding(1) var<uniform> rg_frame: RgFrame;
@group(0) @binding(2) var rg_linear: sampler;
