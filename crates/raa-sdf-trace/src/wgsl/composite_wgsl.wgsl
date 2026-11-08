
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
