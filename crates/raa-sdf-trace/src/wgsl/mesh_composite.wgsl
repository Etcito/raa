@group(0) @binding(0) var raster: texture_2d<f32>;
@group(0) @binding(1) var radiance: texture_2d<f32>;
@group(0) @binding(2) var motion: texture_2d<f32>;
@group(0) @binding(3) var<uniform> options: array<vec4<u32>, 2>;

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((vertex << 1u) & 2u), f32(vertex & 2u));
    return vec4<f32>(2.0 * p - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(frag.xy);
    let trace_size = vec2<f32>(textureDimensions(radiance));
    let uv = frag.xy / vec2<f32>(textureDimensions(raster));
    let traced = textureLoad(radiance, vec2<i32>(uv * trace_size), 0);
    if options[0].w != 0u { return vec4<f32>(traced.rgb, 1.0); }
    let base = textureLoad(raster, pixel, 0);
    if traced.a == 0.0 || textureLoad(motion, pixel, 0).w < 0.5 { return base; }
    return vec4<f32>(mix(base.rgb, traced.rgb, 0.92), base.a);
}
