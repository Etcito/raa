// One solid triangle in clip space, generated from the vertex index (no buffers).
// Counter-clockwise, centred on the canvas.

// True when the canvas format is not *Srgb: the shader then encodes linear to sRGB itself.
override SRGB_ENCODE: bool = false;

// Solid orange, linear RGB.
const COLOR: vec3<f32> = vec3<f32>(1.0, 0.35, 0.05);

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 0.5),
        vec2<f32>(-0.5, -0.5),
        vec2<f32>(0.5, -0.5),
    );
    return vec4<f32>(positions[index], 0.0, 1.0);
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let x = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    let low = x * 12.92;
    let high = 1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, x <= vec3<f32>(0.0031308));
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    var rgb = COLOR;
    if SRGB_ENCODE {
        rgb = linear_to_srgb(rgb);
    }
    return vec4<f32>(rgb, 1.0);
}
