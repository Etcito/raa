@group(0) @binding(0) var raster: texture_2d<f32>;
@group(0) @binding(1) var radiance: texture_2d<f32>;
@group(0) @binding(2) var motion: texture_2d<f32>;
@group(0) @binding(3) var<uniform> options: array<vec4<u32>, 4>;
@group(0) @binding(4) var terminal_samples: texture_2d<f32>;
@group(0) @binding(5) var terminal_surfaces: texture_2d<u32>;
@group(0) @binding(6) var terminal_weights: texture_2d<f32>;

fn guide_normal(key: u32) -> vec3<f32> {
    let oct = unpack4x8snorm(key).xy;
    let z = 1.0 - abs(oct.x) - abs(oct.y);
    let signs = select(vec2<f32>(-1.0), vec2<f32>(1.0), oct >= vec2<f32>(0.0));
    let xy = select(oct, (vec2<f32>(1.0) - abs(oct.yx)) * signs, z < 0.0);
    return normalize(vec3<f32>(xy, z));
}

// Filter incident diffuse only, across paths terminating on the same surface.
// Applying the centre BRDF afterwards preserves vertex and material colour.
fn terminal_correction(pixel: vec2<i32>) -> vec3<f32> {
    let centre = textureLoad(terminal_samples, pixel, 0);
    if centre.w == 0.0 { return vec3<f32>(0.0); }
    let guide = textureLoad(terminal_surfaces, pixel, 0);
    let position = bitcast<vec3<f32>>(guide.xyz);
    let normal = guide_normal(guide.w);
    let bounds = vec2<i32>(textureDimensions(terminal_samples));
    let incident = centre.rgb;
    var sum = vec3<f32>(0.0);
    var weights = 0.0;
    for (var y = -2; y <= 2; y++) {
        for (var x = -2; x <= 2; x++) {
            let offset = vec2<i32>(x, y);
            let neighbour = pixel + offset;
            if any(neighbour < vec2<i32>(0)) || any(neighbour >= bounds) { continue; }
            let sample = textureLoad(terminal_samples, neighbour, 0);
            if sample.w != centre.w { continue; }
            let surface = textureLoad(terminal_surfaces, neighbour, 0);
            if surface.w >> 16u != guide.w >> 16u { continue; }
            let delta = bitcast<vec3<f32>>(surface.xyz) - position;
            let separation = length(delta);
            if separation > 1.0 || dot(guide_normal(surface.w), normal) < 0.98
                || abs(dot(delta, normal)) > 0.02 + 0.015 * separation { continue; }

            let w = exp(-0.28 * f32(x * x + y * y)) * (1.0 - 0.5 * separation);
            sum += sample.rgb * w;
            weights += w;
        }
    }
    let weight = textureLoad(terminal_weights, pixel, 0).rgb;
    return (sum / max(weights, 1.0) - incident) * weight;
}

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
    let trace_pixel = vec2<i32>(uv * trace_size);
    let traced = textureLoad(radiance, trace_pixel, 0);
    if options[0].w != 0u || options[2].y != 0u { return vec4<f32>(traced.rgb, 1.0); }
    let base = textureLoad(raster, pixel, 0);
    if traced.a == 0.0 || textureLoad(motion, pixel, 0).w < 0.5 { return base; }
    let reflected = max(traced.rgb + terminal_correction(trace_pixel), vec3<f32>(0.0));
    return vec4<f32>(mix(base.rgb, reflected, 0.92), base.a);
}
