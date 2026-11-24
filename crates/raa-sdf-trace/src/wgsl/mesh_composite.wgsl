@group(0) @binding(0) var raster: texture_2d<f32>;
@group(0) @binding(1) var radiance: texture_2d<f32>;
@group(0) @binding(2) var motion: texture_2d<f32>;
@group(0) @binding(3) var<uniform> options: array<vec4<u32>, 4>;
@group(0) @binding(4) var terminal_samples: texture_2d<f32>;
@group(0) @binding(5) var terminal_surfaces: texture_2d<u32>;
@group(0) @binding(6) var terminal_weights: texture_2d<f32>;
@group(0) @binding(7) var depth: texture_depth_2d;
@group(0) @binding(8) var normal: texture_2d<f32>;
@group(0) @binding(9) var material: texture_2d<f32>;
struct ReflectionFrame {
    view_proj: mat4x4<f32>, view: mat4x4<f32>, proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>, inv_view: mat4x4<f32>, camera_pos: vec4<f32>,
}
@group(0) @binding(10) var<uniform> frame: ReflectionFrame;

struct Environment { sh: array<vec4<f32>, 9>, info: vec4<f32>, gi: vec4<f32> }
struct Volume { levels: array<vec4<f32>, 5>, resolution: f32, intensity: f32, level_count: u32, flags: u32 }
@group(1) @binding(0) var environment_cube: texture_cube<f32>;
@group(1) @binding(1) var<uniform> environment: Environment;
@group(1) @binding(2) var brdf_lut: texture_2d<f32>;
@group(1) @binding(3) var linear_clamp: sampler;
@group(1) @binding(4) var screen_ao: texture_2d<f32>;
@group(1) @binding(5) var screen_gi: texture_2d<f32>;
@group(1) @binding(6) var<uniform> volume: Volume;

fn view_position(uv: vec2<f32>, z: f32) -> vec3<f32> {
    let h = frame.inv_proj * vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), z, 1.0);
    return h.xyz / h.w;
}

fn replace_indirect_specular(pixel: vec2<i32>, uv: vec2<f32>, incident: vec3<f32>) -> vec3<f32> {
    let nr = textureLoad(normal, pixel, 0);
    let n = normalize(nr.xyz);
    let position = view_position(uv, textureLoad(depth, pixel, 0));
    let v = normalize(view_position(uv, 0.0) - position);
    let n_dot_v = clamp(dot(n, v), 0.0, 1.0);
    let srgb = textureLoad(material, pixel, 0).rgb;
    let f0 = select(pow((srgb + 0.055) / 1.055, vec3<f32>(2.4)), srgb / 12.92, srgb <= vec3<f32>(0.04045));
    let roughness = clamp(nr.w, 0.0, 1.0);
    if environment.info.w < 0.5 {
        return incident * mix(f0, vec3<f32>(1.0), pow(1.0 - n_dot_v, 5.0));
    }
    let ao_pixel = min(pixel, vec2<i32>(textureDimensions(screen_ao)) - 1);
    let gi_pixel = min(pixel, vec2<i32>(textureDimensions(screen_gi)) - 1);
    let gi = textureLoad(screen_gi, gi_pixel, 0);
    let ao = min(textureLoad(screen_ao, ao_pixel, 0).r, gi.a);
    let lut_uv = clamp(vec2<f32>(n_dot_v, roughness) * 128.0, vec2<f32>(0.5), vec2<f32>(127.5))
        / vec2<f32>(128.0, 192.0);
    let dfg = textureSampleLevel(brdf_lut, linear_clamp, lut_uv, 0.0).rg;
    let r = raa_ibl_dominant_dir(n, reflect(-v, n), roughness);
    let world_r = (frame.inv_view * vec4<f32>(r, 0.0)).xyz;
    let sky = textureSampleLevel(environment_cube, linear_clamp, world_r, roughness * environment.info.z).rgb;
    let prior = mix(sky * environment.info.x, gi.rgb * volume.intensity / 3.14159265,
        clamp(environment.gi.x, 0.0, 1.0));
    return (incident - prior) * raa_ibl_specular_response(f0, dfg, n_dot_v, ao, roughness);
}

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
    let reflection_class = textureLoad(motion, pixel, 0).w;
    if traced.a == 0.0 || reflection_class < 0.5 { return base; }
    let reflected = max(traced.rgb + terminal_correction(trace_pixel), vec3<f32>(0.0));
    if reflection_class > 1.5 {
        if traced.a < 1.5 { return base; }
        let sample_uv = (vec2<f32>(trace_pixel) + 0.5) / trace_size;
        let sample_pixel = vec2<i32>(sample_uv * vec2<f32>(textureDimensions(normal)));
        let n = textureLoad(normal, pixel, 0).xyz;
        let sample_n = textureLoad(normal, sample_pixel, 0).xyz;
        if dot(n, sample_n) < 0.9 { return base; }
        let delta = replace_indirect_specular(pixel, uv, reflected);
        return vec4<f32>(max(base.rgb + delta, vec3<f32>(0.0)), base.a);
    }
    if traced.a > 1.5 { return base; }
    return vec4<f32>(mix(base.rgb, reflected, 0.92), base.a);
}
