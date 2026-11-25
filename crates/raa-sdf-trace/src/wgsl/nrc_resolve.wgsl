// Neural radiance cache resolve: depth- and normal-aware upsampling of the reduced-resolution
// cache signal to the frame, then temporal accumulation by reprojection. Output rgb = R
// (indirect irradiance / pi), alpha = view depth (0 where there is no surface).

struct Frame {
    view_proj: mat4x4<f32>, view: mat4x4<f32>, proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>, inv_view: mat4x4<f32>,
    camera_pos: vec4<f32>, sun_dir: vec4<f32>, sun_color: vec4<f32>,
    ambient: vec4<f32>, viewport: vec4<f32>, clip: vec4<f32>, flags: vec4<u32>,
    exposure: vec4<f32>, reserved: array<vec4<f32>, 4>,
    previous_view_proj: mat4x4<f32>, unjittered_view_proj: mat4x4<f32>,
    jitter: vec4<f32>, light_info: vec4<u32>, temporal: vec4<u32>,
}
struct NrcResolve {
    // divisor, history valid, unused, unused
    counts: vec4<u32>,
    // weight of the current frame, intensity, unused, unused
    tuning: vec4<f32>,
}
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> nrc_resolve_params: NrcResolve;
@group(0) @binding(2) var depth: texture_depth_2d;
@group(0) @binding(3) var normals: texture_2d<f32>;
@group(0) @binding(4) var nrc_noisy: texture_2d<f32>;
@group(0) @binding(5) var nrc_history: texture_2d<f32>;
@group(0) @binding(6) var nrc_output: texture_storage_2d<rgba16float, write>;

fn nrc_view_depth(uv: vec2<f32>, z: f32) -> f32 {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, z, 1.0);
    let p = frame.inv_proj * ndc;
    return abs(p.z / p.w);
}

fn nrc_world(uv: vec2<f32>, z: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, z, 1.0);
    let p = frame.inv_view * frame.inv_proj * ndc;
    return p.xyz / p.w;
}

@compute @workgroup_size(8, 8)
fn nrc_resolve(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = vec2<i32>(textureDimensions(depth));
    let pixel = vec2<i32>(id.xy);
    if any(pixel >= dims) { return; }
    let z = textureLoad(depth, pixel, 0);
    let normal = textureLoad(normals, pixel, 0).xyz;
    if z >= 1.0 || dot(normal, normal) <= 0.0 {
        textureStore(nrc_output, pixel, vec4<f32>(0.0));
        return;
    }
    let uv = (vec2<f32>(pixel) + 0.5) / vec2<f32>(dims);
    let view_depth = nrc_view_depth(uv, z);
    let n = normalize(normal);

    // Joint bilateral 2x2 upsampling. Each reduced texel stands for the full-resolution pixel
    // at its block centre.
    let divisor = i32(nrc_resolve_params.counts.x);
    let low = vec2<i32>(textureDimensions(nrc_noisy));
    let at = (vec2<f32>(pixel) + 0.5) / f32(divisor) - 0.5;
    let base = vec2<i32>(floor(at));
    let f = at - vec2<f32>(base);
    var sum = vec3<f32>(0.0);
    var total = 0.0;
    var nearest = vec3<f32>(0.0);
    var nearest_weight = -1.0;
    for (var k = 0; k < 4; k++) {
        let o = vec2<i32>(k & 1, k >> 1);
        let q = clamp(base + o, vec2<i32>(0), low - 1);
        let sample = textureLoad(nrc_noisy, q, 0);
        if sample.a < 0.0 { continue; }
        let bilinear = mix(1.0 - f.x, f.x, f32(o.x)) * mix(1.0 - f.y, f.y, f32(o.y));
        var w = bilinear + 1e-3;
        if divisor > 1 {
            let source = min(q * divisor + divisor / 2, dims - 1);
            let source_depth = nrc_view_depth((vec2<f32>(source) + 0.5) / vec2<f32>(dims),
                textureLoad(depth, source, 0));
            let source_normal = textureLoad(normals, source, 0).xyz;
            let facing = max(dot(n, normalize(source_normal + vec3<f32>(1e-6))), 0.0);
            let facing2 = facing * facing;
            w *= exp(-abs(source_depth - view_depth) / (0.02 * view_depth + 0.01))
                * facing2 * facing2 * facing2 * facing2;
        }
        if bilinear > nearest_weight { nearest_weight = bilinear; nearest = sample.rgb; }
        sum += w * sample.rgb;
        total += w;
    }
    var current = nearest;
    if total > 1e-5 { current = sum / total; }

    // Temporal accumulation: reproject the world point, keep history taps at the same depth.
    var value = current;
    if nrc_resolve_params.counts.y != 0u && frame.proj[3].w == 0.0 {
        let world = nrc_world(uv, z);
        let clip = frame.previous_view_proj * vec4<f32>(world, 1.0);
        if clip.w > 1e-4 {
            let previous_uv = vec2<f32>(clip.x / clip.w * 0.5 + 0.5, 0.5 - clip.y / clip.w * 0.5);
            if all(previous_uv > vec2<f32>(0.0)) && all(previous_uv < vec2<f32>(1.0)) {
                let h = previous_uv * vec2<f32>(dims) - 0.5;
                let h0 = vec2<i32>(floor(h));
                let hf = h - vec2<f32>(h0);
                var history = vec3<f32>(0.0);
                var history_weight = 0.0;
                for (var k = 0; k < 4; k++) {
                    let o = vec2<i32>(k & 1, k >> 1);
                    let q = clamp(h0 + o, vec2<i32>(0), dims - 1);
                    let old = textureLoad(nrc_history, q, 0);
                    if old.a <= 0.0 || abs(old.a - clip.w) > 0.03 * clip.w + 0.02 { continue; }
                    let w = mix(1.0 - hf.x, hf.x, f32(o.x)) * mix(1.0 - hf.y, hf.y, f32(o.y));
                    history += w * old.rgb;
                    history_weight += w;
                }
                if history_weight > 0.05 {
                    value = mix(history / history_weight, current, nrc_resolve_params.tuning.x);
                }
            }
        }
    }
    textureStore(nrc_output, pixel, vec4<f32>(max(value, vec3<f32>(0.0)), max(view_depth, 1e-4)));
}
