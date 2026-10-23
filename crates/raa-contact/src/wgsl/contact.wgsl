// shadow.contact: screen-space contact shadows (CONTACT: 24 steps toward the sun).
// Standalone compute module, group 0 only. Output .r = sun visibility (1 = lit).

struct ContactParams {
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    // (w, h, 1/w, 1/h)
    viewport: vec4<f32>,
    // Unit view-space direction toward the sun, w 0.
    sun_view: vec4<f32>,
    // (distance_m, thickness_m, intensity, frame)
    params: vec4<f32>,
}

@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var<uniform> cp: ContactParams;
@group(0) @binding(2) var out_tex: texture_storage_2d<rgba16float, write>;

const STEPS: u32 = 24u;

// Interleaved gradient noise (Jimenez 2014), shifted per frame.
fn ign(pixel: vec2<f32>, frame: f32) -> f32 {
    let p = pixel + vec2<f32>(5.588238 * frame);
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

// View-space position of screen uv (origin top-left) at clip depth d.
fn view_pos(uv: vec2<f32>, d: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, d, 1.0);
    let v = cp.inv_proj * ndc;
    return v.xyz / v.w;
}

@compute @workgroup_size(8, 8)
fn contact(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(cp.viewport.xy);
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }
    let coord = vec2<i32>(gid.xy);
    let d = textureLoad(depth_tex, coord, 0);
    if (d >= 1.0) {
        textureStore(out_tex, coord, vec4<f32>(1.0, 0.0, 0.0, 1.0));
        return;
    }
    let distance = cp.params.x;
    let thickness = cp.params.y;
    let intensity = cp.params.z;
    let pixel = vec2<f32>(gid.xy) + vec2<f32>(0.5);
    let origin = view_pos(pixel * cp.viewport.zw, d);
    let dir = cp.sun_view.xyz;
    let step_len = distance / f32(STEPS);
    let jitter = ign(vec2<f32>(gid.xy), cp.params.w);
    let max_px = vec2<i32>(size) - vec2<i32>(1);

    var occlusion = 0.0;
    for (var i = 0u; i < STEPS; i = i + 1u) {
        let t = (f32(i) + jitter) * step_len;
        let p = origin + dir * t;
        let clip = cp.proj * vec4<f32>(p, 1.0);
        if (clip.w <= 0.0) {
            break;
        }
        let ndc = clip.xy / clip.w;
        let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) {
            break;
        }
        let sp = clamp(vec2<i32>(uv * cp.viewport.xy), vec2<i32>(0), max_px);
        let sd = textureLoad(depth_tex, sp, 0);
        let scene = view_pos(uv, sd);
        // Positive when the scene surface is in front of (closer than) the ray point.
        let delta = scene.z - p.z;
        if (delta > 0.0 && delta < thickness) {
            let fade_end = clamp((distance - t) / (0.25 * distance), 0.0, 1.0);
            let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
            let fade_edge = clamp(edge / 0.05, 0.0, 1.0);
            occlusion = fade_end * fade_edge;
            break;
        }
    }
    let lit = 1.0 - intensity * occlusion;
    textureStore(out_tex, coord, vec4<f32>(lit, 0.0, 0.0, 1.0));
}
