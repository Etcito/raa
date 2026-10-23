// ao.gtao: ground-truth ambient occlusion (Jimenez 2016, GTAO values: 4 slices x 8 steps
// per side, radius 0.8 m). Standalone compute module (no PRELUDE), own bindings in group 0.
// Workgroup 8x8 over the S12 `depth` and `normal` targets; the output is 1 = unoccluded.
//
// Per pixel: the view position is reconstructed from depth through `inv_proj`; four slice
// directions are rotated by a per-pixel interleaved gradient noise and the step distance is
// offset by a second noise value; the projected radius in pixels is
// `radius_m * proj[1][1] * 0.5 * h / view_depth`. In each slice (the plane through the eye
// spanned by the view vector v and the slice direction) angles are measured from v; the
// projected normal sits at angle gamma and bounds the hemisphere to
// [gamma - pi/2, gamma + pi/2]. Each side marches 8 horizon steps (at least one pixel apart,
// samples at pixel centres), keeping the largest cosine to v, every sample faded toward the
// tangent-plane bound as its distance reaches the radius (XeGTAO falloff, thickness
// heuristic). The cosine-weighted visibility between the two horizons is integrated
// analytically (Jimenez 2016) and normalised by the same integral over the unoccluded
// hemisphere, so an open flat surface is exactly 1; then
// `out = clamp(1 - intensity * (1 - vis^power), 0, 1)`.

struct GtaoParams {
    proj: mat4x4<f32>,     // @0
    inv_proj: mat4x4<f32>, // @64
    viewport: vec4<f32>,   // @128 (w, h, 1/w, 1/h)
    params: vec4<f32>,     // @144 (radius_m, intensity, power, frame index)
}

@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var normal_tex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> gtao_params: GtaoParams;
@group(0) @binding(3) var out_ao: texture_storage_2d<r32float, write>;

const SLICES: u32 = 4u;
const STEPS: u32 = 8u;
const PI: f32 = 3.14159265359;

// Interleaved gradient noise (Jimenez 2014), shifted per frame.
fn raa_gtao_ign(pixel: vec2<f32>, frame: f32) -> f32 {
    let p = pixel + vec2<f32>(5.588238 * frame);
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

// View-space position of screen uv (origin top-left) at clip depth d.
fn raa_gtao_view_pos(uv: vec2<f32>, d: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, d, 1.0);
    let v = gtao_params.inv_proj * ndc;
    return v.xyz / v.w;
}

// Cosine-weighted visibility of one side of a slice (Jimenez 2016), from the view vector
// (angle 0) to the signed horizon angle `h`; `gamma` is the projected normal angle.
fn raa_gtao_arc(h: f32, gamma: f32) -> f32 {
    return 0.25 * (-cos(2.0 * h - gamma) + cos(gamma) + 2.0 * h * sin(gamma));
}

// Horizon cosine of one side (`side` +1 along `screen_dir`, -1 against it): the largest
// cosine to `v` of the samples, each faded toward `low` (the tangent-plane bound) as its
// distance from `p` reaches `radius`. Sky samples carry no occluder.
fn raa_gtao_horizon_cos(
    p: vec3<f32>,
    v: vec3<f32>,
    screen_dir: vec2<f32>,
    pixel: vec2<f32>,
    max_px: vec2<i32>,
    radius_px: f32,
    jitter: f32,
    radius: f32,
    side: f32,
    low: f32,
) -> f32 {
    var hc = low;
    for (var i = 0u; i < STEPS; i = i + 1u) {
        let s = (f32(i) + jitter) / f32(STEPS);
        // Quadratic spacing (denser near the pixel), at least one pixel per step.
        let dist = max(s * s * radius_px, f32(i) + 1.0);
        let sp = clamp(
            vec2<i32>(floor(pixel + side * dist * screen_dir)),
            vec2<i32>(0),
            max_px,
        );
        let sd = textureLoad(depth_tex, sp, 0);
        if (sd >= 1.0) {
            continue;
        }
        let q = raa_gtao_view_pos((vec2<f32>(sp) + vec2<f32>(0.5)) * gtao_params.viewport.zw, sd);
        let delta = q - p;
        let len = length(delta);
        if (len < 1e-5) {
            continue;
        }
        let falloff = clamp((radius - len) / (0.6 * radius), 0.0, 1.0);
        hc = max(hc, mix(low, dot(delta, v) / len, falloff));
    }
    return hc;
}

@compute @workgroup_size(8, 8)
fn gtao(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(gtao_params.viewport.xy);
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }
    let coord = vec2<i32>(gid.xy);
    let d = textureLoad(depth_tex, coord, 0);
    if (d >= 1.0) {
        textureStore(out_ao, coord, vec4<f32>(1.0));
        return;
    }
    let radius = gtao_params.params.x;
    let intensity = gtao_params.params.y;
    let power = gtao_params.params.z;
    let frame = gtao_params.params.w;
    let pixel = vec2<f32>(gid.xy) + vec2<f32>(0.5);
    let p = raa_gtao_view_pos(pixel * gtao_params.viewport.zw, d);
    let v = normalize(-p);
    var n = normalize(textureLoad(normal_tex, coord, 0).xyz);
    // Only the front hemisphere carries occlusion; flip a back-facing normal toward the
    // viewer so gamma below stays in [-pi/2, pi/2].
    if (dot(n, v) < 0.0) {
        n = -n;
    }
    let view_depth = max(-p.z, 1e-4);
    let radius_px = radius * gtao_params.proj[1][1] * 0.5 * gtao_params.viewport.y / view_depth;
    let max_px = vec2<i32>(size) - vec2<i32>(1);
    let noise = raa_gtao_ign(pixel, frame);
    let jitter = raa_gtao_ign(pixel + vec2<f32>(17.0, 23.0), frame);

    var visible = 0.0;
    var unoccluded = 0.0;
    for (var s = 0u; s < SLICES; s = s + 1u) {
        let phi = (f32(s) + noise) * (PI / f32(SLICES));
        let c = cos(phi);
        let sn = sin(phi);
        // Screen y grows downwards; the view-space slice direction is orthonormalised
        // against the view vector, so (v, omega) spans the slice plane exactly.
        let base3 = vec3<f32>(c, -sn, 0.0);
        let omega = normalize(base3 - dot(base3, v) * v);
        let nv = dot(n, v);
        let no = dot(n, omega);
        // Length of the normal projected onto the slice plane and its angle from v.
        let r = sqrt(nv * nv + no * no);
        if (r < 1e-4) {
            continue;
        }
        let gamma = atan2(no, nv);
        let c_pos = raa_gtao_horizon_cos(
            p, v, vec2<f32>(c, sn), pixel, max_px, radius_px, jitter, radius, 1.0,
            cos(gamma + PI * 0.5),
        );
        let c_neg = raa_gtao_horizon_cos(
            p, v, vec2<f32>(c, sn), pixel, max_px, radius_px, jitter, radius, -1.0,
            cos(gamma - PI * 0.5),
        );
        let h_pos = min(acos(clamp(c_pos, -1.0, 1.0)), gamma + PI * 0.5);
        let h_neg = max(-acos(clamp(c_neg, -1.0, 1.0)), gamma - PI * 0.5);
        visible += r * (raa_gtao_arc(h_neg, gamma) + raa_gtao_arc(h_pos, gamma));
        unoccluded += r
            * (raa_gtao_arc(gamma - PI * 0.5, gamma) + raa_gtao_arc(gamma + PI * 0.5, gamma));
    }
    let vis = clamp(visible / max(unoccluded, 1e-6), 0.0, 1.0);
    let ao = clamp(1.0 - intensity * (1.0 - pow(vis, power)), 0.0, 1.0);
    textureStore(out_ao, coord, vec4<f32>(ao));
}
