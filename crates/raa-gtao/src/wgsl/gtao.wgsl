// ao.gtao: ground-truth ambient occlusion (Jimenez 2016, GTAO values: 4 slices x 8 steps
// per side, radius 0.8 m). Standalone compute module (no PRELUDE), own bindings in group 0.
// Workgroup 8x8 over the S12 `depth` and `normal` targets; the output is 1 = unoccluded.
//
// Per pixel: the view position is reconstructed from depth through `inv_proj`; four slice
// directions are rotated by a per-pixel interleaved gradient noise and the step distance is
// offset by a second noise value; the projected radius in pixels is
// `radius_m * proj[1][1] * 0.5 * h / view_depth`; each side marches 8 horizon steps with
// `textureLoad` depths, falling back to the previous horizon when a sample lies more than
// the radius behind (thickness heuristic); the normal is projected onto the slice plane and
// the cosine-weighted visibility of the visible arc is integrated analytically per slice;
// the slices are averaged and `out = clamp(1 - intensity * (1 - vis^power), 0, 1)`.

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

// Antiderivative of the cosine-weighted visibility of one slice (Jimenez 2016), evaluated
// between two signed slice angles `a` and `b`; `gamma` is the projected normal angle and
// `r` the out-of-plane length of the normal. Returns 0 for an empty interval.
fn raa_gtao_arc(a: f32, b: f32, gamma: f32, r: f32) -> f32 {
    if (b <= a) {
        return 0.0;
    }
    return r * 0.25
        * (-cos(2.0 * b - gamma) + cos(2.0 * a - gamma) + 2.0 * (b - a) * sin(gamma));
}

// Signed horizon of the occluders within `radius`: the maximum elevation on the `side`
// (+1) of the view direction or the minimum on the other (-1), both measured with the same
// slice direction `dir`.
fn raa_gtao_horizon(
    p: vec3<f32>,
    v: vec3<f32>,
    dir: vec3<f32>,
    screen_dir: vec2<f32>,
    coord: vec2<i32>,
    max_px: vec2<i32>,
    step_px: f32,
    jitter: f32,
    radius: f32,
    side: f32,
) -> f32 {
    var h = 0.0;
    var prev = 0.0;
    for (var i = 0u; i < STEPS; i = i + 1u) {
        let dist = (f32(i) + jitter + 1.0) * step_px;
        let sp = clamp(
            vec2<f32>(coord) + side * dist * screen_dir,
            vec2<f32>(0.0),
            vec2<f32>(max_px),
        );
        let sd = textureLoad(depth_tex, vec2<i32>(sp), 0);
        if (sd >= 1.0) {
            break;
        }
        let delta = raa_gtao_view_pos(sp * gtao_params.viewport.zw, sd) - p;
        // Thickness heuristic: a sample far behind the point is background, not an
        // occluder; fall off to the previous horizon.
        if (-dot(delta, v) > radius) {
            h = prev;
            break;
        }
        let elev = clamp(
            atan2(dot(delta, dir), dot(delta, v)),
            -PI * 0.5,
            PI * 0.5,
        );
        if ((side > 0.0 && elev > h) || (side < 0.0 && elev < h)) {
            prev = h;
            h = elev;
        }
    }
    return h;
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
    // viewer so the slice bounds below stay well defined.
    if (dot(n, v) < 0.0) {
        n = -n;
    }
    let view_depth = max(-p.z, 1e-4);
    let radius_px = radius * gtao_params.proj[1][1] * 0.5 * gtao_params.viewport.y / view_depth;
    let step_px = radius_px / f32(STEPS);
    let max_px = vec2<i32>(size) - vec2<i32>(1);
    let noise = raa_gtao_ign(pixel, frame);
    let jitter = raa_gtao_ign(pixel + vec2<f32>(17.0, 23.0), frame);

    var visible = 0.0;
    for (var s = 0u; s < SLICES; s = s + 1u) {
        let phi = (f32(s) + noise) * (PI * 0.25);
        let c = cos(phi);
        let sn = sin(phi);
        // Screen y grows downwards; the view-space slice direction is orthonormalised
        // against the view vector so the projection below is exact.
        let base3 = vec3<f32>(c, -sn, 0.0);
        let omega = normalize(base3 - dot(base3, v) * v);
        let slice_normal = cross(v, omega);
        let r = sqrt(max(1.0 - dot(n, slice_normal) * dot(n, slice_normal), 0.0));
        let gamma = atan2(dot(n, omega), dot(n, v));
        let lo = max(0.0, gamma - PI * 0.5);
        let hi = min(PI, gamma + PI * 0.5);
        let h_pos = raa_gtao_horizon(
            p, v, omega, vec2<f32>(c, sn), coord, max_px, step_px, jitter, radius, 1.0,
        );
        let h_neg = raa_gtao_horizon(
            p, v, omega, vec2<f32>(c, sn), coord, max_px, step_px, jitter, radius, -1.0,
        );
        let a = min(h_neg, h_pos);
        let b = max(h_neg, h_pos);
        // The wedge [a, b] is occluded; integrate the visible arcs on both sides.
        visible += raa_gtao_arc(lo, min(hi, a), gamma, r);
        visible += raa_gtao_arc(max(lo, b), hi, gamma, r);
    }
    let vis = clamp((2.0 / f32(SLICES)) * visible, 0.0, 1.0);
    let ao = clamp(1.0 - intensity * (1.0 - pow(vis, power)), 0.0, 1.0);
    textureStore(out_ao, coord, vec4<f32>(ao));
}
