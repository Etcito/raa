// ReLAX-style radiance denoiser (clean-room, after SVGF and the public ReLAX description).
// `OUT_FORMAT` is replaced by the storage format of the signal before compilation.
// Signal: rgb = linear radiance, a = confidence / weight (kept unchanged). A pixel is filtered
// when a > 0 and, if hit metadata is linked, its class is a screen hit (x == 1, y > 0.5).

struct Params {
    size: vec2<u32>,
    layers: u32,
    flags: u32,
    jitter_delta: vec2<f32>,
    max_frames: f32,
    radius: f32,
    normal_power: f32,
    depth_sigma: f32,
    clamp_sigma: f32,
    phi_luminance: f32,
    reset_mask: u32,
    step: u32,
    final_pass: u32,
    frame: u32,
}

const HISTORY_VALID: u32 = 1u;
const HAS_META: u32 = 2u;
const HAS_TRACE: u32 = 4u;

@group(0) @binding(0) var signal_tex: texture_2d<f32>;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(3) var motion_tex: texture_2d<f32>;
@group(0) @binding(4) var hist_prev: texture_2d<f32>;
@group(0) @binding(5) var mom_prev: texture_2d<f32>;
@group(0) @binding(6) var guide_prev: texture_2d<f32>;
@group(0) @binding(7) var<uniform> params: Params;
@group(0) @binding(8) var acc_out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(9) var mom_out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(10) var guide_out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(11) var meta_tex: texture_2d<f32>;
@group(0) @binding(12) var trace_tex: texture_2d<f32>;
@group(0) @binding(13) var src_tex: texture_2d<f32>;
@group(0) @binding(14) var final_out: texture_storage_2d<OUT_FORMAT, write>;

fn view_depth(d: f32) -> f32 {
    return 1.0 / max(1.0 - d, 1e-7);
}

fn in_bounds(p: vec2<i32>) -> bool {
    return all(p >= vec2<i32>(0)) && all(p < vec2<i32>(params.size));
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn valid_at(p: vec2<i32>) -> bool {
    let s = textureLoad(signal_tex, p, 0);
    if !(s.a > 0.0) { return false; }
    if (params.flags & HAS_META) != 0u {
        let m = textureLoad(meta_tex, p, 0);
        return m.x == 1.0 && m.y > 0.5;
    }
    return true;
}

// Current hit UV of p, or -1 when no trace is linked.
fn hit_uv(p: vec2<i32>) -> vec2<f32> {
    if (params.flags & HAS_TRACE) == 0u { return vec2<f32>(-1.0); }
    return textureLoad(trace_tex, p, 0).xy;
}

@compute @workgroup_size(8, 8)
fn relax_temporal(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= params.size) { return; }
    let p = vec2<i32>(id.xy);
    let d = textureLoad(depth_tex, p, 0);
    let s = textureLoad(signal_tex, p, 0);
    let nr = textureLoad(normal_tex, p, 0);
    let n = normalize(nr.xyz);
    let roughness = clamp(nr.w, 0.0, 1.0);
    let z = select(view_depth(d), 0.0, d >= 1.0);
    let valid = d < 1.0 && valid_at(p);
    let hit = hit_uv(p);
    textureStore(guide_out, p, vec4<f32>(select(0.0, z, valid), n));
    if !valid {
        textureStore(acc_out, p, vec4<f32>(s.rgb, 0.0));
        textureStore(mom_out, p, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }
    // Spatial moments of valid current neighbours (3x3): clamp box and early variance.
    var m1 = vec3<f32>(0.0);
    var m2 = vec3<f32>(0.0);
    var l1 = 0.0;
    var l2 = 0.0;
    var count = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let q = p + vec2<i32>(x, y);
            if !in_bounds(q) || !valid_at(q) { continue; }
            let c = textureLoad(signal_tex, q, 0).rgb;
            m1 += c;
            m2 += c * c;
            let l = luminance(c);
            l1 += l;
            l2 += l * l;
            count += 1.0;
        }
    }
    m1 /= count;
    let sigma = sqrt(max(m2 / count - m1 * m1, vec3<f32>(0.0)));
    let spatial_var = max(l2 / count - (l1 / count) * (l1 / count), 0.0);

    var len = 0.0;
    var history = vec3<f32>(0.0);
    var moments = vec2<f32>(0.0);
    let m = textureLoad(motion_tex, p, 0).xyz;
    if (params.flags & HISTORY_VALID) != 0u && m.z < 0.5 && params.max_frames >= 0.5 {
        let q = vec2<i32>(floor(vec2<f32>(p) + 0.5 + (params.jitter_delta - m.xy) * vec2<f32>(params.size)));
        if in_bounds(q) {
            let g = textureLoad(guide_prev, q, 0);
            var ok = g.x > 0.0 && abs(g.x - z) <= 0.05 * z && dot(g.yzw, n) > 0.8;
            let pm = textureLoad(mom_prev, q, 0);
            if ok && hit.x >= 0.0 {
                // Specular validation: the previous hit of q must be where the current hit
                // point was one frame earlier (current hit minus the motion at the hit).
                let hp = clamp(vec2<i32>(hit * vec2<f32>(params.size)), vec2<i32>(0), vec2<i32>(params.size) - 1);
                let mh = textureLoad(motion_tex, hp, 0).xy;
                let expected = hit + params.jitter_delta - mh;
                let tolerance = (1.0 + 8.0 * roughness) / f32(params.size.x);
                ok = pm.w >= 0.0 && distance(expected, vec2<f32>(pm.w, textureLoad(hist_prev, q, 0).a)) <= tolerance;
            }
            if ok {
                len = pm.z;
                history = textureLoad(hist_prev, q, 0).rgb;
                moments = pm.xy;
            }
        }
    }
    let cap = max(1.0, params.max_frames * mix(0.25, 1.0, clamp(roughness * 2.0, 0.0, 1.0)));
    len = min(len, cap);
    if len > 0.0 && params.clamp_sigma > 0.0 {
        history = clamp(history, m1 - params.clamp_sigma * sigma, m1 + params.clamp_sigma * sigma);
    }
    let alpha = 1.0 / (len + 1.0);
    let l = luminance(s.rgb);
    let color = mix(history, s.rgb, alpha);
    let mo = mix(moments, vec2<f32>(l, l * l), alpha);
    let temporal_var = max(mo.y - mo.x * mo.x, 0.0);
    let variance = select(temporal_var, spatial_var, len < 3.0);
    textureStore(acc_out, p, vec4<f32>(color, variance));
    // The hit U of this frame rides in mom.w; hit V is written by the history pass into hist.a.
    textureStore(mom_out, p, vec4<f32>(mo, min(len + 1.0, cap), select(-1.0, hit.x, hit.x >= 0.0)));
}

// SVGF a-trous iteration over the accumulated colour (rgb) and variance (a), reading the
// current guide at binding 6, the moments at 5 (history length) and the trace at 12 (cone).
fn atrous(p: vec2<i32>) -> vec4<f32> {
    let center = textureLoad(src_tex, p, 0);
    let g = textureLoad(guide_prev, p, 0);
    if g.x <= 0.0 {
        return center;
    }
    let z = g.x;
    let n = g.yzw;
    let roughness = clamp(textureLoad(normal_tex, p, 0).w, 0.0, 1.0);
    let step = i32(params.step);
    if step == 0 {
        return center;
    }
    // Gaussian-prefiltered variance (3x3) for the luminance edge stop.
    var gv = 0.0;
    var gw = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let q = p + vec2<i32>(x, y);
            if !in_bounds(q) || textureLoad(guide_prev, q, 0).x <= 0.0 { continue; }
            let k = select(0.25, 0.5, x == 0) * select(0.25, 0.5, y == 0);
            gv += k * textureLoad(src_tex, q, 0).a;
            gw += k;
        }
    }
    let sigma_l = params.phi_luminance * sqrt(max(gv / max(gw, 1e-6), 0.0)) + 1e-4;
    let lc = luminance(center.rgb);
    let normal_power = params.normal_power * mix(1.0, 0.0625, clamp(roughness * 2.0, 0.0, 1.0));
    // Specular footprint gate: blur within the reflection cone footprint when a trace is linked.
    var footprint = 1e9;
    if (params.flags & HAS_TRACE) != 0u {
        footprint = max(exp2(textureLoad(trace_tex, p, 0).w), 0.75);
    }
    // Depth gradient of the guide.
    var grad = 0.0;
    for (var axis = 0; axis < 2; axis++) {
        let e = select(vec2<i32>(0, 1), vec2<i32>(1, 0), axis == 0);
        let a = textureLoad(guide_prev, clamp(p + e, vec2<i32>(0), vec2<i32>(params.size) - 1), 0).x;
        let b = textureLoad(guide_prev, clamp(p - e, vec2<i32>(0), vec2<i32>(params.size) - 1), 0).x;
        grad += min(min(select(1e9, abs(a - z), a > 0.0), select(1e9, abs(b - z), b > 0.0)), 0.25 * z);
    }
    let kernel = array<f32, 3>(0.375, 0.25, 0.0625);
    var sum = vec3<f32>(0.0);
    var var_sum = 0.0;
    var weight = 0.0;
    for (var y = -2; y <= 2; y++) {
        for (var x = -2; x <= 2; x++) {
            let q = p + vec2<i32>(x, y) * step;
            if !in_bounds(q) { continue; }
            let gq = textureLoad(guide_prev, q, 0);
            if gq.x <= 0.0 { continue; }
            let c = textureLoad(src_tex, q, 0);
            let dist = length(vec2<f32>(q - p));
            var w = kernel[abs(x)] * kernel[abs(y)];
            if x != 0 || y != 0 {
                let wz = exp(-abs(gq.x - z) / max(params.depth_sigma * grad * dist + 0.01 * z, 1e-6));
                let wn = pow(max(dot(gq.yzw, n), 0.0), normal_power);
                let wl = exp(-abs(luminance(c.rgb) - lc) / sigma_l);
                let wf = exp(-0.5 * dist * dist / (footprint * footprint));
                w *= wz * wn * wl * wf;
            }
            sum += w * c.rgb;
            var_sum += w * w * c.a;
            weight += w;
        }
    }
    return vec4<f32>(sum / weight, var_sum / (weight * weight));
}

@compute @workgroup_size(8, 8)
fn relax_atrous(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= params.size) { return; }
    let p = vec2<i32>(id.xy);
    textureStore(acc_out, p, atrous(p));
}

// Iteration 0 (SVGF feeds it back as colour history): binding 8 receives the history (rgb,
// this frame's hit V in alpha), binding 9 the filtered colour and variance for the next pass.
@compute @workgroup_size(8, 8)
fn relax_history(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= params.size) { return; }
    let p = vec2<i32>(id.xy);
    let filtered = atrous(p);
    let hit = hit_uv(p);
    textureStore(acc_out, p, vec4<f32>(filtered.rgb, select(-1.0, hit.y, hit.x >= 0.0)));
    textureStore(mom_out, p, filtered);
}

@compute @workgroup_size(8, 8)
fn relax_final(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= params.size) { return; }
    let p = vec2<i32>(id.xy);
    let s = textureLoad(signal_tex, p, 0);
    let d = textureLoad(depth_tex, p, 0);
    if d >= 1.0 || !valid_at(p) {
        textureStore(final_out, p, s);
        return;
    }
    let c = atrous(p).rgb;
    textureStore(final_out, p, vec4<f32>(max(c, vec3<f32>(0.0)), s.a));
}
