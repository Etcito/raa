// Real-time denoisers in the spirit of NRD SIGMA (visibility pairs) and ReLAX (radiance).
// Clean-room implementation after SVGF (Schied et al. 2017) and the public ReBLUR / ReLAX /
// SIGMA descriptions. No NRD source code was used.
//
// Shared conventions: depth is the standard hardware depth (1 = background); 1 / (1 - d) is
// proportional to view depth for far >> near and is used for relative depth tests. Motion is
// current minus previous UV; motion.z > 0.5 marks an invalid vector. `jitter_delta` is the
// previous minus current camera jitter in UV.

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
const EPS_U: f32 = 1e-8;

@group(0) @binding(0) var src: texture_2d_array<f32>;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(3) var motion_tex: texture_2d<f32>;
@group(0) @binding(4) var hist_prev: texture_2d_array<f32>;
@group(0) @binding(5) var mom_prev: texture_2d_array<f32>;
@group(0) @binding(6) var guide_prev: texture_2d<f32>;
@group(0) @binding(7) var<uniform> params: Params;
@group(0) @binding(8) var acc_out: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(9) var mom_out: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(10) var guide_out: texture_storage_2d<rgba32float, write>;

fn view_depth(d: f32) -> f32 {
    return 1.0 / max(1.0 - d, 1e-7);
}

fn in_bounds(p: vec2<i32>) -> bool {
    return all(p >= vec2<i32>(0)) && all(p < vec2<i32>(params.size));
}

// Previous pixel of `p` and whether its stored guide matches the current surface.
fn reproject(p: vec2<i32>, z: f32, n: vec3<f32>) -> vec3<i32> {
    if (params.flags & HISTORY_VALID) == 0u {
        return vec3<i32>(0, 0, 0);
    }
    let m = textureLoad(motion_tex, p, 0).xyz;
    if m.z > 0.5 {
        return vec3<i32>(0, 0, 0);
    }
    let q = vec2<i32>(floor(vec2<f32>(p) + 0.5 + (params.jitter_delta - m.xy) * vec2<f32>(params.size)));
    if !in_bounds(q) {
        return vec3<i32>(q, 0);
    }
    let g = textureLoad(guide_prev, q, 0);
    let ok = g.x > 0.0 && abs(g.x - z) <= 0.05 * z && dot(g.yzw, n) > 0.8;
    return vec3<i32>(q, select(0, 1, ok));
}

fn ratio(pair: vec2<f32>) -> f32 {
    return select(-1.0, clamp(pair.x / pair.y, 0.0, 1.0), pair.y > 1e-6);
}

// ---- SIGMA-style visibility pairs -------------------------------------------------------

@compute @workgroup_size(8, 8)
fn sigma_temporal(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= params.size) { return; }
    let p = vec2<i32>(id.xy);
    let layer = i32(id.z);
    let d = textureLoad(depth_tex, p, 0);
    let cur = textureLoad(src, p, layer, 0);
    if d >= 1.0 {
        textureStore(acc_out, p, layer, cur);
        textureStore(mom_out, p, layer, vec4<f32>(0.0));
        if layer == 0 { textureStore(guide_out, p, vec4<f32>(0.0)); }
        return;
    }
    let z = view_depth(d);
    let n = normalize(textureLoad(normal_tex, p, 0).xyz);
    if layer == 0 { textureStore(guide_out, p, vec4<f32>(z, n)); }
    // Neighbourhood statistics of the current raw visibility, per slot of this texel.
    var sum = vec2<f32>(0.0);
    var sum2 = vec2<f32>(0.0);
    var count = vec2<f32>(0.0);
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let q = clamp(p + vec2<i32>(x, y), vec2<i32>(0), vec2<i32>(params.size) - 1);
            let t = textureLoad(src, q, layer, 0);
            let v = vec2<f32>(ratio(t.xy), ratio(t.zw));
            let ok = v >= vec2<f32>(0.0);
            sum += select(vec2<f32>(0.0), v, ok);
            sum2 += select(vec2<f32>(0.0), v * v, ok);
            count += select(vec2<f32>(0.0), vec2<f32>(1.0), ok);
        }
    }
    let mean = sum / max(count, vec2<f32>(1.0));
    let sigma = sqrt(max(sum2 / max(count, vec2<f32>(1.0)) - mean * mean, vec2<f32>(0.0)));
    let r = reproject(p, z, n);
    var hist = vec4<f32>(0.0);
    var len = vec2<f32>(0.0);
    if r.z == 1 {
        hist = textureLoad(hist_prev, r.xy, layer, 0);
        len = textureLoad(mom_prev, r.xy, layer, 0).xy;
    }
    var res = cur;
    var out_len = vec2<f32>(0.0);
    for (var s = 0u; s < 2u; s++) {
        let slot = u32(layer) * 2u + s;
        let c = select(cur.xy, cur.zw, s == 1u);
        var h = select(hist.xy, hist.zw, s == 1u);
        let reset = (params.reset_mask & (1u << slot)) != 0u;
        var l = select(len.x, len.y, s == 1u);
        if r.z != 1 || reset || h.y <= EPS_U || params.max_frames < 0.5 {
            l = 0.0;
        } else if params.clamp_sigma > 0.0 && count[s] > 0.0 {
            let lo = mean[s] - params.clamp_sigma * sigma[s];
            let hi = mean[s] + params.clamp_sigma * sigma[s];
            h.x = clamp(h.x / h.y, lo, hi) * h.y;
        }
        let alpha = 1.0 / (l + 1.0);
        let a = select(c, mix(h, c, alpha), l > 0.0);
        let next = min(l + 1.0, params.max_frames);
        if s == 0u {
            res = vec4<f32>(a, res.zw);
            out_len.x = next;
        } else {
            res = vec4<f32>(res.xy, a);
            out_len.y = next;
        }
    }
    textureStore(acc_out, p, layer, res);
    textureStore(mom_out, p, layer, vec4<f32>(out_len, 0.0, 0.0));
}

fn hash(p: vec2<u32>, frame: u32) -> f32 {
    var h = p.x * 1973u + p.y * 9277u + frame * 26699u;
    h = (h ^ 61u) ^ (h >> 16u);
    h = h * 9u;
    h = h ^ (h >> 4u);
    h = h * 0x27d4eb2du;
    h = h ^ (h >> 15u);
    return f32(h & 0xffffu) / 65536.0;
}

// Depth gradient magnitude of the guide at p, in view-depth units per pixel.
fn depth_gradient(p: vec2<i32>, z: f32) -> f32 {
    var g = 0.0;
    for (var axis = 0; axis < 2; axis++) {
        let e = select(vec2<i32>(0, 1), vec2<i32>(1, 0), axis == 0);
        let a = textureLoad(guide_prev, clamp(p + e, vec2<i32>(0), vec2<i32>(params.size) - 1), 0).x;
        let b = textureLoad(guide_prev, clamp(p - e, vec2<i32>(0), vec2<i32>(params.size) - 1), 0).x;
        let da = select(1e9, abs(a - z), a > 0.0);
        let db = select(1e9, abs(b - z), b > 0.0);
        g += min(min(da, db), 0.25 * z);
    }
    return g;
}

fn edge_weight(z: f32, n: vec3<f32>, grad: f32, q: vec2<i32>, dist: f32) -> f32 {
    let g = textureLoad(guide_prev, q, 0);
    if g.x <= 0.0 { return 0.0; }
    let tol = params.depth_sigma * grad * dist + 0.01 * z;
    let wz = exp(-abs(g.x - z) / max(tol, 1e-6));
    let wn = pow(max(dot(g.yzw, n), 0.0), params.normal_power);
    return wz * wn;
}

// The spatial pass binds the current guide at binding 6 and the accumulated pairs at 0.
@compute @workgroup_size(8, 8)
fn sigma_spatial(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= params.size) { return; }
    let p = vec2<i32>(id.xy);
    let layer = i32(id.z);
    let center = textureLoad(src, p, layer, 0);
    let g = textureLoad(guide_prev, p, 0);
    if g.x <= 0.0 || params.radius <= 0.0 {
        textureStore(acc_out, p, layer, center);
        return;
    }
    let z = g.x;
    let n = g.yzw;
    let len = textureLoad(mom_prev, p, layer, 0).xy;
    // Penumbra estimate: spread of the accumulated visibility in a 3x3 window.
    var sum = vec2<f32>(0.0);
    var sum2 = vec2<f32>(0.0);
    var count = vec2<f32>(0.0);
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let q = clamp(p + vec2<i32>(x, y), vec2<i32>(0), vec2<i32>(params.size) - 1);
            let t = textureLoad(src, q, layer, 0);
            let v = vec2<f32>(ratio(t.xy), ratio(t.zw));
            let ok = v >= vec2<f32>(0.0);
            sum += select(vec2<f32>(0.0), v, ok);
            sum2 += select(vec2<f32>(0.0), v * v, ok);
            count += select(vec2<f32>(0.0), vec2<f32>(1.0), ok);
        }
    }
    let mean = sum / max(count, vec2<f32>(1.0));
    let spread = sqrt(max(sum2 / max(count, vec2<f32>(1.0)) - mean * mean, vec2<f32>(0.0)));
    let radius = params.radius * clamp(spread * 4.0, vec2<f32>(0.0), vec2<f32>(1.0))
        / sqrt(vec2<f32>(1.0) + len);
    let grad = depth_gradient(p, z);
    let rotation = 6.2831853 * hash(id.xy, params.frame);
    var acc = vec4<f32>(center);
    var weight = vec2<f32>(1.0);
    let taps = 12u;
    for (var i = 0u; i < taps; i++) {
        let t = (f32(i) + 0.5) / f32(taps);
        let angle = rotation + f32(i) * 2.3999632;
        let dir = vec2<f32>(cos(angle), sin(angle)) * sqrt(t);
        for (var s = 0u; s < 2u; s++) {
            let r = radius[s];
            if r < 0.5 { continue; }
            let q = p + vec2<i32>(round(dir * r));
            if !in_bounds(q) || all(q == p) { continue; }
            let dist = length(vec2<f32>(q - p));
            let w = exp(-2.0 * t) * edge_weight(z, n, grad, q, dist);
            if w <= 1e-4 { continue; }
            let pair = textureLoad(src, q, layer, 0);
            if s == 0u {
                acc = vec4<f32>(acc.xy + w * pair.xy, acc.zw);
                weight.x += w;
            } else {
                acc = vec4<f32>(acc.xy, acc.zw + w * pair.zw);
                weight.y += w;
            }
        }
    }
    textureStore(acc_out, p, layer, vec4<f32>(acc.xy / weight.x, acc.zw / weight.y));
}
