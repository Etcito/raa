// pass.lighting.reconstruct: full-resolution accumulation and composition of the forward
// lightingOutput "layers-interlaced" signals. Exact copy of nt_model.step (neural-target) with
// the 3x3 predicted kernel replaced by identity and the four alphas by constants (the analytic
// ablation), then nt_model.composite. One dispatch per frame, workgroup 8x8.
//
// Signals l = (E rgb, a, s): E = layerE.rgb (probe irradiance already x w), a = layerKd.a,
// s = color.a, present at a pixel when color.a >= 0 (R frames, phase pixels, geometry only).
// History (ping-pong, Rgba16Float): H0 = (E rgb, a), H1 = (s, linear view depth z, confidence c,
// 0). z is the linear view depth in metres from the prepass depth through the jittered
// inv_proj of FrameGpu (0 = sky / no geometry).
//
// Per pixel p: reproject q = p + 0.5 - motion.xy * size + jitter_px (jitter_px = (previous -
// current) camera jitter NDC x (0.5 W, -0.5 H) when jitterCorrect, else 0); bilinear H0/H1 at q;
// valid = inside [0, size] && motion.z < 0.5 && |z_hist - z| <= 0.05 z + 0.02 && z > 0 &&
// history valid; hc = valid ? c_hist : 0.
// G frame: out = history; a hole (hc <= 0) takes the guide x hc weighted mean of the 12 RING
// taps, each reprojected with its own motion and depth (kept bilinear when the weight sum <=
// 1e-4); new history = (history, z, hc x decay), the fill is not stored.
// R frame, phase (px, py): per axis the taps are {p} when p has the phase parity, else
// {p - 1, p + 1} (the phase pixels of the two neighbouring blocks); weight b x (guide_w + 1e-4)
// with b = 1 / tap count, 0 for a tap outside the image or without a sample. q = 1 and the hit
// alphas at the phase pixel, qMiss and the miss alphas elsewhere. Blend per group (E,a) and s:
// wc = alpha q (0 without any tap sample), wh = (1 - alpha) hc, out = (wc cur + wh hist) /
// max(wc + wh, 1e-5); c_new = mean over the groups of (wc q + wh hc) / max(wc + wh, 1e-5),
// clamped to [0, 1] x decay.
// Composite: C = d0 s + (kd E + s0) a + em; sky (z = 0): C = em, history zero.

struct Params {
    size: vec2<u32>,
    flags: u32,
    debug: u32,
    phase: vec2<u32>,
    pad: vec2<u32>,
    jitter_px: vec2<f32>,
    q_miss: f32,
    decay: f32,
    // s hit, s miss, (E,a) hit, (E,a) miss.
    alpha: vec4<f32>,
}

const HISTORY_VALID: u32 = 1u;
const REAL_FRAME: u32 = 2u;

@group(0) @binding(0) var color_tex: texture_2d<f32>;
@group(0) @binding(1) var kd_tex: texture_2d<f32>;
@group(0) @binding(2) var s0_tex: texture_2d<f32>;
@group(0) @binding(3) var em_tex: texture_2d<f32>;
@group(0) @binding(4) var e_tex: texture_2d<f32>;
@group(0) @binding(5) var depth_tex: texture_depth_2d;
@group(0) @binding(6) var normal_tex: texture_2d<f32>;
@group(0) @binding(7) var motion_tex: texture_2d<f32>;
@group(0) @binding(8) var h0_prev: texture_2d<f32>;
@group(0) @binding(9) var h1_prev: texture_2d<f32>;
@group(0) @binding(10) var hist_sampler: sampler;
@group(0) @binding(11) var<uniform> params: Params;
// raa-lit FrameGpu (768 bytes): inv_proj (jittered) at vec4 12..15.
@group(0) @binding(12) var<uniform> frame: array<vec4<f32>, 48>;
@group(0) @binding(13) var h0_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(14) var h1_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(15) var out_color: texture_storage_2d<rgba16float, write>;

struct History {
    ea: vec4<f32>,
    s: f32,
    c: f32,
}

fn linear_z(p: vec2<i32>) -> f32 {
    let d = textureLoad(depth_tex, p, 0);
    if (d >= 1.0) {
        return 0.0;
    }
    let size = vec2<f32>(params.size);
    let uv = (vec2<f32>(p) + vec2<f32>(0.5)) / size;
    let inv_proj = mat4x4<f32>(frame[12], frame[13], frame[14], frame[15]);
    let v = inv_proj * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, d, 1.0);
    return max(-v.z / v.w, 0.0);
}

fn view_normal(p: vec2<i32>) -> vec3<f32> {
    let n = textureLoad(normal_tex, p, 0).xyz;
    let len2 = dot(n, n);
    return select(vec3<f32>(0.0), n * inverseSqrt(len2), len2 > 1e-12);
}

// nt_model.guide_w: depth exp(-|dz| / (0.03 zc + 0.01)) x clamp(n.n, 0, 1)^8, 0 off geometry.
fn guide_w(zc: f32, nc: vec3<f32>, zk: f32, nk: vec3<f32>) -> f32 {
    if (zc <= 0.0 || zk <= 0.0) {
        return 0.0;
    }
    let wz = exp(-abs(zk - zc) / (0.03 * zc + 0.01));
    var wn = clamp(dot(nk, nc), 0.0, 1.0);
    wn = wn * wn;
    wn = wn * wn;
    wn = wn * wn;
    return wz * wn;
}

// nt_model.reproject at pixel t of linear depth zt.
fn reproject(t: vec2<i32>, zt: f32) -> History {
    let size = vec2<f32>(params.size);
    let mv = textureLoad(motion_tex, t, 0).xyz;
    let q = vec2<f32>(t) + vec2<f32>(0.5) - mv.xy * size + params.jitter_px;
    let uv = q / size;
    let h0 = textureSampleLevel(h0_prev, hist_sampler, uv, 0.0);
    let h1 = textureSampleLevel(h1_prev, hist_sampler, uv, 0.0);
    let inside = all(q >= vec2<f32>(0.0)) && all(q <= size);
    let valid = (params.flags & HISTORY_VALID) != 0u && inside && mv.z < 0.5 && zt > 0.0
        && abs(h1.y - zt) <= 0.05 * zt + 0.02;
    return History(h0, h1.x, select(0.0, h1.z, valid));
}

fn debug_color(view: u32, e: vec4<f32>, s: f32, c: f32, hc: f32) -> vec3<f32> {
    switch view {
        case 1u: { return vec3<f32>(s); }
        case 2u: { return vec3<f32>(e.a); }
        case 3u: { return e.rgb; }
        case 4u: { return vec3<f32>(c); }
        default: { return vec3<f32>(select(0.0, 1.0, hc <= 0.0), max(hc, 0.0), 0.0); }
    }
}

@compute @workgroup_size(8, 8)
fn reconstruct(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid.xy >= params.size)) {
        return;
    }
    let p = vec2<i32>(gid.xy);
    let max_p = vec2<i32>(params.size) - vec2<i32>(1);
    let base = textureLoad(color_tex, p, 0);
    let kd = textureLoad(kd_tex, p, 0);
    let s0 = textureLoad(s0_tex, p, 0).rgb;
    let em = textureLoad(em_tex, p, 0).rgb;
    let z = linear_z(p);
    if (z <= 0.0) {
        textureStore(h0_out, p, vec4<f32>(0.0));
        textureStore(h1_out, p, vec4<f32>(0.0));
        textureStore(out_color, p, vec4<f32>(select(vec3<f32>(0.0), em, params.debug == 0u), 1.0));
        return;
    }
    let n = view_normal(p);
    let h = reproject(p, z);
    var ea = h.ea;
    var s = h.s;
    var c_store = h.c * params.decay;
    if ((params.flags & REAL_FRAME) == 0u) {
        // G frame: the reprojected history; holes filled from the ring (not stored).
        if (h.c <= 0.0) {
            var ring = array<vec2<i32>, 12>(
                vec2<i32>(-2, 0), vec2<i32>(2, 0), vec2<i32>(0, -2), vec2<i32>(0, 2),
                vec2<i32>(-4, -4), vec2<i32>(4, -4), vec2<i32>(-4, 4), vec2<i32>(4, 4),
                vec2<i32>(-7, 0), vec2<i32>(7, 0), vec2<i32>(0, -7), vec2<i32>(0, 7),
            );
            var acc = vec4<f32>(0.0);
            var acc_s = 0.0;
            var ws = 0.0;
            for (var i = 0u; i < 12u; i = i + 1u) {
                let t = clamp(p + ring[i], vec2<i32>(0), max_p);
                let zt = linear_z(t);
                if (zt <= 0.0) {
                    continue;
                }
                let ht = reproject(t, zt);
                let w = guide_w(z, n, zt, view_normal(t)) * ht.c;
                acc = acc + w * ht.ea;
                acc_s = acc_s + w * ht.s;
                ws = ws + w;
            }
            if (ws > 1e-4) {
                ea = acc / ws;
                s = acc_s / ws;
            }
        }
        textureStore(h0_out, p, h.ea);
        textureStore(h1_out, p, vec4<f32>(h.s, z, c_store, 0.0));
    } else {
        // R frame: the guided current estimate from the phase pixels, blended with history.
        let ph = vec2<i32>(params.phase);
        let own = all((p & vec2<i32>(1)) == ph);
        var xs = array<i32, 2>(p.x, p.x);
        var ys = array<i32, 2>(p.y, p.y);
        var nx = 1;
        var ny = 1;
        if ((p.x & 1) != ph.x) {
            xs = array<i32, 2>(p.x - 1, p.x + 1);
            nx = 2;
        }
        if ((p.y & 1) != ph.y) {
            ys = array<i32, 2>(p.y - 1, p.y + 1);
            ny = 2;
        }
        let b = 1.0 / f32(nx * ny);
        var acc = vec4<f32>(0.0);
        var acc_s = 0.0;
        var ws = 0.0;
        for (var j = 0; j < ny; j = j + 1) {
            for (var i = 0; i < nx; i = i + 1) {
                let t = vec2<i32>(xs[i], ys[j]);
                if (any(t < vec2<i32>(0)) || any(t > max_p)) {
                    continue;
                }
                let st = textureLoad(color_tex, t, 0).a;
                if (st < 0.0) {
                    continue;
                }
                let w = b * (guide_w(z, n, linear_z(t), view_normal(t)) + 1e-4);
                let et = textureLoad(e_tex, t, 0).rgb;
                let at = textureLoad(kd_tex, t, 0).a;
                acc = acc + w * vec4<f32>(et, at);
                acc_s = acc_s + w * st;
                ws = ws + w;
            }
        }
        let has_cur = ws > 0.0;
        let cur = select(vec4<f32>(0.0), acc / max(ws, 1e-30), has_cur);
        let cur_s = select(0.0, acc_s / max(ws, 1e-30), has_cur);
        let q = select(params.q_miss, 1.0, own);
        let alpha_s = select(params.alpha.y, params.alpha.x, own);
        let alpha_e = select(params.alpha.w, params.alpha.z, own);
        let wc_e = select(0.0, alpha_e * q, has_cur);
        let wh_e = (1.0 - alpha_e) * h.c;
        let wc_s = select(0.0, alpha_s * q, has_cur);
        let wh_s = (1.0 - alpha_s) * h.c;
        let den_e = max(wc_e + wh_e, 1e-5);
        let den_s = max(wc_s + wh_s, 1e-5);
        ea = (wc_e * cur + wh_e * h.ea) / den_e;
        s = (wc_s * cur_s + wh_s * h.s) / den_s;
        let c_new = 0.5 * ((wc_e * q + wh_e * h.c) / den_e + (wc_s * q + wh_s * h.c) / den_s);
        c_store = clamp(c_new, 0.0, 1.0) * params.decay;
        textureStore(h0_out, p, ea);
        textureStore(h1_out, p, vec4<f32>(s, z, c_store, 0.0));
    }
    var color = base.rgb * s + (kd.rgb * ea.rgb + s0) * ea.a + em;
    if (params.debug != 0u) {
        color = debug_color(params.debug, ea, s, c_store, h.c);
    }
    textureStore(out_color, p, vec4<f32>(color, 1.0));
}
