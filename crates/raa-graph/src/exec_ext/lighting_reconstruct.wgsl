// pass.lighting.reconstruct: full-resolution accumulation and composition of the forward
// lightingOutput "layers-interlaced" signals. Exact copy of nt_model.step (neural-target) with
// the 3x3 predicted kernel replaced by identity and the four alphas by constants (the analytic
// ablation), then nt_model.composite. One dispatch per frame, workgroup 8x8.
//
// Signals: E = layerE.rgb (probe irradiance already x w, present when layerE.a = w >= 0), a =
// layerKd.a (present when >= 0), s = color.a (present when >= 0). Where and when the forward
// samples them is set by the signal rates (raa-pipeline InterlaceSignals, flags below): by
// default all three at one phase pixel per 2x2 block on R frames; s may be sampled at every
// pixel of R frames (SHADOW_REAL_FULL) or of every frame (SHADOW_EXACT), E at the phase pixel of
// G frames too (E_SAMPLED_G, `params.phase` is then the G frame's phase) or at every pixel of
// every frame (E_EXACT). AO stays at the R-frame phase pixel.
// History (ping-pong, Rgba16Float): H0 = (E rgb, cE), H1 = (s, linear view depth z, c, a). z is
// the linear view depth in metres from the prepass depth through the jittered inv_proj of
// FrameGpu (0 = sky / no geometry). c is the confidence of s and a, and of E while E is sampled
// with a (default rate); cE the confidence of E otherwise (E_SEPARATE).
//
// Per pixel p: reproject q = p + 0.5 - motion.xy * size + jitter_px (jitter_px = (previous -
// current) camera jitter NDC x (0.5 W, -0.5 H) when jitterCorrect, else 0); bilinear H0/H1 at q;
// valid = inside [0, size] && motion.z < 0.5 && |z_hist - z| <= 0.05 z + 0.02 && z > 0 &&
// history valid; hc = valid ? c_hist : 0 (per confidence).
// A group sampled this frame takes the guided current estimate from the phase pixels: per axis
// the taps are {p} when p has the phase parity, else {p - 1, p + 1} (the phase pixels of the two
// neighbouring blocks); weight b x (guide_w + 1e-4) with b = 1 / tap count, 0 for a tap outside
// the image or without the group's sample. q = 1 and the hit alphas at the phase pixel, qMiss
// and the miss alphas elsewhere. Blend: wc = alpha q (0 without any tap sample), wh = (1 -
// alpha) hc, out = (wc cur + wh hist) / max(wc + wh, 1e-5); confidence (wc q + wh hc) / max(wc +
// wh, 1e-5), c = the mean over its sampled groups, clamped to [0, 1] x decay. An exact group (s
// or E at every pixel this frame) takes the pixel's own sample. A group not sampled this frame
// keeps the reprojected history; on a G frame a hole (hc <= 0) takes the guide x hc weighted mean
// of the 12 RING taps, each reprojected with its own motion and depth (kept bilinear when the
// weight sum <= 1e-4); the fill is not stored (history, z, hc x decay).
// Composite: C = d0 s + (kd E + s0 vis) a + em; sky (z = 0): C = em, history zero. vis is the
// DDGI sky-specular visibility of raa-shader opaque_layers.wgsl (raa_layers_sky_vis) from the
// layerS0 alpha t of the current pixel and the accumulated (E, a) with w = 1 (the history keeps
// no probe weight; the layers assume w = 1 whenever a probe GI is enabled).

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
// s is present at every pixel of an R frame (ShadowRate::Real).
const SHADOW_REAL_FULL: u32 = 4u;
// s is present at every pixel of this frame (ShadowRate::Every): taken as is.
const SHADOW_EXACT: u32 = 8u;
// This G frame sampled E at its phase pixels (IrradianceRate::Phase60).
const E_SAMPLED_G: u32 = 16u;
// E is present at every pixel of this frame (IrradianceRate::Full): taken as is.
const E_EXACT: u32 = 32u;
// E is not sampled with a (rate other than the default): its own confidence cE.
const E_SEPARATE: u32 = 64u;

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
    e: vec3<f32>,
    a: f32,
    s: f32,
    // Confidence of s and a (and E without E_SEPARATE), of E with E_SEPARATE.
    c: f32,
    ce: f32,
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
    let c = select(0.0, h1.z, valid);
    let ce = select(c, select(0.0, h0.a, valid), (params.flags & E_SEPARATE) != 0u);
    return History(h0.rgb, h1.w, h1.x, c, ce);
}

// One blended group: value, confidence before decay.
struct Blend {
    v: vec4<f32>,
    c: f32,
}

// wc = alpha q (0 without a current estimate), wh = (1 - alpha) hc.
fn blend(cur: vec4<f32>, has: bool, q: f32, alpha: f32, hist: vec4<f32>, hc: f32) -> Blend {
    let wc = select(0.0, alpha * q, has);
    let wh = (1.0 - alpha) * hc;
    let den = max(wc + wh, 1e-5);
    return Blend((wc * cur + wh * hist) / den, (wc * q + wh * hc) / den);
}

// raa-shader opaque_layers.wgsl raa_layers_sky_vis.
fn sky_vis(t: f32, e: vec3<f32>, w: f32, a: f32) -> f32 {
    if (t == 0.0 || w <= 0.0) {
        return 1.0;
    }
    let luma = vec3<f32>(0.2126, 0.7152, 0.0722);
    let l = dot(e, luma) / w * select(1.0, a, t > 0.0);
    return mix(1.0, clamp(l / abs(t), 0.0, 1.0), w);
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
    let s0 = textureLoad(s0_tex, p, 0);
    let em = textureLoad(em_tex, p, 0).rgb;
    let z = linear_z(p);
    if (z <= 0.0) {
        textureStore(h0_out, p, vec4<f32>(0.0));
        textureStore(h1_out, p, vec4<f32>(0.0));
        textureStore(out_color, p, vec4<f32>(select(vec3<f32>(0.0), em, params.debug == 0u), 1.0));
        return;
    }
    let flags = params.flags;
    let real = (flags & REAL_FRAME) != 0u;
    let separate = (flags & E_SEPARATE) != 0u;
    let n = view_normal(p);
    let h = reproject(p, z);
    let e_own = textureLoad(e_tex, p, 0);
    // Exact groups: the pixel's own sample (the forward wrote it at every pixel this frame).
    let s_exact = (flags & SHADOW_EXACT) != 0u && base.a >= 0.0;
    let e_exact = (flags & E_EXACT) != 0u && e_own.a >= 0.0;
    // Groups whose phase-pixel samples this frame holds.
    let sample_sa = real;
    let sample_e = !e_exact && (real || (flags & E_SAMPLED_G) != 0u);

    var e = h.e;
    var a = h.a;
    var s = h.s;
    var c_new = h.c;
    var ce_new = h.ce;
    if (sample_sa || sample_e) {
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
        var acc_e = vec3<f32>(0.0);
        var acc_a = 0.0;
        var acc_s = 0.0;
        var ws_e = 0.0;
        var ws_a = 0.0;
        var ws_s = 0.0;
        for (var j = 0; j < ny; j = j + 1) {
            for (var i = 0; i < nx; i = i + 1) {
                let t = vec2<i32>(xs[i], ys[j]);
                if (any(t < vec2<i32>(0)) || any(t > max_p)) {
                    continue;
                }
                let st = textureLoad(color_tex, t, 0).a;
                let at = textureLoad(kd_tex, t, 0).a;
                let et = textureLoad(e_tex, t, 0);
                let has_e = sample_e && et.a >= 0.0;
                let has_a = sample_sa && at >= 0.0;
                let has_s = sample_sa && st >= 0.0;
                if (!(has_e || has_a || has_s)) {
                    continue;
                }
                let w = b * (guide_w(z, n, linear_z(t), view_normal(t)) + 1e-4);
                if (has_e) {
                    acc_e = acc_e + w * et.rgb;
                    ws_e = ws_e + w;
                }
                if (has_a) {
                    acc_a = acc_a + w * at;
                    ws_a = ws_a + w;
                }
                if (has_s) {
                    acc_s = acc_s + w * st;
                    ws_s = ws_s + w;
                }
            }
        }
        let q = select(params.q_miss, 1.0, own);
        let alpha_e = select(params.alpha.w, params.alpha.z, own);
        if (sample_e) {
            let cur = select(vec3<f32>(0.0), acc_e / max(ws_e, 1e-30), ws_e > 0.0);
            let r = blend(vec4<f32>(cur, 0.0), ws_e > 0.0, q, alpha_e, vec4<f32>(h.e, 0.0), h.ce);
            e = r.v.rgb;
            ce_new = r.c;
        }
        if (sample_sa) {
            let cur_a = select(0.0, acc_a / max(ws_a, 1e-30), ws_a > 0.0);
            let ra = blend(vec4<f32>(cur_a), ws_a > 0.0, q, alpha_e, vec4<f32>(h.a), h.c);
            a = ra.v.x;
            // c: the mean over the sampled groups of c (s unless exact; (E, a) or a).
            var c_sum = ra.c;
            var c_n = 1.0;
            if (!separate && sample_e) {
                // Default rate: E and a share the taps and c (one (E, a) group).
                c_sum = ce_new;
            }
            if (!s_exact) {
                var cur_s = select(0.0, acc_s / max(ws_s, 1e-30), ws_s > 0.0);
                var q_s = q;
                var alpha_s = select(params.alpha.y, params.alpha.x, own);
                var has_s = ws_s > 0.0;
                if ((flags & SHADOW_REAL_FULL) != 0u && base.a >= 0.0) {
                    // s sampled at this pixel: exact, weighted like a phase-pixel sample.
                    cur_s = base.a;
                    q_s = 1.0;
                    alpha_s = params.alpha.x;
                    has_s = true;
                }
                let rs = blend(vec4<f32>(cur_s), has_s, q_s, alpha_s, vec4<f32>(h.s), h.c);
                s = rs.v.x;
                c_sum = c_sum + rs.c;
                c_n = 2.0;
            }
            c_new = c_sum / c_n;
            if (!separate) {
                ce_new = c_new;
            }
        }
    }
    if (s_exact) {
        s = base.a;
    }
    if (e_exact) {
        e = e_own.rgb;
        ce_new = 1.0;
    }
    let c_store = clamp(c_new, 0.0, 1.0) * params.decay;
    let ce_store = clamp(ce_new, 0.0, 1.0) * params.decay;
    textureStore(h0_out, p, vec4<f32>(e, ce_store));
    textureStore(h1_out, p, vec4<f32>(s, z, c_store, a));

    // G frame: holes of the groups not sampled this frame are filled from the ring (not stored).
    if (!real) {
        let fill_sa = h.c <= 0.0;
        let fill_s = fill_sa && !s_exact;
        let fill_e = !e_exact && !sample_e && h.ce <= 0.0;
        if (fill_sa || fill_e) {
            var ring = array<vec2<i32>, 12>(
                vec2<i32>(-2, 0), vec2<i32>(2, 0), vec2<i32>(0, -2), vec2<i32>(0, 2),
                vec2<i32>(-4, -4), vec2<i32>(4, -4), vec2<i32>(-4, 4), vec2<i32>(4, 4),
                vec2<i32>(-7, 0), vec2<i32>(7, 0), vec2<i32>(0, -7), vec2<i32>(0, 7),
            );
            var acc_e = vec3<f32>(0.0);
            var acc_a = 0.0;
            var acc_s = 0.0;
            var ws = 0.0;
            var ws_e = 0.0;
            for (var i = 0u; i < 12u; i = i + 1u) {
                let t = clamp(p + ring[i], vec2<i32>(0), max_p);
                let zt = linear_z(t);
                if (zt <= 0.0) {
                    continue;
                }
                let ht = reproject(t, zt);
                let g = guide_w(z, n, zt, view_normal(t));
                let w = g * ht.c;
                let we = g * ht.ce;
                acc_a = acc_a + w * ht.a;
                acc_s = acc_s + w * ht.s;
                ws = ws + w;
                acc_e = acc_e + we * ht.e;
                ws_e = ws_e + we;
            }
            if (fill_sa && ws > 1e-4) {
                a = acc_a / ws;
                if (fill_s) {
                    s = acc_s / ws;
                }
            }
            if (fill_e && ws_e > 1e-4) {
                e = acc_e / ws_e;
            }
        }
    }
    let vis = sky_vis(s0.a, e, 1.0, a);
    var color = base.rgb * s + (kd.rgb * e + s0.rgb * vis) * a + em;
    if (params.debug != 0u) {
        color = debug_color(params.debug, vec4<f32>(e, a), s, c_store, h.c);
    }
    textureStore(out_color, p, vec4<f32>(color, 1.0));
}
