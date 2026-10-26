// HDR colour grading before tonemap (S12 `grading`): one fullscreen pass hdr.src() ->
// hdr.dst() applying, in the contract's order, white balance, exposure, log-space contrast
// around 0.18, ASC CDL lift/gamma/gain in log space, split toning by luminance, luma-
// preserving saturation then vibrance, and the vignette (cos^4 natural falloff x artistic
// amount). Output stays scene-referred HDR; negative or NaN values are clamped to 0.
// Standalone: own bindings in group 0, no PRELUDE.

struct GradingParams {
    // CAT02 white balance matrix (linear Rec.709), columns, w unused
    wb_col0: vec4<f32>,
    wb_col1: vec4<f32>,
    wb_col2: vec4<f32>,
    // (2^ev, contrast, saturation, vibrance)
    exposure_contrast: vec4<f32>,
    // ASC CDL offset in the log encoding, rgb (w unused)
    lift: vec4<f32>,
    // > 0, power 1 / gamma in the log encoding, rgb (w unused)
    gamma: vec4<f32>,
    // ASC CDL slope in the log encoding, rgb (w unused)
    gain: vec4<f32>,
    // soft-light tint of the shadows, 0.5 grey = none (w unused)
    shadows_tint: vec4<f32>,
    // soft-light tint of the highlights, 0.5 grey = none (w unused)
    highlights_tint: vec4<f32>,
    // (split_balance, vignette amount, viewport aspect w / h, 0)
    misc: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: GradingParams;
// hdr.src()
@group(0) @binding(1) var src_tex: texture_2d<f32>;
// Linear filtering, clamp to edge.
@group(0) @binding(2) var src_samp: sampler;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    // Texture coordinates of the target, (0, 0) top-left.
    @location(0) uv: vec2<f32>,
}

// Fullscreen triangle (3 vertices, no buffers).
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VsOut {
    let p = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: VsOut;
    out.position = vec4<f32>(p * 2.0 - vec2<f32>(1.0), 0.0, 1.0);
    out.uv = vec2<f32>(p.x, 1.0 - p.y);
    return out;
}

const MID_GREY: f32 = 0.18;
// Stops covered by the [0, 1] log encoding, centred on MID_GREY (0.5 = mid grey).
const LOG_STOPS: f32 = 16.0;
// Smallest value taken to log2 (about -27 stops below mid grey).
const LOG_FLOOR: f32 = 1e-9;

fn grading_luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// 0 for negative or NaN components (a NaN fails every comparison).
fn grading_sanitize(c: vec3<f32>) -> vec3<f32> {
    return select(vec3<f32>(0.0), c, c >= vec3<f32>(0.0));
}

// Stops relative to mid grey.
fn grading_stops(c: vec3<f32>) -> vec3<f32> {
    return log2(max(c, vec3<f32>(LOG_FLOOR)) / MID_GREY);
}

// Sign-preserving power, so the CDL leaves values outside [0, 1] of the encoding intact.
fn grading_spow(v: vec3<f32>, p: vec3<f32>) -> vec3<f32> {
    return sign(v) * pow(abs(v), p);
}

// Soft light (Pegtop): b = 0.5 returns a unchanged.
fn grading_soft_light(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return (vec3<f32>(1.0) - 2.0 * b) * a * a + 2.0 * b * a;
}

@fragment
fn fs_grading(in: VsOut) -> @location(0) vec4<f32> {
    let src = textureSampleLevel(src_tex, src_samp, in.uv, 0.0);
    var c = grading_sanitize(src.rgb);

    // (1) white balance.
    let wb = mat3x3<f32>(params.wb_col0.xyz, params.wb_col1.xyz, params.wb_col2.xyz);
    c = grading_sanitize(wb * c);

    // (2) exposure, 2^ev.
    c = c * params.exposure_contrast.x;

    // (3) contrast in log2 around mid grey.
    let stops = grading_stops(c) * params.exposure_contrast.y;

    // (4) ASC CDL in the log encoding: (l * gain + lift)^(1 / gamma).
    var l = stops / LOG_STOPS + vec3<f32>(0.5);
    l = l * params.gain.xyz + params.lift.xyz;
    l = grading_spow(l, vec3<f32>(1.0) / params.gamma.xyz);

    // (5) split toning: the log luminance against the split_balance pivot blends the
    // shadow and highlight tints, applied by soft light in the log encoding.
    let lum = grading_luma(MID_GREY * exp2((l - vec3<f32>(0.5)) * LOG_STOPS));
    let lum_l = log2(max(lum, LOG_FLOOR) / MID_GREY) / LOG_STOPS + 0.5;
    let h = smoothstep(0.0, 1.0, clamp(lum_l + 0.5 * params.misc.x, 0.0, 1.0));
    let tint = mix(params.shadows_tint.xyz, params.highlights_tint.xyz, h);
    l = grading_soft_light(l, tint);
    c = MID_GREY * exp2((l - vec3<f32>(0.5)) * LOG_STOPS);

    // (6) luma-preserving saturation, then vibrance boosting low-saturation pixels more.
    let y = grading_luma(c);
    c = vec3<f32>(y) + (c - vec3<f32>(y)) * params.exposure_contrast.z;
    let hi = max(c.r, max(c.g, c.b));
    let lo = min(c.r, min(c.g, c.b));
    let sat = clamp((hi - lo) / max(hi, 1e-6), 0.0, 1.0);
    let y2 = grading_luma(c);
    c = vec3<f32>(y2) + (c - vec3<f32>(y2)) * (1.0 + params.exposure_contrast.w * (1.0 - sat));

    // (7) vignette: cos^4 of the angle from the image centre (half image height = tan 1),
    // aspect-corrected, scaled by the artistic amount.
    let d = (in.uv - vec2<f32>(0.5)) * vec2<f32>(params.misc.z, 1.0) * 2.0;
    let cos2 = 1.0 / (1.0 + dot(d, d));
    c = c * mix(1.0, cos2 * cos2, params.misc.y);

    return vec4<f32>(grading_sanitize(c), src.a);
}
