// ao.gtao denoise: two separable 7-tap depth-aware bilateral passes over the raw GTAO.
// Standalone compute module (no PRELUDE), own bindings in group 0. `denoise_h` reads the
// source AO and writes the horizontal result into an R32Float texture; `denoise_v` reads
// that (bound as the same `src_ao` in the second dispatch) and writes the final texture,
// whose AO is in `.r` (R16Float is not a core WebGPU storage format). The tap weight is a
// Gaussian times `exp(-|dz| / (0.02 * z))` on the linear view depth, reconstructed from
// depth through the same `inv_proj`; the weights are normalised.

struct GtaoParams {
    proj: mat4x4<f32>,     // @0
    inv_proj: mat4x4<f32>, // @64
    viewport: vec4<f32>,   // @128 (w, h, 1/w, 1/h)
    params: vec4<f32>,     // @144 (radius_m, intensity, power, frame index)
    quality: vec4<f32>,    // @160 (slices, steps per side, pixels per texel, bilateral)
    phase: vec4<f32>,      // @176 (px, py, 1 = phase mode, 0): forward "layers-interlaced"
}

@group(0) @binding(0) var src_ao: texture_2d<f32>;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var out_h: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var out_v: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> gtao_params: GtaoParams;

const TAPS: u32 = 7u;
// Normalised Gaussian, sigma = 2 (sum = 1).
const GAUSS: array<f32, 7> = array<f32, 7>(
    0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125,
);

// View-space position of screen uv (origin top-left) at clip depth d.
fn raa_gtao_view_pos(uv: vec2<f32>, d: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, d, 1.0);
    let v = gtao_params.inv_proj * ndc;
    return v.xyz / v.w;
}

// Positive linear view depth of pixel `coord` at clip depth `d`.
fn raa_gtao_depth(coord: vec2<i32>, d: f32) -> f32 {
    let uv = (vec2<f32>(coord) + vec2<f32>(0.5)) * gtao_params.viewport.zw;
    return -raa_gtao_view_pos(uv, d).z;
}

// Pass-scale study: quality.z pixels per AO texel side (0/1 = full resolution); a texel stands
// for the centre pixel of its block (the phase pixel in phase mode, see raa_gtao_offset).
fn raa_gtao_factor() -> u32 {
    return max(u32(gtao_params.quality.z), 1u);
}

fn raa_gtao_low_size() -> vec2<u32> {
    let f = raa_gtao_factor();
    return (vec2<u32>(gtao_params.viewport.xy) + vec2<u32>(f - 1u)) / f;
}

// The pixel offset of a texel inside its block: the centre, or the phase pixel (px, py) in phase
// mode (forward "layers-interlaced", factor 2).
fn raa_gtao_offset() -> vec2<u32> {
    if (gtao_params.phase.z > 0.5) {
        return vec2<u32>(gtao_params.phase.xy);
    }
    return vec2<u32>(raa_gtao_factor() / 2u);
}

fn raa_gtao_full(texel: vec2<i32>) -> vec2<i32> {
    let f = raa_gtao_factor();
    let full = vec2<u32>(gtao_params.viewport.xy) - vec2<u32>(1u);
    return vec2<i32>(min(vec2<u32>(texel) * f + raa_gtao_offset(), full));
}

// Bilateral 7-tap filter of the source AO along `axis` (x or y), over AO texels.
fn raa_gtao_denoise(coord: vec2<i32>, axis: vec2<i32>) -> f32 {
    let max_px = vec2<i32>(raa_gtao_low_size()) - vec2<i32>(1);
    let c = raa_gtao_full(coord);
    let z = raa_gtao_depth(c, textureLoad(depth_tex, c, 0));
    var sum = 0.0;
    var wsum = 0.0;
    for (var i = 0u; i < TAPS; i = i + 1u) {
        let sp = clamp(coord + (i32(i) - 3) * axis, vec2<i32>(0), max_px);
        let fp = raa_gtao_full(sp);
        let sz = raa_gtao_depth(fp, textureLoad(depth_tex, fp, 0));
        let w = GAUSS[i] * exp(-abs(sz - z) / (0.02 * max(z, 1e-4)));
        sum = sum + textureLoad(src_ao, sp, 0).r * w;
        wsum = wsum + w;
    }
    return sum / max(wsum, 1e-6);
}

@compute @workgroup_size(8, 8)
fn denoise_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = raa_gtao_low_size();
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }
    let coord = vec2<i32>(gid.xy);
    textureStore(out_h, coord, vec4<f32>(raa_gtao_denoise(coord, vec2<i32>(1, 0))));
}

@compute @workgroup_size(8, 8)
fn denoise_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = raa_gtao_low_size();
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }
    let coord = vec2<i32>(gid.xy);
    let ao = raa_gtao_denoise(coord, vec2<i32>(0, 1));
    textureStore(out_v, coord, vec4<f32>(ao, 0.0, 0.0, 1.0));
}

// Full-resolution reconstruction of the low-resolution AO (factor > 1): the four nearest AO
// texels with bilinear weights, times the denoise's depth weight when quality.w is 1
// (joint-bilateral upsampling); sky pixels are unoccluded.
@compute @workgroup_size(8, 8)
fn upsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(gtao_params.viewport.xy);
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }
    let coord = vec2<i32>(gid.xy);
    let d = textureLoad(depth_tex, coord, 0);
    if (d >= 1.0) {
        textureStore(out_v, coord, vec4<f32>(1.0, 0.0, 0.0, 1.0));
        return;
    }
    let f = raa_gtao_factor();
    let z = raa_gtao_depth(coord, d);
    let at = (vec2<f32>(gid.xy) - vec2<f32>(raa_gtao_offset())) / f32(f);
    let corner = floor(at);
    let t = at - corner;
    let max_px = vec2<i32>(raa_gtao_low_size()) - vec2<i32>(1);
    // Phase mode: the phase pixel of a block reads its own texel exactly (no reconstruction),
    // so the forward's signal at that pixel is the AO computed for it.
    if (gtao_params.phase.z > 0.5 && all(t == vec2<f32>(0.0))) {
        let own = clamp(vec2<i32>(corner), vec2<i32>(0), max_px);
        textureStore(out_v, coord, vec4<f32>(textureLoad(src_ao, own, 0).r, 0.0, 0.0, 1.0));
        return;
    }
    let bilateral = gtao_params.quality.w > 0.5;
    var sum = 0.0;
    var wsum = 0.0;
    var lin = 0.0;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let o = vec2<i32>(i32(i & 1u), i32(i >> 1u));
        let texel = clamp(vec2<i32>(corner) + o, vec2<i32>(0), max_px);
        let b = select(1.0 - t.x, t.x, o.x == 1) * select(1.0 - t.y, t.y, o.y == 1);
        let v = textureLoad(src_ao, texel, 0).r;
        lin = lin + v * b;
        var w = b + 1e-4;
        if (bilateral) {
            let fp = raa_gtao_full(texel);
            let sd = textureLoad(depth_tex, fp, 0);
            let sz = select(raa_gtao_depth(fp, sd), 1e9, sd >= 1.0);
            w = w * exp(-abs(sz - z) / (0.02 * max(z, 1e-4)));
        }
        sum = sum + v * w;
        wsum = wsum + w;
    }
    // No depth-matching texel (thin feature lost at this resolution): bilinear.
    let ao = select(lin, sum / max(wsum, 1e-30), bilateral && wsum > 1e-6);
    textureStore(out_v, coord, vec4<f32>(ao, 0.0, 0.0, 1.0));
}
