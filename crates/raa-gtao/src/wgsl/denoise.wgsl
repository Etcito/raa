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

// Bilateral 7-tap filter of the source AO along `axis` (x or y).
fn raa_gtao_denoise(coord: vec2<i32>, axis: vec2<i32>) -> f32 {
    let max_px = vec2<i32>(gtao_params.viewport.xy) - vec2<i32>(1);
    let z = raa_gtao_depth(coord, textureLoad(depth_tex, coord, 0));
    var sum = 0.0;
    var wsum = 0.0;
    for (var i = 0u; i < TAPS; i = i + 1u) {
        let sp = clamp(coord + (i32(i) - 3) * axis, vec2<i32>(0), max_px);
        let sz = raa_gtao_depth(sp, textureLoad(depth_tex, sp, 0));
        let w = GAUSS[i] * exp(-abs(sz - z) / (0.02 * max(z, 1e-4)));
        sum = sum + textureLoad(src_ao, sp, 0).r * w;
        wsum = wsum + w;
    }
    return sum / max(wsum, 1e-6);
}

@compute @workgroup_size(8, 8)
fn denoise_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(gtao_params.viewport.xy);
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }
    let coord = vec2<i32>(gid.xy);
    textureStore(out_h, coord, vec4<f32>(raa_gtao_denoise(coord, vec2<i32>(1, 0))));
}

@compute @workgroup_size(8, 8)
fn denoise_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(gtao_params.viewport.xy);
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }
    let coord = vec2<i32>(gid.xy);
    let ao = raa_gtao_denoise(coord, vec2<i32>(0, 1));
    textureStore(out_v, coord, vec4<f32>(ao, 0.0, 0.0, 1.0));
}
