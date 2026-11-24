// P4 M8 screen-space indirect diffuse: the temporal resolve of `gi.ssgi`. It filters the raw
// trace with a depth- and normal-aware cross, accumulates it in a raa-taa history pair reprojected through
// the S12 motion target (gated by FrameGpu temporal.z), and writes the composite with the probe
// irradiance, `cpu::compose`: (E_ss + V x E_probe, probe a). Composed after
// raa_taa::REPROJECT_WGSL (RaaTemporalFrame, raa_temporal_surface_uv). RAA's own code.
//
// The history blends at 0.1 while it agrees with this frame; a history further than
// RAA_SSGI_CHANGE_SIGMA standard deviations of the filtered taps from their mean (a light that
// went out or moved, an occluder) blends faster, at weight 1 from twice that (`cpu::change_alpha`),
// so the screen-space bounce follows a lighting change within a frame instead of fading over ~20.

// The per-frame parameters (32 B), declared as trace.wgsl does.
struct SsgiParamsGpu {
    slices: u32,
    steps: u32,
    history_valid: u32,
    frame: u32,
    radius_m: f32,
    thickness_m: f32,
    temporal_valid: u32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> raa_ssgi_frame: RaaTemporalFrame;
@group(0) @binding(1) var raa_ssgi_depth: texture_depth_2d;
@group(0) @binding(2) var raa_ssgi_motion: texture_2d<f32>;
@group(0) @binding(3) var raa_ssgi_raw: texture_2d<f32>;
@group(0) @binding(4) var raa_ssgi_history: texture_2d<f32>;
@group(0) @binding(5) var raa_ssgi_probe: texture_2d<f32>;
@group(0) @binding(6) var raa_ssgi_history_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var raa_ssgi_output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(8) var<uniform> raa_ssgi_params: SsgiParamsGpu;
@group(0) @binding(9) var raa_ssgi_normal: texture_2d<f32>;

// The width of the change test in standard deviations of the filtered taps (`cpu::CHANGE_SIGMA`).
override RAA_SSGI_CHANGE_SIGMA: f32 = 3.0;

// The history weight of the temporal blend (`cpu::change_alpha`): 0.1 while the history's
// luminance lies within RAA_SSGI_CHANGE_SIGMA sigma (+5 % slack) of the taps' mean, rising
// linearly to 1 at twice that.
fn raa_ssgi_change_alpha(history: f32, mean: f32, variance: f32) -> f32 {
    let slack = 0.05 * max(abs(history), abs(mean)) + 1.0e-4;
    let z = abs(history - mean) / (sqrt(max(variance, 0.0)) + slack);
    let cut = clamp(z / RAA_SSGI_CHANGE_SIGMA - 1.0, 0.0, 1.0);
    return mix(0.1, 1.0, cut);
}

fn raa_ssgi_luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// The positive view depth of pixel p at standard-Z depth d, through inv_proj.
fn raa_ssgi_view_depth(p: vec2<i32>, d: f32, size: vec2<f32>) -> f32 {
    let uv = (vec2<f32>(p) + 0.5) / size;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, d, 1.0);
    let view = raa_ssgi_frame.inv_proj * ndc;
    return abs(view.z / view.w);
}

@compute @workgroup_size(8, 8)
fn raa_gi_ssgi_resolve(@builtin(global_invocation_id) id: vec3<u32>) {
    let size_u = vec2<u32>(raa_ssgi_frame.viewport.xy);
    if id.x >= size_u.x || id.y >= size_u.y {
        return;
    }
    let p = vec2<i32>(id.xy);
    let size = vec2<f32>(size_u);
    let depth = textureLoad(raa_ssgi_depth, p, 0);
    if depth >= 1.0 {
        textureStore(raa_ssgi_history_out, p, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        textureStore(raa_ssgi_output, p, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    let centre_z = raa_ssgi_view_depth(p, depth, size);
    let normal = normalize(textureLoad(raa_ssgi_normal, p, 0).xyz);
    let last = vec2<i32>(size_u) - vec2<i32>(1);
    var sum = textureLoad(raa_ssgi_raw, p, 0);
    var weight = 1.0;
    let l0 = raa_ssgi_luminance(sum.rgb);
    var l1 = l0;
    var l2 = l0 * l0;
    let taps = array<vec2<i32>, 4>(
        vec2<i32>(2, 0),
        vec2<i32>(-2, 0),
        vec2<i32>(0, 2),
        vec2<i32>(0, -2),
    );
    for (var i = 0u; i < 4u; i = i + 1u) {
        let q = clamp(p + taps[i], vec2<i32>(0), last);
        let dq = textureLoad(raa_ssgi_depth, q, 0);
        if dq < 1.0 && abs(raa_ssgi_view_depth(q, dq, size) - centre_z) <= 0.05 * centre_z
            && dot(normal, normalize(textureLoad(raa_ssgi_normal, q, 0).xyz)) > 0.8 {
            let tap = textureLoad(raa_ssgi_raw, q, 0);
            sum = sum + tap;
            weight = weight + 1.0;
            let l = raa_ssgi_luminance(tap.rgb);
            l1 = l1 + l;
            l2 = l2 + l * l;
        }
    }
    let current = sum / weight;

    let uv = (vec2<f32>(p) + 0.5) / size;
    let motion = textureLoad(raa_ssgi_motion, p, 0);
    let r = raa_temporal_surface_uv(uv, motion);
    var acc = current;
    if r.z == 1.0 && raa_ssgi_params.temporal_valid != 0u && raa_ssgi_frame.temporal.z != 0u {
        let hp = clamp(vec2<i32>(r.xy * size), vec2<i32>(0), last);
        let prev = textureLoad(raa_ssgi_history, hp, 0);
        // cpu::temporal_alpha: a fixed weight, so a moving camera keeps converging; the change
        // test raises it when the history no longer matches this frame's light.
        let mean = l1 / weight;
        let alpha = raa_ssgi_change_alpha(raa_ssgi_luminance(prev.rgb), mean, l2 / weight - mean * mean);
        acc = mix(prev, current, alpha);
    }

    textureStore(raa_ssgi_history_out, p, acc);
    // The probe can be a 1x1 dummy (0, 0, 0, 1) in a standalone graph stage.
    let probe_pixel = min(p, vec2<i32>(textureDimensions(raa_ssgi_probe)) - vec2<i32>(1));
    let probe = textureLoad(raa_ssgi_probe, probe_pixel, 0);
    // Coverage picks the irradiance source. Ambient occlusion comes from the probe's alpha.
    textureStore(raa_ssgi_output, p, vec4<f32>(acc.rgb + clamp(acc.a, 0.0, 1.0) * probe.rgb, probe.a));
}
