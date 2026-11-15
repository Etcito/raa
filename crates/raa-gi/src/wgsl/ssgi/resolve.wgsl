// P4 M8 screen-space indirect diffuse: the temporal resolve of `gi.ssgi`. It filters the raw
// trace with a depth-aware cross, accumulates it in a raa-taa history pair reprojected through
// the S12 motion target (gated by FrameGpu temporal.z), and writes the composite with the probe
// irradiance, `cpu::compose`: (E_ss + V x E_probe, min(V, probe a)). Composed after
// raa_taa::REPROJECT_WGSL (RaaTemporalFrame, raa_temporal_surface_uv). RAA's own code.

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
    let last = vec2<i32>(size_u) - vec2<i32>(1);
    var sum = textureLoad(raa_ssgi_raw, p, 0);
    var weight = 1.0;
    let taps = array<vec2<i32>, 4>(
        vec2<i32>(2, 0),
        vec2<i32>(-2, 0),
        vec2<i32>(0, 2),
        vec2<i32>(0, -2),
    );
    for (var i = 0u; i < 4u; i = i + 1u) {
        let q = clamp(p + taps[i], vec2<i32>(0), last);
        let dq = textureLoad(raa_ssgi_depth, q, 0);
        if dq < 1.0 && abs(raa_ssgi_view_depth(q, dq, size) - centre_z) <= 0.05 * centre_z {
            sum = sum + textureLoad(raa_ssgi_raw, q, 0);
            weight = weight + 1.0;
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
        // cpu::temporal_alpha
        let alpha = min(0.1 + 0.03 * length(motion.xy * size), 0.5);
        acc = mix(prev, current, alpha);
    }

    textureStore(raa_ssgi_history_out, p, acc);
    let probe = textureLoad(raa_ssgi_probe, p, 0);
    textureStore(raa_ssgi_output, p, vec4<f32>(acc.rgb + acc.a * probe.rgb, min(acc.a, probe.a)));
}
