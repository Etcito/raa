// ReSTIR GI, stage 3 (denoise) of the screen part's `restir` mode: a depth- and normal-aware
// bilateral filter over the spatial stage's irradiance (5 x 5 taps at a 2-pixel stride), then
// the screen part's temporal accumulation: the same history test, bilinear history read and
// screen occlusion as the probe integrate (probe_integrate.wgsl, composed ahead of this file:
// bindings 3 motion, 4 previous output, 6 output, 7 GiScreenGpu), blended at
// RAA_RGI_TEMPORAL_ALPHA. The output (irradiance rgb, occlusion a) is the gi.screen output that
// raa_slot_indirect reads at binding 14 and that the ssgi composite takes as its probe input.
// RAA's own code.

@group(2) @binding(14) var raa_rgi_raw_in: texture_2d<f32>;

// The temporal weight of the new irradiance. The reservoirs already carry up to m_cap frames,
// so the accumulation only has to remove the residual per-pixel noise.
const RAA_RGI_TEMPORAL_ALPHA: f32 = 0.12;

@compute @workgroup_size(8, 8)
fn raa_gi_restir_gi_denoise(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<u32>(raa_gi_frame.viewport.xy);
    if any(id.xy >= size) {
        return;
    }
    let pixel = vec2<i32>(id.xy);
    let depth = textureLoad(raa_gi_depth, pixel, 0);
    if raa_depth_is_far(depth) {
        textureStore(raa_gi_output, pixel, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }
    let center = vec2<f32>(id.xy) + vec2<f32>(0.5);
    let normal = raa_gi_world_normal(pixel);
    let world = raa_gi_world_position(center, depth);
    let view_z = -raa_gi_view_position(center, depth).z;
    let last = vec2<i32>(size) - vec2<i32>(1);

    var sum = vec3<f32>(0.0);
    var total = 0.0;
    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let q = clamp(pixel + vec2<i32>(dx, dy) * 2, vec2<i32>(0), last);
            let dq = textureLoad(raa_gi_depth, q, 0);
            if raa_depth_is_far(dq) {
                continue;
            }
            let zq = -raa_gi_view_position(vec2<f32>(q) + vec2<f32>(0.5), dq).z;
            let depth_w = exp(-abs(zq - view_z) / max(0.02 * view_z, 1.0e-3));
            let facing = max(dot(raa_gi_world_normal(q), normal), 0.0);
            let facing2 = facing * facing;
            let normal_w = facing2 * facing2 * facing2 * facing2;
            let spatial_w = exp(-f32(dx * dx + dy * dy) / 4.5);
            let w = depth_w * normal_w * spatial_w;
            sum += textureLoad(raa_rgi_raw_in, q, 0).rgb * w;
            total += w;
        }
    }
    let irradiance = select(vec3<f32>(0.0), sum / total, total > 1.0e-6);

    let noise = raa_gi_probe_ign(vec2<f32>(id.xy), raa_gi_frame.counts.y);
    let occlusion = raa_gi_screen_occlusion(center, world, normal, noise);
    var current = vec4<f32>(irradiance, occlusion);

    let m = textureLoad(raa_gi_motion, pixel, 0);
    let own_px = m.xy * raa_gi_frame.viewport.xy;
    let previous_uv = center * raa_gi_frame.viewport.zw - m.xy;
    if m.z <= 0.5
        && raa_gi_screen_params.history_valid != 0u
        && all(previous_uv > vec2<f32>(0.0))
        && all(previous_uv < vec2<f32>(1.0))
        && raa_gi_history_neighbours_agree(pixel, own_px)
    {
        current = mix(raa_gi_history_bilinear(previous_uv), current, RAA_RGI_TEMPORAL_ALPHA);
    }
    textureStore(raa_gi_output, pixel, current);
}
