// ReSTIR GI, stage 3 (denoise) of the screen part's `restir` mode: a depth- and normal-aware
// bilateral filter over the spatial stage's irradiance (5 x 5 taps at a 2-pixel stride), whose
// weighted luminance mean and variance are the clamp's neighbourhood, then the screen part's
// temporal accumulation (accumulate.wgsl, composed ahead of this file: the disocclusion test,
// the clamp, the 1 / n blend and every record). The a-trous passes (atrous.wgsl) run after it.
// The output (irradiance rgb, occlusion a) is the gi.screen history; the last a-trous pass
// writes what raa_slot_indirect reads at binding 14. RAA's own code.

@group(2) @binding(14) var raa_rgi_raw_in: texture_2d<f32>;

@compute @workgroup_size(8, 8)
fn raa_gi_restir_gi_denoise(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<u32>(raa_gi_frame.viewport.xy);
    if any(id.xy >= size) {
        return;
    }
    let pixel = vec2<i32>(id.xy);
    let depth = textureLoad(raa_gi_depth, pixel, 0);
    if raa_depth_is_far(depth) {
        raa_gi_acc_store_far(pixel);
        return;
    }
    let center = vec2<f32>(id.xy) + vec2<f32>(0.5);
    let normal = raa_gi_world_normal(pixel);
    let world = raa_gi_world_position(center, depth);
    let view_z = -raa_gi_view_position(center, depth).z;
    let last = vec2<i32>(size) - vec2<i32>(1);

    var sum = vec3<f32>(0.0);
    var total = 0.0;
    var l1 = 0.0;
    var l2 = 0.0;
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
            let e = textureLoad(raa_rgi_raw_in, q, 0).rgb;
            let l = raa_gi_acc_luminance(e);
            sum += e * w;
            l1 += l * w;
            l2 += l * l * w;
            total += w;
        }
    }
    var irradiance = vec3<f32>(0.0);
    var mean = 0.0;
    var spread = 0.0;
    if total > 1.0e-6 {
        irradiance = sum / total;
        mean = l1 / total;
        spread = max(l2 / total - mean * mean, 0.0);
    }

    let noise = raa_gi_probe_ign(vec2<f32>(id.xy), raa_gi_frame.counts.y);
    let occlusion = raa_gi_screen_occlusion(center, world, normal, noise);

    // raw.a = M + 256 x (1 for a sky sample): the kept reservoir of the spatial stage.
    let tag = textureLoad(raa_rgi_raw_in, pixel, 0).a;
    let sky = tag >= 256.0;
    let confidence = tag - select(0.0, 256.0, sky);
    let hit_code = select(0x0F0u, 0x00Fu, sky);

    raa_gi_acc_resolve(RaaGiAccInput(
        pixel,
        world,
        normal,
        view_z,
        vec4<f32>(irradiance, occlusion),
        true,
        mean,
        spread,
        hit_code,
        confidence,
    ));
}
