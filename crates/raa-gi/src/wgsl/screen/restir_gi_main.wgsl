// ReSTIR GI, stage 1 (initial sample + temporal reuse) of the screen part's `restir` mode. Per
// pixel: one secondary ray from the visible point x_v along a cosine-weighted direction, traced
// first through the depth pyramid (an exact near hit lit by last frame's HDR) and then through
// the GI voxel distance field (radiance from the voxel lighting, normal from the field's
// gradient), the sky when it escapes. The sample forms a one-candidate reservoir (M = 1,
// W = 1 / pdf = pi / cos theta), which is merged with the previous frame's reservoir of the
// reprojected pixel (motion vectors, depth and normal rejection, M clamped to m_cap times the
// current M) as in Ouyang et al. 2021, Algorithm 3; a workgroup boiling filter drops outliers.
// The result goes to binding 11, next frame's binding 10.
//
// Composed after restir_gi_common.wgsl and screen_trace.wgsl (raa_gi_trace_screen,
// raa_gi_screen_hit_radiance: bindings 3 history, 4 pyramid). RAA's own code.

const RAA_RGI_SCREEN_DISTANCE: f32 = 8.0;
const RAA_RGI_SCREEN_STEPS: u32 = 32u;
const RAA_RGI_SCREEN_THICKNESS: f32 = 0.15;
const RAA_RGI_VOLUME_DISTANCE: f32 = 320.0;
const RAA_RGI_VOLUME_STEPS: u32 = 40u;
const RAA_RGI_VOLUME_HIT_VOXELS: f32 = 0.85;
const RAA_RGI_SALT_INITIAL: u32 = 0x68E31DA4u;
const RAA_RGI_SALT_TEMPORAL: u32 = 0xB5297A4Du;

var<workgroup> raa_rgi_boiling_q: array<f32, 64>;
var<workgroup> raa_rgi_boiling_mean: f32;

fn raa_rgi_seed(pixel: vec2<u32>, frame: u32, salt: u32) -> u32 {
    return raa_restir_pcg(pixel.x * 7919u + pixel.y * 104729u + frame * 31337u + salt);
}

// The unit normal of the distance field at `p` (central differences of half a voxel), or -dir
// where the field gives no gradient.
fn raa_rgi_field_normal(p: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    let level = raa_gi_level_at(p, 0u);
    if level >= raa_gi_volume_params.level_count {
        return -dir;
    }
    let h = 0.5 * raa_gi_volume_params.levels[level].w;
    let g = vec3<f32>(
        raa_gi_sample(p + vec3<f32>(h, 0.0, 0.0), level).a - raa_gi_sample(p - vec3<f32>(h, 0.0, 0.0), level).a,
        raa_gi_sample(p + vec3<f32>(0.0, h, 0.0), level).a - raa_gi_sample(p - vec3<f32>(0.0, h, 0.0), level).a,
        raa_gi_sample(p + vec3<f32>(0.0, 0.0, h), level).a - raa_gi_sample(p - vec3<f32>(0.0, 0.0, h), level).a,
    );
    let len = length(g);
    if !(len > 1.0e-6) {
        return -dir;
    }
    var n = g / len;
    // The radiance leaves toward the ray origin: a gradient across a thin wall's far side faces it.
    if dot(n, dir) > 0.0 {
        n = -n;
    }
    return n;
}

// The initial reservoir of a valid surface: one cosine-weighted ray, M = 1, W = 1 / pdf when the
// target is positive (GRIS with one candidate: w = p_hat / pdf, W = w / p_hat).
fn raa_rgi_initial(pixel: vec2<u32>, s: RaaRgiSurface) -> RaaRgiReservoir {
    let frame = raa_rgi_params.frame_index;
    let seed = raa_rgi_seed(pixel, frame, RAA_RGI_SALT_INITIAL);
    let uv = vec2<f32>(raa_restir_random(seed), raa_restir_random(seed + 1u));
    let dir = raa_rgi_cosine_direction(uv, s.normal);
    let cos_theta = max(dot(dir, s.normal), 1.0e-4);

    var x_s = s.position + dir * RAA_RGI_SKY_DISTANCE;
    var n_s = -dir;
    var radiance = textureSampleLevel(raa_env_specular, raa_linear_clamp, dir, 0.0).rgb;
    var found = false;
    if raa_rgi_params.screen_traces == 1u {
        let origin = s.position + s.normal * 0.03;
        let screen = raa_gi_trace_screen(origin, dir, RAA_RGI_SCREEN_DISTANCE, RAA_RGI_SCREEN_STEPS,
            RAA_RGI_SCREEN_THICKNESS);
        if screen.hit && screen.weight >= 1.0 {
            let hit = origin + dir * screen.t;
            let lit = raa_gi_screen_hit_radiance(hit);
            if lit.a == 1.0 {
                let size = vec2<i32>(raa_gi_frame.viewport.xy);
                let hit_pixel = clamp(vec2<i32>(screen.uv * raa_gi_frame.viewport.xy), vec2<i32>(0), size - vec2<i32>(1));
                x_s = hit;
                n_s = raa_gi_world_normal(hit_pixel);
                radiance = lit.rgb;
                found = true;
            }
        }
    }
    let level_count = raa_gi_volume_params.level_count;
    if !found && level_count > 0u {
        let level = min(raa_gi_level_at(s.position, 0u), level_count - 1u);
        let voxel = raa_gi_volume_params.levels[level].w;
        let origin = s.position + s.normal * voxel * 1.5;
        let hit = raa_gi_gather(origin, dir, voxel * 0.5, RAA_RGI_VOLUME_DISTANCE, RAA_RGI_VOLUME_STEPS,
            RAA_RGI_VOLUME_HIT_VOXELS);
        if hit.hit {
            x_s = origin + dir * hit.t;
            n_s = raa_rgi_field_normal(x_s, dir);
            radiance = hit.radiance;
        }
    }
    var r = raa_rgi_empty();
    r.position = x_s;
    r.normal = n_s;
    r.radiance = radiance;
    r.confidence = 1.0;
    r.owner_normal = s.normal;
    r.owner_depth = s.depth;
    let p_hat = raa_rgi_target(s.position, s.normal, x_s, n_s, radiance);
    let pdf = cos_theta / 3.14159265359;
    r.weight = raa_restir_contribution_weight(p_hat / pdf, p_hat);
    return r;
}

// The previous frame's reservoir of the reprojected pixel, or an empty one (M = 0) when the
// history, the reprojection, the depth or the normal does not hold. Motion follows RAA's S12
// convention: previous uv = uv - motion, z > 0.5 flags a rejected pixel.
fn raa_rgi_temporal_candidate(pixel: vec2<u32>, s: RaaRgiSurface) -> RaaRgiReservoir {
    let empty = raa_rgi_empty();
    if raa_rgi_params.history_usable == 0u {
        return empty;
    }
    let m = textureLoad(raa_rgi_motion, vec2<i32>(pixel), 0);
    if m.z > 0.5 {
        return empty;
    }
    let uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) * raa_gi_frame.viewport.zw - m.xy;
    if any(uv <= vec2<f32>(0.0)) || any(uv >= vec2<f32>(1.0)) {
        return empty;
    }
    let size = vec2<u32>(raa_gi_frame.viewport.xy);
    let prev = min(vec2<u32>(uv * raa_gi_frame.viewport.xy), size - vec2<u32>(1u));
    let t = raa_rgi_load_previous(prev);
    if !(t.confidence > 0.0)
        || !raa_restir_depth_similar(t.owner_depth, s.depth)
        || !(dot(t.owner_normal, s.normal) >= raa_rgi_params.normal_cos) {
        return empty;
    }
    return t;
}

struct RaaRgiStage1 {
    reservoir: RaaRgiReservoir,
    target_value: f32,
}

fn raa_rgi_stage1(pixel: vec2<u32>) -> RaaRgiStage1 {
    let s = raa_rgi_surface(vec2<i32>(pixel));
    if !s.valid {
        return RaaRgiStage1(raa_rgi_empty(), 0.0);
    }
    let c = raa_rgi_initial(pixel, s);
    var t = raa_rgi_temporal_candidate(pixel, s);
    let target_c = raa_rgi_target(s.position, s.normal, c.position, c.normal, c.radiance);
    if !(t.confidence > 0.0) {
        return RaaRgiStage1(c, target_c);
    }
    t.confidence = raa_restir_clamp_confidence(t.confidence, c.confidence, raa_rgi_params.m_cap);
    let target_t = raa_rgi_target(s.position, s.normal, t.position, t.normal, t.radiance);
    let seed = raa_rgi_seed(pixel, raa_rgi_params.frame_index, RAA_RGI_SALT_TEMPORAL);
    var merge = raa_rgi_merge_start();
    raa_rgi_merge(&merge, c, target_c, raa_restir_random(seed));
    raa_rgi_merge(&merge, t, target_t, raa_restir_random(seed + 1u));
    var out = merge.kept;
    out.confidence = merge.confidence;
    // W = w_sum / (M p_hat(z)): every candidate was counted in M with its own target positive or
    // not (Algorithm 3); the history's samples keep their stored radiance.
    out.weight = raa_restir_contribution_weight(merge.w_sum / max(merge.confidence, 1.0e-6), merge.target_value);
    out.owner_normal = s.normal;
    out.owner_depth = s.depth;
    return RaaRgiStage1(out, merge.target_value);
}

// Every invocation reaches both barriers; outside the viewport q = 0 and nothing is written.
@compute @workgroup_size(8, 8)
fn raa_gi_restir_gi_main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let inside = all(id.xy < vec2<u32>(raa_gi_frame.viewport.xy));
    var result = RaaRgiStage1(raa_rgi_empty(), 0.0);
    if inside {
        result = raa_rgi_stage1(id.xy);
    }
    let q = raa_rgi_finite(result.target_value * result.reservoir.weight);
    raa_rgi_boiling_q[lane] = q;
    workgroupBarrier();
    if lane == 0u {
        var sum = 0.0;
        var count = 0u;
        for (var i = 0u; i < 64u; i = i + 1u) {
            let value = raa_rgi_boiling_q[i];
            if value > 0.0 {
                sum = sum + value;
                count = count + 1u;
            }
        }
        raa_rgi_boiling_mean = sum / f32(max(count, 1u));
    }
    workgroupBarrier();
    if !raa_restir_boiling_keep(q, raa_rgi_boiling_mean, raa_rgi_params.boiling_factor) {
        result.reservoir.weight = 0.0;
    }
    if inside {
        raa_rgi_store(id.xy, result.reservoir);
    }
}
