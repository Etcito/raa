// ReSTIR GI, stage 2 (spatial reuse and resolve) of the screen part's `restir` mode. Per pixel:
// the pixel's stage-1 reservoir and `spatial_samples` neighbours within `spatial_radius` pixels
// (rejected on depth or normal) are merged at the pixel's visible point x_r. A neighbour's
// sample keeps its point x_s (reconnection shift): its target is evaluated at x_r and multiplied
// by the shift Jacobian (raa_rgi_jacobian), and it counts only when x_s is visible from x_r
// through the GI voxel field. The confidence normaliser Z sums the M of every merged reservoir
// whose surface sees the kept sample with a positive target (Ouyang et al. 2021, Algorithm 4).
// The output is the irradiance estimate E = L cos W at binding 13; the spatial result is not
// written back, so next frame's temporal reuse reads stage-1 reservoirs only. RAA's own code.

@group(2) @binding(13) var raa_rgi_raw: texture_storage_2d<rgba16float, write>;

const RAA_RGI_SALT_SPATIAL: u32 = 0x1B56C4E9u;
const RAA_RGI_MAX_NEIGHBOURS: u32 = 8u;
const RAA_RGI_VISIBILITY_STEPS: u32 = 32u;
const RAA_RGI_VISIBILITY_DISTANCE: f32 = 320.0;

// True when the sample point is visible from the receiver through the GI voxel field: the ray
// leaves 1.5 finest voxels along the receiver normal and stops two voxels short of x_s.
fn raa_rgi_visible(s: RaaRgiSurface, x_s: vec3<f32>) -> bool {
    let count = raa_gi_volume_params.level_count;
    if count == 0u {
        return true;
    }
    let level = min(raa_gi_level_at(s.position, 0u), count - 1u);
    let voxel = raa_gi_volume_params.levels[level].w;
    let origin = s.position + s.normal * voxel * 1.5;
    let d = x_s - origin;
    let len = length(d);
    let max_t = min(len - 2.0 * voxel, RAA_RGI_VISIBILITY_DISTANCE);
    if max_t <= voxel {
        return true;
    }
    return raa_gi_visibility(origin, d / len, voxel * 0.5, max_t, RAA_RGI_VISIBILITY_STEPS) > 0.5;
}

@compute @workgroup_size(8, 8)
fn raa_gi_restir_gi_spatial(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<u32>(raa_gi_frame.viewport.xy);
    if any(id.xy >= size) {
        return;
    }
    let pixel = id.xy;
    let s = raa_rgi_surface(vec2<i32>(pixel));
    if !s.valid {
        textureStore(raa_rgi_raw, pixel, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }
    let frame = raa_rgi_params.frame_index;
    let seed = raa_restir_pcg(pixel.x * 7919u + pixel.y * 104729u + frame * 31337u + RAA_RGI_SALT_SPATIAL);

    let own = raa_rgi_load(pixel);
    var merge = raa_rgi_merge_start();
    raa_rgi_merge(&merge, own, raa_rgi_target(s.position, s.normal, own.position, own.normal, own.radiance),
        raa_restir_random(seed));

    // The merged neighbours' surfaces and confidences, for Z.
    var positions: array<vec3<f32>, RAA_RGI_MAX_NEIGHBOURS>;
    var normals: array<vec3<f32>, RAA_RGI_MAX_NEIGHBOURS>;
    var confidences: array<f32, RAA_RGI_MAX_NEIGHBOURS>;
    var merged = 0u;
    let samples = min(raa_rgi_params.spatial_samples, RAA_RGI_MAX_NEIGHBOURS);
    let last = vec2<i32>(size) - vec2<i32>(1);
    for (var k = 0u; k < samples; k = k + 1u) {
        let offset = raa_restir_spatial_offset(pixel, frame, k + 1u, raa_rgi_params.spatial_radius);
        if all(offset == vec2<i32>(0)) {
            continue;
        }
        let q = clamp(vec2<i32>(pixel) + offset, vec2<i32>(0), last);
        let sq = raa_rgi_surface(q);
        if !sq.valid || !raa_restir_depth_similar(sq.depth, s.depth)
            || !(dot(sq.normal, s.normal) >= raa_rgi_params.normal_cos) {
            continue;
        }
        let rn = raa_rgi_load(vec2<u32>(q));
        if !(rn.confidence > 0.0) {
            continue;
        }
        let jacobian = raa_rgi_jacobian(s.position, sq.position, rn.position, rn.normal);
        if !(jacobian > 1.0 / RAA_RGI_JACOBIAN_MAX) || jacobian > RAA_RGI_JACOBIAN_MAX {
            continue;
        }
        var target_n = raa_rgi_target(s.position, s.normal, rn.position, rn.normal, rn.radiance) * jacobian;
        if target_n > 0.0 && !raa_rgi_visible(s, rn.position) {
            target_n = 0.0;
        }
        raa_rgi_merge(&merge, rn, target_n, raa_restir_random(seed + 1u + k));
        positions[merged] = sq.position;
        normals[merged] = sq.normal;
        confidences[merged] = rn.confidence;
        merged = merged + 1u;
    }

    let z = merge.kept;
    let target_z = raa_rgi_target(s.position, s.normal, z.position, z.normal, z.radiance);
    // Z: the receiver's own M when its target is positive, plus every merged neighbour whose
    // surface gives the kept sample a positive target.
    var total = select(0.0, own.confidence, target_z > 0.0);
    for (var i = 0u; i < merged; i = i + 1u) {
        if raa_rgi_target(positions[i], normals[i], z.position, z.normal, z.radiance) > 0.0 {
            total = total + confidences[i];
        }
    }
    let weight = raa_restir_contribution_weight(merge.w_sum / max(total, 1.0e-6), target_z);
    let d = z.position - s.position;
    let cos_v = max(dot(s.normal, d / max(length(d), 1.0e-6)), 0.0);
    let irradiance = z.radiance * cos_v * weight;
    let safe = select(vec3<f32>(0.0), irradiance, all(irradiance == irradiance) && all(irradiance < vec3<f32>(65504.0)));
    textureStore(raa_rgi_raw, pixel, vec4<f32>(safe, 1.0));
}
