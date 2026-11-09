// Depth reduction/range-plane tests adapted from solstice@7e465b5,
// crates/solstice-render/shaders/lighting/light_tiles.wgsl:25-112 (MIT).
// Copyright (c) 2026 SamG-Coder. Point window from deferred.wgsl:72-83.
// RAA uses standard Z: clear 1, nearest MIN, farthest MAX (source uses reversed Z).
// Transparent/displaced-fur lighting approximates the opaque/fur-base prepass bounds;
// without prepass coverage a tile has no candidates, including directionals.
@group(2) @binding(0) var tile_depth: texture_depth_2d;
@group(2) @binding(1) var<uniform> tile_lights: array<LightGpu, MAX_LOCAL_LIGHTS>;
@group(2) @binding(2) var tile_words: texture_storage_2d_array<r32uint, write>;
var<workgroup> nearest_bits: atomic<u32>;
var<workgroup> farthest_bits: atomic<u32>;
// Handoff to fixed-network selection: one lane per dense handle-ordered GPU index.
// Eligibility is independent of score; zero-score conservative candidates survive.
var<workgroup> candidate_eligible: array<u32, MAX_LOCAL_LIGHTS>;
var<workgroup> candidate_scores: array<f32, MAX_LOCAL_LIGHTS>;
var<workgroup> candidate_indices: array<u32, MAX_LOCAL_LIGHTS>;

fn tile_world(pixel: vec2<f32>, depth: f32) -> vec3<f32> {
    let uv = pixel / raa_frame.viewport.xy;
    let view = raa_frame.inv_proj * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    return (raa_frame.inv_view * vec4<f32>(view.xyz / view.w, 1.0)).xyz;
}

fn tile_score(light: LightGpu, point: vec3<f32>) -> f32 {
    var factor = 1.0;
    if light.kind != 2u {
        var origin = light.position_range.xyz;
        let travel = light.direction_cos_outer.xyz;
        if light.kind == 3u {
            let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(travel.y) < 0.99);
            let tangent = normalize(cross(up, travel));
            let bitangent = cross(travel, tangent);
            let offset = point - origin;
            origin += tangent * clamp(dot(offset, tangent), -light.width * 0.5, light.width * 0.5)
                + bitangent * clamp(dot(offset, bitangent), -light.height * 0.5, light.height * 0.5);
            if dot(offset, travel) <= 0.0 { return 0.0; }
        }
        let delta = point - origin;
        let distance = length(delta);
        if distance >= light.position_range.w { return 0.0; }
        let ratio = distance / light.position_range.w;
        let window = clamp(1.0 - (ratio * ratio) * (ratio * ratio), 0.0, 1.0);
        factor = window * window / max(distance * distance, 0.04);
        if light.kind == 1u {
            let cosine = dot(travel, delta / max(distance, 1e-20));
            let outer = light.direction_cos_outer.w;
            if light.cos_inner <= outer {
                factor *= select(0.0, 1.0, cosine >= outer);
            } else {
                factor *= smoothstep(outer, light.cos_inner, cosine);
            }
        }
    }
    let raw = dot(light.color_inv_range_sq.rgb, vec3<f32>(0.2126, 0.7152, 0.0722)) * factor;
    // NaN maps to zero; positive overflow saturates, never poisons the sorting key.
    return select(0.0, min(raw, 3.402823466e38), raw >= 0.0);
}

fn tile_reaches(light: LightGpu, lo: vec2<f32>, hi: vec2<f32>, near: vec3<f32>, far: vec3<f32>) -> bool {
    if light.kind == 2u { return true; }
    var radius = light.position_range.w;
    if light.kind == 3u { radius += length(vec2<f32>(light.width, light.height) * 0.5); }
    let position = light.position_range.xyz;
    let forward = -vec3<f32>(raa_frame.view[0].z, raa_frame.view[1].z, raa_frame.view[2].z);
    if dot(position - near, forward) < -radius || dot(position - far, forward) > radius { return false; }
    // Homogeneous clip half-spaces give inward world-space planes for both
    // perspective and orthographic cameras, including the current projection jitter.
    let rows = transpose(raa_frame.view_proj);
    let left = 2.0 * lo.x / raa_frame.viewport.x - 1.0;
    let right = 2.0 * hi.x / raa_frame.viewport.x - 1.0;
    let top = 1.0 - 2.0 * lo.y / raa_frame.viewport.y;
    let bottom = 1.0 - 2.0 * hi.y / raa_frame.viewport.y;
    let planes = array<vec4<f32>, 4>(rows[0] - left * rows[3], right * rows[3] - rows[0],
        rows[1] - bottom * rows[3], top * rows[3] - rows[1]);
    for (var p = 0u; p < 4u; p += 1u) {
        if dot(planes[p], vec4<f32>(position, 1.0)) < -radius * length(planes[p].xyz) { return false; }
    }
    return true;
}

@compute @workgroup_size(TILE_SIZE, TILE_SIZE)
fn light_tiles(@builtin(workgroup_id) group: vec3<u32>,
    @builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    if lane == 0u {
        atomicStore(&nearest_bits, bitcast<u32>(1.0));
        atomicStore(&farthest_bits, 0u);
        // Selection's emission stage replaces this covered-tile count later.
        textureStore(tile_words, vec2<i32>(group.xy), 0, vec4<u32>(0u));
    }
    candidate_eligible[lane] = 0u;
    candidate_scores[lane] = 0.0;
    candidate_indices[lane] = lane;
    workgroupBarrier();
    if all(id.xy < vec2<u32>(raa_frame.viewport.xy)) && all(id.xy < textureDimensions(tile_depth)) {
        let depth = textureLoad(tile_depth, vec2<i32>(id.xy), 0);
        // Also excludes NaN/infinity and accepts valid near-plane depth zero.
        if depth >= 0.0 && depth < 1.0 {
            atomicMin(&nearest_bits, bitcast<u32>(max(depth, 0.0)));
            atomicMax(&farthest_bits, bitcast<u32>(max(depth, 0.0)));
        }
    }
    workgroupBarrier();
    let nearest = bitcast<f32>(atomicLoad(&nearest_bits));
    let farthest = bitcast<f32>(atomicLoad(&farthest_bits));
    if nearest < 1.0 && lane < min(raa_frame.light_info.x, MAX_LOCAL_LIGHTS) {
        let lo = vec2<f32>(group.xy * TILE_SIZE);
        let hi = min(lo + vec2<f32>(f32(TILE_SIZE)), raa_frame.viewport.xy);
        let centre = (lo + hi) * 0.5;
        let near = tile_world(centre, nearest);
        let far = tile_world(centre, farthest);
        let light = tile_lights[lane];
        if tile_reaches(light, lo, hi, near, far) {
            candidate_eligible[lane] = 1u;
            candidate_scores[lane] = tile_score(light, near);
        }
    }
    workgroupBarrier();
    // Fixed-capacity strongest-127 selection attaches here, consuming all 256 lanes.
}
