// The `gi` provider of the `LocalVisibility` slot (S11/S39): traces the GI volume toward a local
// light so a wall stops the light. This is RAA code; it calls `raa_gi_level_at` and
// `raa_gi_visibility` of `raa_shader::GI_VOLUME_WGSL`, which `local_visibility_provider`
// prepends to this text. That library samples only with `textureSampleLevel`, so the slot is
// safe inside the nonuniform light loop and this file uses no implicit derivative.
//
// This file defines exactly `raa_slot_local_visibility` and no helper; the volume bindings are
// declared by the composed prelude.

fn raa_slot_local_visibility(
    light_index: u32,
    world_position: vec3<f32>,
    surface_normal: vec3<f32>,
    direction_to_light: vec3<f32>,
    light_distance: f32,
) -> f32 {
    if !raa_feature_enabled(30u) {
        return 1.0;
    }
    // The offsets are in voxels of the coarsest level the ray samples at its ends (the shading
    // point and the light), never of the finest level alone: a surface sampled in a coarser
    // level reads a distance quantised to that level's cells, so an origin 1.5 finest voxels off
    // it still lies under that level's hit distance (0.85 of its voxel). A ray leaving the
    // surface at a grazing angle then saw a plateau of equal distances, counted it as
    // "approaching" and shadowed itself, which drew contour blotches over directly lit walls.
    let count = raa_gi_volume_params.level_count;
    if count == 0u {
        return 1.0;
    }
    let directional = light_distance == 0.0;
    let near_level = min(raa_gi_level_at(world_position, 0u), count - 1u);
    let light_position = world_position + direction_to_light * light_distance;
    let far_level = select(
        min(raa_gi_level_at(light_position, near_level), count - 1u),
        near_level,
        directional,
    );
    let voxel = max(
        raa_gi_volume_params.levels[near_level].w,
        raa_gi_volume_params.levels[far_level].w,
    );
    let origin = world_position + surface_normal * (1.5 * voxel);
    let max_t = select(max(light_distance - 3.0 * voxel, 0.0), 400.0, directional);
    return clamp(raa_gi_visibility(origin, direction_to_light, voxel, max_t, 24u), 0.0, 1.0);
}
