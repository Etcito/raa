// The `gi` provider of the `LocalVisibility` slot (S11/S39): traces the GI volume toward a local
// light so a wall stops the light. This is RAA code; it calls `raa_gi_finest_voxel` and
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
    let v0 = raa_gi_finest_voxel();
    let origin = world_position + surface_normal * (1.5 * v0);
    let max_t = select(max(light_distance - 3.0 * v0, 0.0), 400.0, light_distance == 0.0);
    return clamp(raa_gi_visibility(origin, direction_to_light, v0, max_t, 24u), 0.0, 1.0);
}
