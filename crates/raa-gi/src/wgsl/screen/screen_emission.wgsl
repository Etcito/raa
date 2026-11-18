// The emissive gate at screen hits (contract 'gi-screen-emission-wgsl'): the emission of the
// surface a screen ray hits, read from the GI voxel words, and the hit radiance without it when
// the screen emissive source is off.

@group(2) @binding(6) var<storage, read> raa_gi_screen_voxels: array<u32>;

override RAA_GI_SCREEN_EMISSIVE: f32 = 1.0;

// The per-channel maximum emission of the 8 voxels around `world` in its finest covering level
// (word 2 of each voxel: RGBM over 128); 0 outside the volume.
fn raa_gi_screen_hit_emission(world: vec3<f32>) -> vec3<f32> {
    let level = raa_gi_level_at(world, 0u);
    if level >= raa_gi_volume_params.level_count {
        return vec3<f32>(0.0);
    }
    let grid = raa_gi_volume_params.levels[level];
    let n = u32(raa_gi_volume_params.resolution);
    let base = vec3<i32>(floor((world - grid.xyz) / grid.w - 0.5));
    var emission = vec3<f32>(0.0);
    for (var i = 0u; i < 8u; i = i + 1u) {
        let cell = base + vec3<i32>(i32(i & 1u), i32((i >> 1u) & 1u), i32((i >> 2u) & 1u));
        if any(cell < vec3<i32>(0)) || any(cell >= vec3<i32>(i32(n))) {
            continue;
        }
        let c = vec3<u32>(cell);
        let voxel = level * n * n * n + c.z * n * n + c.y * n + c.x;
        let rgbm = unpack4x8unorm(raa_gi_screen_voxels[voxel * 3u + 2u]);
        emission = max(emission, rgbm.rgb * rgbm.a * 128.0);
    }
    return emission;
}

// `radiance` when the screen emissive source is on, else without the hit surface's emission.
fn raa_gi_screen_without_emission(radiance: vec3<f32>, world: vec3<f32>) -> vec3<f32> {
    if RAA_GI_SCREEN_EMISSIVE >= 0.5 {
        return radiance;
    }
    return max(radiance - raa_gi_screen_hit_emission(world), vec3<f32>(0.0));
}
