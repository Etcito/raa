// Shadow slot of provider `csm` (S12 `shadow.csm`): cascaded soft sun shadows. A
// normal-offset receiver picks the cascade by view depth (blending across the cascade
// band and fading to 1 past the shadow distance) and estimates the penumbra from the
// shared shadow-map PCSS kernel, which lives in the shader prelude
// (crates/raa-shader/src/wgsl/prelude.wgsl: RAA_CSM_POISSON, raa_csm_rotation,
// raa_shadow_blockers, raa_shadow_pcf). The result is multiplied by the contact-shadow
// texture and by the shared cloud-shadow transmittance `raa_cloud_shadow`. Reads only the
// prelude bindings raa_shadow_map, raa_shadow_cmp, raa_shadow, raa_contact_tex and
// raa_frame; the cloud helper consumes the group-1 weather declarations and binding-6
// sampler, not this file.

// Shadow of one cascade: normal-offset receiver, then the prelude's 16-sample blocker
// search and 32-sample rotated Poisson PCF (bilinear comparison taps), the kernel at least
// the world filter floor (params2.x). 1 when unshadowed or outside the cascade.
// `view_depth` sizes the blocker search region from the light angle.
fn raa_csm_cascade(
    world_pos: vec3<f32>,
    n: vec3<f32>,
    frag: vec4<f32>,
    cascade_idx: u32,
    view_depth: f32
) -> f32 {
    let m = raa_shadow.cascade_view_proj[cascade_idx];
    let res = 1.0 / raa_shadow.params.z;
    let row0 = vec3<f32>(m[0].x, m[1].x, m[2].x);
    let row2 = vec3<f32>(m[0].z, m[1].z, m[2].z);
    let row2_len = max(length(row2), 1e-6);
    // World size of one shadow-map texel: the ortho width is 2 / |row 0|.
    let texel = 2.0 / max(length(row0), 1e-6) / res;
    // Normal-offset receiver: push along the normal by 1.5 world texels.
    let world = world_pos + n * (raa_shadow.params.y * texel);
    let clip = m * vec4<f32>(world, 1.0);
    let uv = vec2<f32>(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5);
    if (clip.z <= 0.0 || clip.z >= 1.0
        || uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return 1.0;
    }
    let sun_angle = raa_frame.sun_dir.w * raa_shadow.params.x;
    let tan_angle = max(tan(sun_angle), 0.0);
    // Filter floor in world metres (params2.x), the same size in every cascade so the edge
    // does not change at a cascade boundary; at least 1.5 texels of this cascade.
    let min_texels = max(1.5, raa_shadow.params2.x / max(texel, 1e-6));
    // The blocker search covers at least the filter, so the penumbra is never cut short.
    let search_texels = clamp(max(tan_angle * view_depth / max(texel, 1e-6), min_texels + 1.0),
        1.0, 64.0);
    let b = raa_shadow_blockers(i32(cascade_idx), uv, clip.z, search_texels, res);
    if (b.y < 0.5) {
        return 1.0;
    }
    let blocker_depth = b.x;
    // Penumbra: light-depth delta / |row 2| in metres x tan(angle), in texels, at least the
    // filter floor.
    let penumbra_texels = clamp(
        (clip.z - blocker_depth) / row2_len * tan_angle / max(texel, 1e-6),
        min_texels, 48.0
    );
    // A jittered camera (TAA, reconstruction) resolves the rotated Poisson noise over frames;
    // without jitter that per-pixel noise stays as a fixed diagonal hatching, so the kernel is
    // a noise-free tent-weighted grid of bilinear comparisons instead.
    if any(raa_frame.camera_jitter.xy != vec2<f32>(0.0)) {
        return raa_shadow_pcf(i32(cascade_idx), uv, clip.z, penumbra_texels, res, raa_csm_frame_rotation(frag.xy));
    }
    return raa_csm_grid_pcf(i32(cascade_idx), uv, clip.z, penumbra_texels, res);
}

fn raa_csm_grid_tap(layer: i32, uv: vec2<f32>, depth: f32, offset_texels: vec2<f32>, res: f32) -> f32 {
    let coord = clamp(uv + offset_texels / res, vec2<f32>(0.0), vec2<f32>(1.0));
    return textureSampleCompareLevel(raa_shadow_map, raa_shadow_cmp, coord, layer, depth);
}

// Noise-free PCF: an n x n grid (n = 4..8, about one tap per texel) of bilinear comparison
// taps over [-r, r] texels, tent-weighted. The centre and four corners run first; when they
// agree (whole kernel lit or in umbra) their value is returned.
fn raa_csm_grid_pcf(layer: i32, uv: vec2<f32>, depth: f32, radius_texels: f32, res: f32) -> f32 {
    let r = radius_texels;
    let probe = raa_csm_grid_tap(layer, uv, depth, vec2<f32>(0.0), res)
        + raa_csm_grid_tap(layer, uv, depth, vec2<f32>(r, r), res)
        + raa_csm_grid_tap(layer, uv, depth, vec2<f32>(-r, r), res)
        + raa_csm_grid_tap(layer, uv, depth, vec2<f32>(r, -r), res)
        + raa_csm_grid_tap(layer, uv, depth, vec2<f32>(-r, -r), res);
    if (probe == 0.0 || probe == 5.0) {
        return probe / 5.0;
    }
    let n = u32(clamp(ceil(r) + 2.0, 4.0, 8.0));
    let step = 2.0 * r / f32(n - 1u);
    var sum = 0.0;
    var wsum = 0.0;
    for (var j = 0u; j < n; j = j + 1u) {
        let y = -r + step * f32(j);
        let wy = 1.0 - abs(y) / (r + step);
        for (var i = 0u; i < n; i = i + 1u) {
            let x = -r + step * f32(i);
            let w = wy * (1.0 - abs(x) / (r + step));
            sum = sum + w * raa_csm_grid_tap(layer, uv, depth, vec2<f32>(x, y), res);
            wsum = wsum + w;
        }
    }
    return sum / max(wsum, 1e-6);
}

// Per-pixel PCF rotation; when the camera is jittered (TAA, neural reconstruction) it also
// turns with the jitter, so the history averages the rotated taps instead of keeping a
// fixed per-pixel pattern. Without jitter the pattern is fixed (no shimmer).
fn raa_csm_frame_rotation(p: vec2<f32>) -> f32 {
    let j = raa_frame.camera_jitter.xy;
    let turn = fract(dot(j, vec2<f32>(4093.1, 7919.7)));
    return raa_csm_rotation(p) + turn * 6.28318530718;
}

// The reflected view can see surfaces outside the main camera's frustum.
// Choose a sun map by its light-space footprint, since the view-depth splits
// describe the main camera. Keep a margin for the normal offset and PCSS taps.
fn raa_csm_planar(world_pos: vec3<f32>, n: vec3<f32>, frag: vec4<f32>) -> f32 {
    let view_depth = max(-(raa_frame.view * vec4<f32>(world_pos, 1.0)).z, 0.0);
    let count = select(4u, u32(raa_shadow.params2.z), raa_shadow.params2.z > 0.5);
    for (var index = 0u; index < count; index++) {
        let clip = raa_shadow.cascade_view_proj[index] * vec4<f32>(world_pos, 1.0);
        if all(abs(clip.xy) < vec2<f32>(0.95)) && clip.z > 0.0 && clip.z < 1.0 {
            return raa_csm_cascade(world_pos, n, frag, index, view_depth);
        }
    }
    return 1.0;
}

// S11 Shadow slot: contact and cloud terms when shadows are cleared, the virtual shadow map
// when ShadowGpu.params.w is 2, otherwise the cascade
// shadow (blended across the 10% band, faded to 1 past the shadow distance) x contact,
// times the shared cloud-shadow visibility. Disabled clouds return 1 from the helper.
fn raa_slot_shadow(world_pos: vec3<f32>, n: vec3<f32>, frag: vec4<f32>) -> f32 {
    if raa_feature_enabled(RAA_PLANAR_CAPTURE_FEATURE) {
        let cloud = raa_cloud_shadow(world_pos);
        if raa_shadow.params.w < 0.5 { return cloud; }
        if raa_shadow.params.w > 1.5 {
            // VSM: the reflected view reads the scene-wide coarse map (cascade layer 3).
            let vd = max(-(raa_frame.view * vec4<f32>(world_pos, 1.0)).z, 0.0);
            return raa_csm_cascade(world_pos, n, frag, 3u, vd) * cloud;
        }
        return raa_csm_planar(world_pos, n, frag) * cloud;
    }
    // Clamped like crates/raa-gi/src/wgsl/slots/slot_ao.wgsl:9-27: the 1x1 dummy reads 1, never out of bounds.
    let contact_size = vec2<i32>(textureDimensions(raa_contact_tex));
    let contact_px = clamp(vec2<i32>(frag.xy), vec2<i32>(0), contact_size - vec2<i32>(1));
    let contact = textureLoad(raa_contact_tex, contact_px, 0).r;
    if (raa_shadow.params.w < 0.5) {
        return contact * raa_cloud_shadow(world_pos);
    }
    let view_depth = -(raa_frame.view * vec4<f32>(world_pos, 1.0)).z;
    // params.w 2: the virtual shadow map (vsm.wgsl) instead of the cascades.
    if (raa_shadow.params.w > 1.5) {
        return raa_csm_vsm_shadow(world_pos, n, frag, view_depth) * contact * raa_cloud_shadow(world_pos);
    }
    let s = raa_shadow.splits;
    let band = max(raa_shadow.params2.y, 1e-6);
    // params2.z = cascade count (1..4; 0 = legacy 4): the last used cascade fades to 1.
    let last = select(3u, u32(raa_shadow.params2.z) - 1u, raa_shadow.params2.z > 0.5);
    var idx = 0u;
    if (view_depth > s.x) { idx = 1u; }
    if (view_depth > s.y) { idx = 2u; }
    if (view_depth > s.z) { idx = 3u; }
    idx = min(idx, last);
    var near_split = 0.0;
    var far_split = s.x;
    if (idx == 1u) { near_split = s.x; far_split = s.y; }
    if (idx == 2u) { near_split = s.y; far_split = s.z; }
    if (idx == 3u) { near_split = s.z; far_split = s.w; }
    let band_world = max(band * max(far_split - near_split, 1e-6), 1e-6);
    var vis = raa_csm_cascade(world_pos, n, frag, idx, view_depth);
    if (idx < last) {
        let blend = clamp((view_depth - (far_split - band_world)) / band_world, 0.0, 1.0);
        // Only the band pixels pay for the next cascade's PCSS: mix(vis, next, 0) is vis.
        // The kernel uses textureLoad / textureSampleCompareLevel, valid in this branch.
        if (blend > 0.0) {
            let next_vis = raa_csm_cascade(world_pos, n, frag, idx + 1u, view_depth);
            vis = mix(vis, next_vis, blend);
        }
    } else {
        let fade = clamp((view_depth - far_split) / band_world, 0.0, 1.0);
        vis = mix(vis, 1.0, fade);
    }
    return vis * contact * raa_cloud_shadow(world_pos);
}
