// Shadow slot of provider `csm` (S12 `shadow.csm`): cascaded soft sun shadows. A
// normal-offset receiver picks the cascade by view depth (blending across the cascade
// band and fading to 1 past the shadow distance), estimates the penumbra from a
// 16-sample Poisson blocker search and filters 32 Poisson samples with hardware
// comparison; the result is multiplied by the contact-shadow texture. Reads only the
// prelude bindings raa_shadow_map, raa_shadow_cmp, raa_shadow, raa_contact_tex and
// raa_frame.

// 32 fixed Poisson-disk positions in the unit disk; the blocker search uses the first 16.
const RAA_CSM_POISSON: array<vec2<f32>, 32> = array<vec2<f32>, 32>(
    vec2<f32>(0.000000, 0.000000),
    vec2<f32>(0.245572, 0.025625),
    vec2<f32>(0.395874, -0.069721),
    vec2<f32>(0.233324, -0.288666),
    vec2<f32>(-0.072749, -0.222033),
    vec2<f32>(0.343213, 0.169892),
    vec2<f32>(0.282429, 0.319118),
    vec2<f32>(0.391256, -0.326816),
    vec2<f32>(0.214275, 0.519212),
    vec2<f32>(-0.041023, 0.441851),
    vec2<f32>(0.149747, 0.189052),
    vec2<f32>(0.528000, -0.535515),
    vec2<f32>(0.579855, -0.197497),
    vec2<f32>(0.590327, 0.103002),
    vec2<f32>(-0.066529, 0.613064),
    vec2<f32>(0.126337, -0.108697),
    vec2<f32>(0.067175, -0.364676),
    vec2<f32>(0.848151, -0.185183),
    vec2<f32>(-0.239579, -0.077709),
    vec2<f32>(0.081520, -0.668538),
    vec2<f32>(-0.213674, 0.357494),
    vec2<f32>(-0.193341, -0.509093),
    vec2<f32>(0.513411, 0.310819),
    vec2<f32>(-0.217035, 0.149110),
    vec2<f32>(-0.009268, 0.246234),
    vec2<f32>(0.295802, -0.527740),
    vec2<f32>(0.775168, 0.373917),
    vec2<f32>(0.104490, 0.636529),
    vec2<f32>(0.642104, 0.522812),
    vec2<f32>(-0.404827, 0.019434),
    vec2<f32>(0.841166, 0.019880),
    vec2<f32>(-0.418969, 0.395065)
);

// Interleaved gradient noise (Jimenez 2014) in [0, 2 pi): the per-pixel PCF rotation.
fn raa_csm_rotation(p: vec2<f32>) -> f32 {
    let g = fract(dot(p, vec2<f32>(0.06711056, 0.00583715)));
    return fract(52.9829189 * g) * 6.28318530718;
}

fn raa_csm_rotate(v: vec2<f32>, a: f32) -> vec2<f32> {
    let s = sin(a);
    let c = cos(a);
    return vec2<f32>(c * v.x - s * v.y, s * v.x + c * v.y);
}

// Shadow of one cascade: normal-offset receiver, 16-sample PCSS blocker search, then a
// 32-sample rotated Poisson PCF with hardware comparison. 1 when unshadowed or outside
// the cascade. `view_depth` sizes the blocker search region from the light angle.
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
    let search_texels = clamp(tan_angle * view_depth / max(texel, 1e-6), 1.0, 64.0);
    // Blocker search over raw depths (no comparison), averaged over blockers in front.
    var blocker_sum = 0.0;
    var blocker_count = 0.0;
    for (var i = 0u; i < 16u; i = i + 1u) {
        let offset = RAA_CSM_POISSON[i] * search_texels;
        let coord = vec2<i32>(
            clamp(uv * res + offset, vec2<f32>(0.0), vec2<f32>(res - 1.0))
        );
        let d = textureLoad(raa_shadow_map, coord, i32(cascade_idx), 0);
        if (d < clip.z) {
            blocker_sum = blocker_sum + d;
            blocker_count = blocker_count + 1.0;
        }
    }
    if (blocker_count < 0.5) {
        return 1.0;
    }
    let blocker_depth = blocker_sum / blocker_count;
    // Penumbra: light-depth delta / |row 2| in metres x tan(angle), in texels, min 1.5.
    let penumbra_texels = max(
        1.5,
        (clip.z - blocker_depth) / row2_len * tan_angle / max(texel, 1e-6)
    );
    let rot = raa_csm_rotation(frag.xy);
    var pcf = 0.0;
    for (var i = 0u; i < 32u; i = i + 1u) {
        let offset = raa_csm_rotate(RAA_CSM_POISSON[i], rot) * penumbra_texels;
        let coord = clamp(uv + offset / res, vec2<f32>(0.0), vec2<f32>(1.0));
        pcf = pcf + textureSampleCompareLevel(
            raa_shadow_map,
            raa_shadow_cmp,
            coord,
            i32(cascade_idx),
            clip.z
        );
    }
    return pcf / 32.0;
}

// S11 Shadow slot: contact term when shadows are cleared, otherwise the cascade shadow
// (blended across the 10% band, faded to 1 past the shadow distance) x contact.
fn raa_slot_shadow(world_pos: vec3<f32>, n: vec3<f32>, frag: vec4<f32>) -> f32 {
    let contact = textureLoad(raa_contact_tex, vec2<i32>(frag.xy), 0).r;
    if (raa_shadow.params.w < 0.5) {
        return contact;
    }
    let view_depth = -(raa_frame.view * vec4<f32>(world_pos, 1.0)).z;
    let s = raa_shadow.splits;
    let band = max(raa_shadow.params2.y, 1e-6);
    var idx = 0u;
    if (view_depth > s.x) { idx = 1u; }
    if (view_depth > s.y) { idx = 2u; }
    if (view_depth > s.z) { idx = 3u; }
    var near_split = 0.0;
    var far_split = s.x;
    if (idx == 1u) { near_split = s.x; far_split = s.y; }
    if (idx == 2u) { near_split = s.y; far_split = s.z; }
    if (idx == 3u) { near_split = s.z; far_split = s.w; }
    let band_world = max(band * max(far_split - near_split, 1e-6), 1e-6);
    var vis = raa_csm_cascade(world_pos, n, frag, idx, view_depth);
    if (idx < 3u) {
        let blend = clamp((view_depth - (far_split - band_world)) / band_world, 0.0, 1.0);
        let next_vis = raa_csm_cascade(world_pos, n, frag, idx + 1u, view_depth);
        vis = mix(vis, next_vis, blend);
    } else {
        let fade = clamp((view_depth - far_split) / band_world, 0.0, 1.0);
        vis = mix(vis, 1.0, fade);
    }
    return vis * contact;
}
