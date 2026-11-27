
// Virtual shadow map of the sun (ShadowGpu.params.w == 2, crates/raa-shadow/src/vsm.rs). The
// virtual map is a clipmap of 16384 x 16384 texel levels around the camera (level l texel =
// texel0 x 2^l), paged in 128 x 128 pages held in the pool layers of raa_shadow_map; the
// page table of level l (128 x 128 entries, toroidal, entry = pool slot + 1, 0 = absent) is
// itself pool slot l, entries stored as depth = entry / 2^24. ShadowGpu in this mode:
// cascade_view_proj[0] world -> (light x, light y, depth 0..1); [1] the level window origins
// in pages (level l: x, y at component 2 (l % 2) of column l / 2); [2] column 0 (texel0,
// levels, pool base layer, pages per layer side), column 1 (pixel angle, level bias, blocker
// reach m, depth range m), column 2 x (constant footprint m of an orthographic camera); [3] the scene-wide coarse map in cascade layer 3 (the fallback of
// non-resident pages, through raa_csm_cascade). The kernel is the cascade one: PCSS blocker
// search, then the noise-free tent grid or (camera jitter) the turned Poisson disk, with
// per-tap page translation so taps can cross page borders.

struct RaaVsmPage {
    page: vec2<i32>,
    // Texel origin of the page's slot (xy) and its array layer (z).
    origin: vec3<i32>,
    // The whole kernel lies inside this page: taps skip the page translation.
    inside: bool,
}

fn raa_csm_vsm_param(column: u32) -> vec4<f32> {
    return raa_shadow.cascade_view_proj[2][column];
}

fn raa_csm_vsm_origin(level: u32) -> vec2<i32> {
    let c = raa_shadow.cascade_view_proj[1][level >> 1u];
    return vec2<i32>(select(c.xy, c.zw, (level & 1u) == 1u));
}

fn raa_csm_vsm_texel_m(level: u32) -> f32 {
    return raa_csm_vsm_param(0u).x * exp2(f32(level));
}

// Clipmap level whose texel is closest to the pixel footprint at `view_depth` (shared with
// vsm_mark.wgsl).
fn raa_csm_vsm_level(view_depth: f32) -> u32 {
    let p = raa_csm_vsm_param(1u);
    let footprint = max(view_depth, 1e-4) * p.x + raa_csm_vsm_param(2u).x;
    let l = floor(log2(max(footprint / raa_csm_vsm_param(0u).x, 1e-8)) + 0.5 + p.y);
    return u32(clamp(l, 0.0, raa_csm_vsm_param(0u).y - 1.0));
}

fn raa_csm_vsm_slot_origin(slot: i32) -> vec3<i32> {
    let pps = i32(raa_csm_vsm_param(0u).w);
    let per_layer = pps * pps;
    let local = slot % per_layer;
    return vec3<i32>(vec2<i32>(local % pps, local / pps) * 128, i32(raa_csm_vsm_param(0u).z) + slot / per_layer);
}

// Pool slot of `page` of `level`, -1 when it is outside the level window or not resident.
fn raa_csm_vsm_slot(level: u32, page: vec2<i32>) -> i32 {
    let rel = page - raa_csm_vsm_origin(level);
    if any(rel < vec2<i32>(0)) || any(rel >= vec2<i32>(128)) {
        return -1;
    }
    let pps = i32(raa_csm_vsm_param(0u).w);
    let table = vec2<i32>(i32(level) % pps, i32(level) / pps) * 128 + (page & vec2<i32>(127));
    let d = textureLoad(raa_shadow_map, table, i32(raa_csm_vsm_param(0u).z), 0);
    return i32(round(d * 16777216.0)) - 1;
}

// Physical texel position (xy) and layer (z) of virtual texel position `texel`: its own page
// when resident, otherwise clamped into `home`.
fn raa_csm_vsm_resolve(level: u32, texel: vec2<f32>, home: RaaVsmPage) -> vec3<f32> {
    if (home.inside) {
        return vec3<f32>(vec2<f32>(home.origin.xy) + texel - vec2<f32>(home.page * 128), f32(home.origin.z));
    }
    let page = vec2<i32>(floor(texel / 128.0));
    var p = home.page;
    var o = home.origin;
    if any(page != home.page) {
        let s = raa_csm_vsm_slot(level, page);
        if s >= 0 {
            p = page;
            o = raa_csm_vsm_slot_origin(s);
        }
    }
    let local = clamp(texel - vec2<f32>(p * 128), vec2<f32>(0.5), vec2<f32>(127.5));
    return vec3<f32>(vec2<f32>(o.xy) + local, f32(o.z));
}

fn raa_csm_vsm_cmp(level: u32, texel: vec2<f32>, depth: f32, home: RaaVsmPage) -> f32 {
    let r = raa_csm_vsm_resolve(level, texel, home);
    return textureSampleCompareLevel(raa_shadow_map, raa_shadow_cmp, r.xy * raa_shadow.params.z, i32(r.z), depth);
}

fn raa_csm_vsm_load(level: u32, texel: vec2<f32>, home: RaaVsmPage) -> f32 {
    let r = raa_csm_vsm_resolve(level, texel, home);
    return textureLoad(raa_shadow_map, vec2<i32>(r.xy), i32(r.z), 0);
}

// Noise-free tent grid PCF (as raa_csm_grid_pcf) over virtual texels.
fn raa_csm_vsm_grid_pcf(level: u32, uv: vec2<f32>, depth: f32, r: f32, home: RaaVsmPage) -> f32 {
    let probe = raa_csm_vsm_cmp(level, uv, depth, home)
        + raa_csm_vsm_cmp(level, uv + vec2<f32>(r, r), depth, home)
        + raa_csm_vsm_cmp(level, uv + vec2<f32>(-r, r), depth, home)
        + raa_csm_vsm_cmp(level, uv + vec2<f32>(r, -r), depth, home)
        + raa_csm_vsm_cmp(level, uv + vec2<f32>(-r, -r), depth, home);
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
            sum = sum + w * raa_csm_vsm_cmp(level, uv + vec2<f32>(x, y), depth, home);
            wsum = wsum + w;
        }
    }
    return sum / max(wsum, 1e-6);
}

// Turned 32-tap Poisson PCF (as raa_shadow_pcf, probe taps first) over virtual texels.
fn raa_csm_vsm_poisson_pcf(level: u32, uv: vec2<f32>, depth: f32, r: f32, rotation: f32, home: RaaVsmPage) -> f32 {
    var probe = 0.0;
    for (var i = 0u; i < 32u; i = i + 1u) {
        if ((RAA_PCF_PROBE_MASK >> i) & 1u) != 0u {
            probe = probe + raa_csm_vsm_cmp(level, uv + raa_csm_rotate(RAA_CSM_POISSON[i], rotation) * r, depth, home);
        }
    }
    if (probe == 0.0 || probe == RAA_PCF_PROBE_COUNT) {
        return probe / RAA_PCF_PROBE_COUNT;
    }
    var pcf = probe;
    for (var i = 0u; i < 32u; i = i + 1u) {
        if ((RAA_PCF_PROBE_MASK >> i) & 1u) == 0u {
            pcf = pcf + raa_csm_vsm_cmp(level, uv + raa_csm_rotate(RAA_CSM_POISSON[i], rotation) * r, depth, home);
        }
    }
    return pcf / 32.0;
}

// Sun visibility from the virtual shadow map: the finest resident level at or above the
// pixel's level (the coarse scene map when none is), normal-offset receiver, PCSS.
fn raa_csm_vsm_shadow(world_pos: vec3<f32>, n: vec3<f32>, frag: vec4<f32>, view_depth: f32) -> f32 {
    let levels = u32(raa_csm_vsm_param(0u).y);
    if (levels == 0u) {
        return 1.0;
    }
    let m = raa_shadow.cascade_view_proj[0];
    var level = raa_csm_vsm_level(view_depth);
    var home = RaaVsmPage(vec2<i32>(0), vec3<i32>(-1), false);
    var texel_m = 1.0;
    var uv = vec2<f32>(0.0);
    var depth = 0.0;
    for (var l = level; l < levels; l = l + 1u) {
        texel_m = raa_csm_vsm_texel_m(l);
        let lp = m * vec4<f32>(world_pos + n * (raa_shadow.params.y * texel_m), 1.0);
        uv = vec2<f32>(lp.x, -lp.y) / texel_m;
        depth = lp.z;
        let page = vec2<i32>(floor(uv / 128.0));
        let s = raa_csm_vsm_slot(l, page);
        if (s >= 0) {
            home = RaaVsmPage(page, raa_csm_vsm_slot_origin(s), false);
            level = l;
            break;
        }
    }
    if (home.origin.z < 0) {
        return raa_csm_cascade(world_pos, n, frag, 3u, view_depth);
    }
    if (depth <= 0.0 || depth >= 1.0) {
        return 1.0;
    }
    let tan_angle = max(tan(raa_frame.sun_dir.w * raa_shadow.params.x), 0.0);
    let min_texels = 1.5;
    let search_texels = clamp(max(tan_angle * raa_csm_vsm_param(1u).z / texel_m, min_texels + 1.0), 1.0, 32.0);
    // Texels from uv to the nearest border of the home page: kernels that stay inside it
    // (most pixels) address the slot directly.
    let local = uv - vec2<f32>(home.page * 128);
    let room = min(min(local.x, local.y), 128.0 - max(local.x, local.y)) - 1.0;
    home.inside = search_texels < room;
    var blocker_sum = 0.0;
    var blocker_count = 0.0;
    for (var i = 0u; i < 16u; i = i + 1u) {
        let d = raa_csm_vsm_load(level, uv + RAA_CSM_POISSON[i] * search_texels, home);
        if (d < depth) {
            blocker_sum = blocker_sum + d;
            blocker_count = blocker_count + 1.0;
        }
    }
    if (blocker_count < 0.5) {
        return 1.0;
    }
    // Light-depth delta in metres (depth range in param 1.w) x tan(angle), in texels.
    let delta_m = (depth - blocker_sum / blocker_count) * raa_csm_vsm_param(1u).w;
    let penumbra = clamp(delta_m * tan_angle / texel_m, min_texels, 48.0);
    home.inside = penumbra < room;
    if any(raa_frame.camera_jitter.xy != vec2<f32>(0.0)) {
        return raa_csm_vsm_poisson_pcf(level, uv, depth, penumbra, raa_csm_frame_rotation(frag.xy), home);
    }
    return raa_csm_vsm_grid_pcf(level, uv, depth, penumbra, home);
}
