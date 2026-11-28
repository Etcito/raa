
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
    // The 2 x 2 page block around the kernel (the kernel radius stays below half a page):
    // the neighbour page offsets (+-1 per axis) and the slot origins of the x, y and
    // diagonal neighbours (z -1: not resident, the tap clamps into the home page). Looked up
    // once per pixel, so a tap that crosses a page border costs no page-table read.
    step: vec2<i32>,
    ox: vec3<i32>,
    oy: vec3<i32>,
    oxy: vec3<i32>,
}

// Levels between a pixel's level and the level its wide penumbrae are filtered at (FILTER_UP
// in vsm_mark.wgsl).
const RAA_CSM_VSM_FILTER_UP: u32 = 2u;

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

// Origin of a page of the block (z -1 when absent).
fn raa_csm_vsm_neighbour(level: u32, page: vec2<i32>) -> vec3<i32> {
    let s = raa_csm_vsm_slot(level, page);
    return select(vec3<i32>(-1), raa_csm_vsm_slot_origin(s), s >= 0);
}

// Fills the 2 x 2 block of `home` for a kernel centred at virtual texel `uv` (3 page-table reads).
fn raa_csm_vsm_block(level: u32, uv: vec2<f32>, home: ptr<function, RaaVsmPage>) {
    let local = uv - vec2<f32>((*home).page * 128);
    let step = select(vec2<i32>(-1), vec2<i32>(1), local >= vec2<f32>(64.0));
    (*home).step = step;
    (*home).ox = raa_csm_vsm_neighbour(level, (*home).page + vec2<i32>(step.x, 0));
    (*home).oy = raa_csm_vsm_neighbour(level, (*home).page + vec2<i32>(0, step.y));
    (*home).oxy = raa_csm_vsm_neighbour(level, (*home).page + step);
}

// Physical texel position (xy) and layer (z) of virtual texel position `texel`: its page of
// the home block when resident, otherwise clamped into the home page.
fn raa_csm_vsm_resolve(texel: vec2<f32>, home: RaaVsmPage) -> vec3<f32> {
    if (home.inside) {
        return vec3<f32>(vec2<f32>(home.origin.xy) + texel - vec2<f32>(home.page * 128), f32(home.origin.z));
    }
    let d = vec2<i32>(floor(texel / 128.0)) - home.page;
    let cx = d.x != 0;
    let cy = d.y != 0;
    var o = select(select(home.origin, home.ox, cx), select(home.oy, home.oxy, cx), cy);
    var p = home.page + vec2<i32>(select(0, home.step.x, cx), select(0, home.step.y, cy));
    if (o.z < 0) {
        o = home.origin;
        p = home.page;
    }
    let local = clamp(texel - vec2<f32>(p * 128), vec2<f32>(0.5), vec2<f32>(127.5));
    return vec3<f32>(vec2<f32>(o.xy) + local, f32(o.z));
}

fn raa_csm_vsm_cmp(level: u32, texel: vec2<f32>, depth: f32, home: RaaVsmPage) -> f32 {
    let r = raa_csm_vsm_resolve(texel, home);
    return textureSampleCompareLevel(raa_shadow_map, raa_shadow_cmp, r.xy * raa_shadow.params.z, i32(r.z), depth);
}

fn raa_csm_vsm_load(level: u32, texel: vec2<f32>, home: RaaVsmPage) -> f32 {
    let r = raa_csm_vsm_resolve(texel, home);
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

// Cheap visibility for blended (OIT) fragments: no blocker search, the five-tap probe of the
// tent grid at a fixed 1.5-texel radius, all taps inside the home page.
fn raa_csm_vsm_lite(level: u32, uv: vec2<f32>, depth: f32, home_in: RaaVsmPage) -> f32 {
    var home = home_in;
    let local = clamp(uv - vec2<f32>(home.page * 128), vec2<f32>(2.0), vec2<f32>(126.0));
    let c = vec2<f32>(home.page * 128) + local;
    home.inside = true;
    let r = 1.5;
    return 0.2 * (raa_csm_vsm_cmp(level, c, depth, home)
        + raa_csm_vsm_cmp(level, c + vec2<f32>(r, r), depth, home)
        + raa_csm_vsm_cmp(level, c + vec2<f32>(-r, r), depth, home)
        + raa_csm_vsm_cmp(level, c + vec2<f32>(r, -r), depth, home)
        + raa_csm_vsm_cmp(level, c + vec2<f32>(-r, -r), depth, home));
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
    var home = RaaVsmPage(vec2<i32>(0), vec3<i32>(-1), false, vec2<i32>(0), vec3<i32>(-1), vec3<i32>(-1), vec3<i32>(-1));
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
            home = RaaVsmPage(page, raa_csm_vsm_slot_origin(s), false, vec2<i32>(0), vec3<i32>(-1), vec3<i32>(-1), vec3<i32>(-1));
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
    if (raa_shadow_lite) {
        return raa_csm_vsm_lite(level, uv, depth, home);
    }
    let tan_angle = max(tan(raa_frame.sun_dir.w * raa_shadow.params.x), 0.0);
    let min_texels = 1.5;
    let search_texels = clamp(max(tan_angle * raa_csm_vsm_param(1u).z / texel_m, min_texels + 1.0), 1.0, 32.0);
    // Texels from uv to the nearest border of the home page: kernels that stay inside it
    // (most pixels) address the slot directly; the others read the 2 x 2 block once.
    let local = uv - vec2<f32>(home.page * 128);
    var room = min(min(local.x, local.y), 128.0 - max(local.x, local.y)) - 1.0;
    home.inside = search_texels < room;
    var have_block = !home.inside;
    if (have_block) {
        raa_csm_vsm_block(level, uv, &home);
    }
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
    var penumbra = delta_m * tan_angle / texel_m;
    // A wide penumbra (over 16 texels) is filtered at the level 4x coarser (its pages are
    // marked for every pixel by vsm_mark.wgsl, FILTER_UP): the same world kernel over 16x
    // fewer texels, so the taps stay close in memory and the 48-texel cap no longer cuts the
    // physical penumbra.
    let up = level + RAA_CSM_VSM_FILTER_UP;
    if (penumbra > 16.0 && up < levels) {
        let up_m = raa_csm_vsm_texel_m(up);
        let lp = m * vec4<f32>(world_pos + n * (raa_shadow.params.y * up_m), 1.0);
        let up_uv = vec2<f32>(lp.x, -lp.y) / up_m;
        let page = vec2<i32>(floor(up_uv / 128.0));
        let s = raa_csm_vsm_slot(up, page);
        if (s >= 0 && lp.z > 0.0 && lp.z < 1.0) {
            home = RaaVsmPage(page, raa_csm_vsm_slot_origin(s), false, vec2<i32>(0), vec3<i32>(-1), vec3<i32>(-1), vec3<i32>(-1));
            level = up;
            uv = up_uv;
            depth = lp.z;
            penumbra = penumbra * texel_m / up_m;
            let up_local = uv - vec2<f32>(home.page * 128);
            room = min(min(up_local.x, up_local.y), 128.0 - max(up_local.x, up_local.y)) - 1.0;
            have_block = false;
        }
    }
    penumbra = clamp(penumbra, min_texels, 48.0);
    home.inside = penumbra < room;
    if (!home.inside && !have_block) {
        raa_csm_vsm_block(level, uv, &home);
    }
    if any(raa_frame.camera_jitter.xy != vec2<f32>(0.0)) {
        return raa_csm_vsm_poisson_pcf(level, uv, depth, penumbra, raa_csm_frame_rotation(frag.xy), home);
    }
    return raa_csm_vsm_grid_pcf(level, uv, depth, penumbra, home);
}
