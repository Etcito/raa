// dof: circle of confusion, autofocus and tile classification (S12 `dof`, DOF values).
// Standalone compute module (no PRELUDE), own bindings in group 0; each entry uses only
// the bindings listed on it, so each pipeline gets its own group-0 layout.
//
// Depth is the S12 `depth` target (Depth32Float, zero-to-one, right-handed view looking
// down -Z); the linear depth of a pixel is `-v.z / v.w` with `v = inv_proj * ndc`. The CoC
// is the thin-lens formula of `raa_dof::coc_px` (36 mm sensor on the viewport height):
//   coc_px = K * (z - s) / z,  K = A * f / (s - f) * 1000 / 36 * viewport_h  (lens.y)
// signed (negative in front of the focus plane = near layer), clamped to +-lens.z, and 0
// for a non-finite or non-positive depth. With autofocus the focus is the GPU distance
// s' = max(s', f + 0.001) and the aperture term is recomputed exactly at it:
//   K' = K * (s - f) / (s' - f)   (f = lens2.x; never the K * s / s' approximation)

struct DofParams {
    inv_proj: mat4x4<f32>, // @0
    lens: vec4<f32>,       // @64 (focus m, aperture term K px, max coc px at viewport_h, near m)
    lens2: vec4<f32>,      // @80 (focal length f m, 0, 0, 0)
    viewport: vec4<f32>,   // @96 (w, h, 1/w, 1/h)
    autofocus: vec4<f32>,  // @112 (u, v, enabled 0|1, blend factor per frame)
}

// All entries.
@group(0) @binding(0) var<uniform> params: DofParams;
// cs_autofocus, cs_coc: the S12 depth target (textureLoad only).
@group(0) @binding(1) var depth_tex: texture_depth_2d;
// cs_autofocus (written), cs_coc (read when autofocus is enabled): focus distance in m.
@group(0) @binding(2) var<storage, read_write> focus: f32;
// cs_coc: full-res signed CoC px.
@group(0) @binding(3) var coc_out: texture_storage_2d<r32float, write>;
// cs_tiles: the full-res CoC written by cs_coc (unfilterable float, textureLoad only).
@group(0) @binding(4) var coc_tex: texture_2d<f32>;
// cs_tiles: per 16x16 tile (min CoC, max CoC).
@group(0) @binding(5) var tiles_out: texture_storage_2d<rg32float, write>;
// cs_dilate: the tiles written by cs_tiles (textureLoad only).
@group(0) @binding(6) var tiles_tex: texture_2d<f32>;
// cs_dilate: per tile (min of the 3x3 mins, max of the 3x3 |max|).
@group(0) @binding(7) var dilated_out: texture_storage_2d<rg32float, write>;

const TILE: u32 = 16u;
const TILE_TEXELS: u32 = 256u;
// Larger than any finite depth or CoC; also the "non-finite" bound (WGSL has no isinf).
const BIG: f32 = 3.0e38;

// Linear view depth (m) at screen uv (origin top-left) and clip depth d.
fn dof_linear_depth(uv: vec2<f32>, d: f32) -> f32 {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, d, 1.0);
    let v = params.inv_proj * ndc;
    return -v.z / v.w;
}

fn dof_valid_depth(z: f32) -> bool {
    // false for NaN as well
    return z > 0.0 && z < BIG;
}

// Signed CoC px of depth z for focus s and aperture term k (see the header).
fn dof_coc(z: f32, s: f32, k: f32) -> f32 {
    if !dof_valid_depth(z) {
        return 0.0;
    }
    let px = k * (z - s) / z;
    return clamp(px, -params.lens.z, params.lens.z);
}

// Focus distance and aperture term of this frame: lens.x / lens.y, or the autofocus
// distance s' from `focus` when enabled. lens.y is computed by the host for s = lens.x; at
// s' (kept beyond the focal point) it becomes K * (s - f) / (s' - f), exactly the thin-lens
// term A * f / (s' - f).
fn dof_focus() -> vec2<f32> {
    let s = params.lens.x;
    let k = params.lens.y;
    if params.autofocus.z > 0.5 {
        let af = focus;
        if dof_valid_depth(af) {
            let f = params.lens2.x;
            let sp = max(af, f + 0.001);
            return vec2<f32>(sp, k * (s - f) / (sp - f));
        }
    }
    return vec2<f32>(s, k);
}

// Autofocus (1 thread): linear depth at the autofocus uv, smoothed per frame (never per dt:
// lookdev renders its warm-up frames at a fixed time, dt 0) with the blend factor. The host
// sends blend 1 on the first frame, so it snaps. Disabled: tracks the manual focus, so a
// later enable starts from it. Sky or invalid depth keeps the previous focus.
@compute @workgroup_size(1)
fn cs_autofocus() {
    let prev = focus;
    if params.autofocus.z < 0.5 {
        focus = params.lens.x;
        return;
    }
    let size = textureDimensions(depth_tex);
    let p = min(vec2<u32>(params.autofocus.xy * vec2<f32>(size)), size - vec2<u32>(1u));
    let uv = (vec2<f32>(p) + vec2<f32>(0.5)) / vec2<f32>(size);
    var z = dof_linear_depth(uv, textureLoad(depth_tex, p, 0));
    if !dof_valid_depth(z) {
        z = select(params.lens.x, prev, dof_valid_depth(prev));
    }
    z = max(z, params.lens.w);
    let b = clamp(params.autofocus.w, 0.0, 1.0);
    // mix() with an uninitialised (NaN, 0) previous value would poison the result
    if b >= 1.0 || !dof_valid_depth(prev) {
        focus = z;
    } else {
        focus = mix(prev, z, b);
    }
}

// Full-res signed CoC, 8x8 workgroups over the viewport.
@compute @workgroup_size(8, 8)
fn cs_coc(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(coc_out);
    if any(gid.xy >= size) {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * params.viewport.zw;
    let z = dof_linear_depth(uv, textureLoad(depth_tex, gid.xy, 0));
    let sk = dof_focus();
    textureStore(coc_out, gid.xy, vec4<f32>(dof_coc(z, sk.x, sk.y), 0.0, 0.0, 0.0));
}

var<workgroup> tile_min: array<f32, TILE_TEXELS>;
var<workgroup> tile_max: array<f32, TILE_TEXELS>;

// One workgroup per 16x16 tile (dispatch ceil(w / 16) x ceil(h / 16)): min and max signed
// CoC of the tile's pixels by a shared-memory tree reduction.
@compute @workgroup_size(16, 16)
fn cs_tiles(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let size = textureDimensions(coc_tex);
    let p = wid.xy * TILE + lid.xy;
    var lo = BIG;
    var hi = -BIG;
    if all(p < size) {
        let c = textureLoad(coc_tex, p, 0).r;
        lo = c;
        hi = c;
    }
    tile_min[li] = lo;
    tile_max[li] = hi;
    workgroupBarrier();
    for (var stride = TILE_TEXELS / 2u; stride > 0u; stride = stride >> 1u) {
        if li < stride {
            tile_min[li] = min(tile_min[li], tile_min[li + stride]);
            tile_max[li] = max(tile_max[li], tile_max[li + stride]);
        }
        workgroupBarrier();
    }
    if li == 0u && all(wid.xy < textureDimensions(tiles_out)) {
        textureStore(tiles_out, wid.xy, vec4<f32>(tile_min[0], tile_max[0], 0.0, 0.0));
    }
}

// 3x3 tile dilation, 8x8 workgroups over the tile grid: x = min of the neighbours' min
// (strongest near CoC), y = max of the neighbours' |max| (largest far blur radius).
@compute @workgroup_size(8, 8)
fn cs_dilate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(dilated_out);
    if any(gid.xy >= size) {
        return;
    }
    let last = vec2<i32>(textureDimensions(tiles_tex)) - vec2<i32>(1);
    var lo = BIG;
    var hi = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let q = clamp(vec2<i32>(gid.xy) + vec2<i32>(dx, dy), vec2<i32>(0), last);
            let t = textureLoad(tiles_tex, q, 0).rg;
            lo = min(lo, t.x);
            hi = max(hi, abs(t.y));
        }
    }
    textureStore(dilated_out, gid.xy, vec4<f32>(lo, hi, 0.0, 0.0));
}
