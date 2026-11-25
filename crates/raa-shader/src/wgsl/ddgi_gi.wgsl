// Dynamic diffuse GI (Majercik et al. 2019) probe volume lookup.
// The enclosing shader defines `fn raa_ddgi_word(index: u32) -> vec4<u32>` over the
// one storage buffer that holds the header, the per-probe records and both atlases.
// Row 0 stays zero so a spherical lookup reading the same binding finds no grid.

const RAA_DDGI_MAGIC: u32 = 0x49474444u; // "DDGI"
const RAA_DDGI_IRR: u32 = 8u;
const RAA_DDGI_IRR_TILE: u32 = 10u;
const RAA_DDGI_IRR_TEXELS: u32 = 100u;
const RAA_DDGI_DEPTH: u32 = 16u;
const RAA_DDGI_DEPTH_TILE: u32 = 18u;
const RAA_DDGI_DEPTH_TEXELS: u32 = 324u;
// Rows of a probe record: coordinate and state, offset, light mask (2 rows, 256 bits).
const RAA_DDGI_META_ROWS: u32 = 4u;
const RAA_DDGI_VALID: u32 = 1u;
const RAA_DDGI_INACTIVE: u32 = 2u;
const RAA_DDGI_INSIDE: u32 = 4u;
// Header flags (row 4.w): screen AO multiplies the DDGI term; legacy far fade to ambient.
const RAA_DDGI_SCREEN_AO: u32 = 0x10000u;
const RAA_DDGI_FAR_AMBIENT: u32 = 0x20000u;
// Cascade blend band, in cells from an open side (see schedule.rs BAND_START, BAND_WIDTH).
const RAA_DDGI_BAND_START: f32 = 0.5;
const RAA_DDGI_BAND_WIDTH: f32 = 1.5;
const RAA_DDGI_MAX_CASCADES: u32 = 3u;

struct RaaDdgiVolume {
    base: vec3<i32>,
    flags: u32,
    counts: vec3<u32>,
    spacing: f32,
    intensity: f32,
    normal_bias: f32,
    view_bias: f32,
    meta_row: u32,
    irr_row: u32,
    depth_row: u32,
    max_depth: f32,
    // First slot of the cascade, and its sides that reach the scene bounds (bit 2i low side of
    // axis i, 2i + 1 high side).
    offset: u32,
    closed: u32,
}

fn raa_ddgi_active() -> bool {
    return raa_ddgi_word(3u).x == RAA_DDGI_MAGIC;
}

// Cascade 0 (rows 4-8).
fn raa_ddgi_volume() -> RaaDdgiVolume {
    let base = raa_ddgi_word(4u);
    let counts = raa_ddgi_word(5u);
    let tuning = bitcast<vec4<f32>>(raa_ddgi_word(6u));
    let rows = raa_ddgi_word(7u);
    let extra = raa_ddgi_word(8u);
    return RaaDdgiVolume(bitcast<vec3<i32>>(base.xyz), base.w, counts.xyz, tuning.x, tuning.y,
        tuning.z, tuning.w, rows.x, rows.y, rows.z, bitcast<f32>(extra.x), 0u, extra.z);
}

fn raa_ddgi_cascades() -> u32 {
    return clamp(raa_ddgi_word(8u).y, 1u, RAA_DDGI_MAX_CASCADES);
}

// Cascade k: rows 11 + 2(k - 1) (base, first slot | closed << 16) and 12 + 2(k - 1) (counts,
// spacing); biases and the distance cap scale with the spacing.
fn raa_ddgi_volume_at(k: u32) -> RaaDdgiVolume {
    var v = raa_ddgi_volume();
    if k == 0u { return v; }
    let a = raa_ddgi_word(9u + 2u * k);
    let b = raa_ddgi_word(10u + 2u * k);
    let spacing = bitcast<f32>(b.w);
    let scale = spacing / max(v.spacing, 1e-6);
    v.base = bitcast<vec3<i32>>(a.xyz);
    v.counts = b.xyz;
    v.spacing = spacing;
    v.normal_bias *= scale;
    v.view_bias *= scale;
    v.max_depth *= scale;
    v.offset = a.w & 0xffffu;
    v.closed = a.w >> 16u;
    return v;
}

// The cascade that owns a slot.
fn raa_ddgi_cascade_of(slot: u32) -> u32 {
    var k = 0u;
    for (var c = 1u; c < raa_ddgi_cascades(); c++) {
        if slot >= (raa_ddgi_word(9u + 2u * c).w & 0xffffu) { k = c; }
    }
    return k;
}

fn raa_ddgi_sign(v: vec2<f32>) -> vec2<f32> {
    return select(vec2<f32>(-1.0), vec2<f32>(1.0), v >= vec2<f32>(0.0));
}

fn raa_ddgi_oct_encode(n: vec3<f32>) -> vec2<f32> {
    let l1 = abs(n.x) + abs(n.y) + abs(n.z);
    var uv = n.xy / max(l1, 1e-8);
    if n.z < 0.0 {
        uv = (1.0 - abs(uv.yx)) * raa_ddgi_sign(uv);
    }
    return uv;
}

fn raa_ddgi_oct_decode(uv: vec2<f32>) -> vec3<f32> {
    var n = vec3<f32>(uv, 1.0 - abs(uv.x) - abs(uv.y));
    if n.z < 0.0 {
        n = vec3<f32>((1.0 - abs(n.yx)) * raa_ddgi_sign(n.xy), n.z);
    }
    return normalize(n);
}

// Toroidal slot of a world probe coordinate: a scrolled volume keeps every probe that stays inside.
fn raa_ddgi_slot(v: RaaDdgiVolume, g: vec3<i32>) -> u32 {
    let n = vec3<i32>(v.counts);
    let s = ((g % n) + n) % n;
    return v.offset + u32(s.x) + v.counts.x * (u32(s.y) + v.counts.y * u32(s.z));
}

fn raa_ddgi_irr_texel(v: RaaDdgiVolume, slot: u32, xy: vec2<u32>) -> vec3<f32> {
    let t = slot * RAA_DDGI_IRR_TEXELS + xy.y * RAA_DDGI_IRR_TILE + xy.x;
    let row = raa_ddgi_word(v.irr_row + t / 2u);
    let pair = select(row.xy, row.zw, (t & 1u) == 1u);
    return vec3<f32>(unpack2x16float(pair.x), unpack2x16float(pair.y).x);
}

fn raa_ddgi_depth_texel(v: RaaDdgiVolume, slot: u32, xy: vec2<u32>) -> vec2<f32> {
    let t = slot * RAA_DDGI_DEPTH_TEXELS + xy.y * RAA_DDGI_DEPTH_TILE + xy.x;
    let row = raa_ddgi_word(v.depth_row + t / 4u);
    return unpack2x16float(row[t & 3u]);
}

// Bilinear position inside a bordered octahedral tile of `interior` texels per side.
struct RaaDdgiTap {
    i0: vec2<u32>,
    i1: vec2<u32>,
    f: vec2<f32>,
}

fn raa_ddgi_tap(dir: vec3<f32>, interior: f32) -> RaaDdgiTap {
    let pixel = (raa_ddgi_oct_encode(dir) * 0.5 + 0.5) * interior + 1.0;
    let c = clamp(pixel - 0.5, vec2<f32>(0.0), vec2<f32>(interior + 0.999));
    let i0 = vec2<u32>(floor(c));
    return RaaDdgiTap(i0, min(i0 + 1u, vec2<u32>(u32(interior) + 1u)), fract(c));
}

fn raa_ddgi_irradiance(v: RaaDdgiVolume, slot: u32, dir: vec3<f32>) -> vec3<f32> {
    let t = raa_ddgi_tap(dir, f32(RAA_DDGI_IRR));
    let a = raa_ddgi_irr_texel(v, slot, t.i0);
    let b = raa_ddgi_irr_texel(v, slot, vec2<u32>(t.i1.x, t.i0.y));
    let c = raa_ddgi_irr_texel(v, slot, vec2<u32>(t.i0.x, t.i1.y));
    let d = raa_ddgi_irr_texel(v, slot, t.i1);
    return mix(mix(a, b, t.f.x), mix(c, d, t.f.x), t.f.y);
}

fn raa_ddgi_depth(v: RaaDdgiVolume, slot: u32, dir: vec3<f32>) -> vec2<f32> {
    let t = raa_ddgi_tap(dir, f32(RAA_DDGI_DEPTH));
    let a = raa_ddgi_depth_texel(v, slot, t.i0);
    let b = raa_ddgi_depth_texel(v, slot, vec2<u32>(t.i1.x, t.i0.y));
    let c = raa_ddgi_depth_texel(v, slot, vec2<u32>(t.i0.x, t.i1.y));
    let d = raa_ddgi_depth_texel(v, slot, t.i1);
    return mix(mix(a, b, t.f.x), mix(c, d, t.f.x), t.f.y);
}

// Surface probes: a pool of extra probe slots after the cascades, on a sparse lattice (at most
// one probe per cell, anywhere inside it), found through a table of one row per pool slot (cell
// xyz, live | start offset << 1) with two hashed slot choices per cell. Header row 15: table row,
// pool size, first slot, lattice pitch.
struct RaaDdgiExtra {
    table_row: u32,
    count: u32,
    first: u32,
    spacing: f32,
}

fn raa_ddgi_extra() -> RaaDdgiExtra {
    let r = raa_ddgi_word(15u);
    return RaaDdgiExtra(r.x, r.y, r.z, bitcast<f32>(r.w));
}

const RAA_DDGI_NO_SLOT: u32 = 0xffffffffu;
const RAA_DDGI_SECOND_CHOICE: u32 = 0x51ed270bu;

// Cell hash (the CPU mirrors it: ddgi/extras.rs `hash`).
fn raa_ddgi_hash(c: vec3<i32>, salt: u32) -> u32 {
    let u = bitcast<vec3<u32>>(c);
    var h = u.x * 0x9e3779b1u + salt;
    h = (h ^ (h >> 15u)) * 0x85ebca77u + u.y * 0xc2b2ae3du;
    h = (h ^ (h >> 13u)) * 0x27d4eb2fu + u.z * 0x165667b1u;
    h = (h ^ (h >> 16u)) * 0x9e3779b1u;
    return h ^ (h >> 15u);
}

// The slot of the surface probe of lattice cell `c`, or RAA_DDGI_NO_SLOT.
fn raa_ddgi_extra_slot(e: RaaDdgiExtra, c: vec3<i32>) -> u32 {
    for (var j = 0u; j < 2u; j++) {
        let i = raa_ddgi_hash(c, j * RAA_DDGI_SECOND_CHOICE) % e.count;
        let row = raa_ddgi_word(e.table_row + i);
        if (row.w & 1u) != 0u && all(bitcast<vec3<i32>>(row.xyz) == c) { return e.first + i; }
    }
    return RAA_DDGI_NO_SLOT;
}

// The pool as a volume: same rows as the cascades, biases and distance cap at its own pitch.
fn raa_ddgi_extra_volume(e: RaaDdgiExtra) -> RaaDdgiVolume {
    var v = raa_ddgi_volume();
    let scale = e.spacing / max(v.spacing, 1e-6);
    v.base = vec3<i32>(0);
    v.spacing = e.spacing;
    v.normal_bias *= scale;
    v.view_bias *= scale;
    v.max_depth *= scale;
    v.offset = e.first;
    v.closed = 63u;
    return v;
}

// Weight of one probe at `probe` for the surface point `p` (smooth backface x Chebyshev
// visibility from `biased`, small weights crushed) and how well it serves that surface: 1 for a
// probe in front of the surface, visible from it and within the quality reach (header row 3.z,
// metres, 0 = any distance), falling to 0 behind it, hidden or out of reach.
struct RaaDdgiProbeWeight {
    weight: f32,
    quality: f32,
}

fn raa_ddgi_probe_weight(v: RaaDdgiVolume, slot: u32, probe: vec3<f32>, p: vec3<f32>,
                         n: vec3<f32>, biased: vec3<f32>, chebyshev: bool) -> RaaDdgiProbeWeight {
    // Smooth backface: probes behind the surface fade without a hard cut.
    let to_probe = probe - p;
    let range = length(to_probe);
    let facing = dot(to_probe / max(range, 1e-4), n);
    var weight = 0.5 * (facing + 1.0);
    weight = weight * weight + 0.2;
    var visible = 1.0;
    if chebyshev {
        // Chebyshev visibility from the probe's mean distance and its square.
        let from_probe = biased - probe;
        let distance = length(from_probe);
        let moments = raa_ddgi_depth(v, slot, from_probe / max(distance, 1e-4));
        if distance > moments.x {
            let variance = abs(moments.x * moments.x - moments.y);
            let gap = distance - moments.x;
            visible = variance / max(variance + gap * gap, 1e-8);
            visible = visible * visible * visible;
            weight *= max(0.05, visible);
        }
    }
    weight = max(weight, 1e-6);
    if weight < 0.2 { weight *= weight * weight / 0.04; }
    let reach = bitcast<f32>(raa_ddgi_word(3u).z);
    var near = 1.0;
    if reach > 0.0 { near = saturate(2.0 - 2.0 * range / reach); }
    return RaaDdgiProbeWeight(weight, saturate(2.0 * facing) * visible * near);
}

// The eight probes of one cascade around a surface point and their weights (trilinear, smooth
// backface, Chebyshev visibility, small weights crushed), with the trilinear coverage of usable
// probes and the cascade's own weight `band`:
// - an inner cascade fades over RAA_DDGI_BAND_* cells from its open sides (none where it reaches
//   the scene bounds), the next cascade taking over;
// - the last cascade, with `clamp`, takes points beyond it from its border probes (trilinear at
//   the clamped position, the normal-oriented weights only);
// - legacy (`clamp` off on the last cascade): fades to zero over its outer half cell.
// Inactive probes (classified "no surface in my cell", which 64 rays decide at random near small
// geometry such as a pillar foot) are used only when no active probe is: otherwise all eight can
// be inactive and the point got nothing (the flat ambient term in forward). Inactive probes stay
// valid, refreshed one turn in eight. Probes inside geometry are never used.
struct RaaDdgiCorners {
    slot: array<u32, 8>,
    weight: array<f32, 8>,
    total: f32,
    coverage: f32,
    band: f32,
    // The weights come from inactive probes (no active probe was usable).
    filled: bool,
    // Trilinear mass of the probes that serve the surface well (see raa_ddgi_probe_weight).
    quality: f32,
}

// The surface probes of the eight lattice cells around a surface point: trilinear over the cell
// centres (continuous: a cell's weight vanishes one pitch away from its centre whatever the
// position of its probe inside the cell) x the weight of the probe where it really is.
fn raa_ddgi_extra_corners(e: RaaDdgiExtra, v: RaaDdgiVolume, p: vec3<f32>, n: vec3<f32>,
                          view_dir: vec3<f32>) -> RaaDdgiCorners {
    var out: RaaDdgiCorners;
    out.band = 1.0;
    let biased = p + n * v.normal_bias + view_dir * v.view_bias;
    let local = biased / v.spacing - 0.5;
    let cell = vec3<i32>(floor(local));
    let alpha = local - floor(local);
    for (var i = 0u; i < 8u; i++) {
        let corner = vec3<u32>(i & 1u, (i >> 1u) & 1u, (i >> 2u) & 1u);
        let c = cell + vec3<i32>(corner);
        let slot = raa_ddgi_extra_slot(e, c);
        if slot == RAA_DDGI_NO_SLOT { continue; }
        out.slot[i] = slot;
        let record = raa_ddgi_word(v.meta_row + RAA_DDGI_META_ROWS * slot);
        if any(bitcast<vec3<i32>>(record.xyz) != c) || (record.w & RAA_DDGI_VALID) == 0u
            || (record.w & RAA_DDGI_INSIDE) != 0u {
            continue;
        }
        let offset = bitcast<vec3<f32>>(raa_ddgi_word(v.meta_row + RAA_DDGI_META_ROWS * slot + 1u).xyz);
        let probe = (vec3<f32>(c) + 0.5) * v.spacing + offset;
        let tri = mix(1.0 - alpha, alpha, vec3<f32>(corner));
        let trilinear = tri.x * tri.y * tri.z;
        let w = raa_ddgi_probe_weight(v, slot, probe, p, n, biased, true);
        out.weight[i] = w.weight * trilinear;
        out.total += out.weight[i];
        out.coverage += trilinear;
        out.quality += trilinear * w.quality;
    }
    return out;
}

// Share of a lookup taken by the surface probes around it: full once their weights sum to half
// a probe, fading continuously to the cascades where they thin out.
fn raa_ddgi_extra_share(c: RaaDdgiCorners) -> f32 {
    return saturate(2.0 * c.total);
}

fn raa_ddgi_corners(v: RaaDdgiVolume, p: vec3<f32>, n: vec3<f32>, view_dir: vec3<f32>,
                    last: bool, clamp_far: bool) -> RaaDdgiCorners {
    var out: RaaDdgiCorners;
    if any(v.counts < vec3<u32>(2u)) || v.spacing <= 0.0 { return out; }
    let biased = p + n * v.normal_bias + view_dir * v.view_bias;
    let local = biased / v.spacing - vec3<f32>(v.base);
    let top = vec3<f32>(v.counts) - 1.0;
    let lo = select(local, vec3<f32>(1e9), vec3<bool>((v.closed & 1u) != 0u, (v.closed & 4u) != 0u,
        (v.closed & 16u) != 0u));
    let hi = select(top - local, vec3<f32>(1e9), vec3<bool>((v.closed & 2u) != 0u,
        (v.closed & 8u) != 0u, (v.closed & 32u) != 0u));
    let inside = min(min(min(lo.x, lo.y), min(lo.z, hi.x)), min(hi.y, hi.z));
    var outside = false;
    if last && clamp_far {
        out.band = 1.0;
        outside = any(local < vec3<f32>(0.0)) || any(local > top);
    } else if last {
        // Legacy: fade over the outer half cell whatever the scene bounds.
        let edge = min(local, top - local);
        out.band = saturate(2.0 * min(min(edge.x, edge.y), edge.z));
    } else {
        out.band = saturate((inside - RAA_DDGI_BAND_START) / RAA_DDGI_BAND_WIDTH);
    }
    if out.band <= 0.0 { return out; }
    let cell = clamp(vec3<i32>(floor(local)), vec3<i32>(0), vec3<i32>(v.counts) - 2);
    let alpha = clamp(local - vec3<f32>(cell), vec3<f32>(0.0), vec3<f32>(1.0));
    var spare_total = 0.0;
    var spare_coverage = 0.0;
    for (var i = 0u; i < 8u; i++) {
        let corner = vec3<u32>(i & 1u, (i >> 1u) & 1u, (i >> 2u) & 1u);
        let g = v.base + cell + vec3<i32>(corner);
        let slot = raa_ddgi_slot(v, g);
        out.slot[i] = slot;
        let record = raa_ddgi_word(v.meta_row + RAA_DDGI_META_ROWS * slot);
        if any(bitcast<vec3<i32>>(record.xyz) != g) || (record.w & RAA_DDGI_VALID) == 0u
            || (record.w & RAA_DDGI_INSIDE) != 0u {
            continue;
        }
        let inactive = (record.w & RAA_DDGI_INACTIVE) != 0u;
        if inactive && !clamp_far { continue; }
        let offset = bitcast<vec3<f32>>(raa_ddgi_word(v.meta_row + RAA_DDGI_META_ROWS * slot + 1u).xyz);
        let probe = vec3<f32>(g) * v.spacing + offset;
        let tri = mix(1.0 - alpha, alpha, vec3<f32>(corner));
        let trilinear = tri.x * tri.y * tri.z;
        let w = raa_ddgi_probe_weight(v, slot, probe, p, n, biased, !outside);
        var weight = w.weight;
        weight *= trilinear;
        if inactive {
            // Kept negative until the end: used only if no active probe is.
            out.weight[i] = -weight;
            spare_total += weight;
            spare_coverage += trilinear;
            continue;
        }
        out.weight[i] = weight;
        out.total += weight;
        out.coverage += trilinear;
        out.quality += trilinear * w.quality;
    }
    let fill = out.total <= 0.0 && spare_total > 0.0;
    for (var i = 0u; i < 8u; i++) {
        out.weight[i] = select(max(out.weight[i], 0.0), max(-out.weight[i], 0.0), fill);
    }
    if fill {
        out.total = spare_total;
        out.coverage = spare_coverage;
        out.filled = true;
    }
    return out;
}

// Weight of one cascade's estimate. Default far mode: its band alone, as soon as one probe is
// usable (the estimate is renormalized over the usable probes). Weighting by the probe coverage
// handed about half the weight of every floor and wall foot (the probes below the floor are
// inside geometry: coverage ~0.5) to the coarser, leakier cascade: a light halo at pillar feet
// that came and went with the coarse probes' state (Sponza V4). Legacy: band x coverage.
fn raa_ddgi_confidence(c: RaaDdgiCorners, clamp_far: bool) -> f32 {
    if c.total <= 0.0 { return 0.0; }
    if clamp_far { return c.band; }
    return c.band * saturate(4.0 * c.coverage);
}

// Irradiance / pi at a surface (the cosine-weighted mean incident radiance) in rgb and a
// confidence in w. Cascades from the finest: each takes the weight its confidence leaves to the
// coarser ones. With the default far mode the result is renormalized and w is 1 wherever any
// probe is usable (no flat-ambient remainder); legacy mode keeps the faded confidence.
// Per-sample diagnostics of the last raa_ddgi_sample (debug "probes"): weight taken from
// cascade 0, from the coarser cascades, and 1 where inactive probes filled a hole.
var<private> raa_ddgi_diag: vec3<f32>;
// Of the last lookup (raa_ddgi_sample, ddgi_bounce): quality of the cascades' estimate (see
// raa_ddgi_probe_weight; 1 when no cascade was needed) and the share the surface probes took.
var<private> raa_ddgi_quality: vec2<f32>;

fn raa_ddgi_sample(p: vec3<f32>, n: vec3<f32>, view_dir: vec3<f32>) -> vec4<f32> {
    raa_ddgi_diag = vec3<f32>(0.0);
    let v0 = raa_ddgi_volume();
    let clamp_far = (v0.flags & RAA_DDGI_FAR_AMBIENT) == 0u;
    let cascades = raa_ddgi_cascades();
    var rgb = vec3<f32>(0.0);
    var weight = 0.0;
    var remaining = 1.0;
    raa_ddgi_quality = vec2<f32>(0.0);
    let e = raa_ddgi_extra();
    if e.count > 0u {
        let v = raa_ddgi_extra_volume(e);
        let c = raa_ddgi_extra_corners(e, v, p, n, view_dir);
        let share = raa_ddgi_extra_share(c);
        if share > 0.0 {
            var sum = vec3<f32>(0.0);
            for (var i = 0u; i < 8u; i++) {
                if c.weight[i] <= 0.0 { continue; }
                sum += sqrt(max(raa_ddgi_irradiance(v, c.slot[i], n), vec3<f32>(0.0))) * c.weight[i];
            }
            let root = sum / c.total;
            rgb += share * root * root;
            weight += share;
            remaining -= share;
            raa_ddgi_quality.y = share;
        }
    }
    var grid = 0.0;
    var grid_quality = 0.0;
    for (var k = 0u; k < cascades; k++) {
        if remaining <= 1e-3 { break; }
        let v = raa_ddgi_volume_at(k);
        let c = raa_ddgi_corners(v, p, n, view_dir, k + 1u == cascades, clamp_far);
        let confidence = raa_ddgi_confidence(c, clamp_far);
        if confidence <= 0.0 { continue; }
        grid += remaining * confidence;
        grid_quality += remaining * confidence * c.quality;
        var sum = vec3<f32>(0.0);
        for (var i = 0u; i < 8u; i++) {
            if c.weight[i] <= 0.0 { continue; }
            let irradiance = raa_ddgi_irradiance(v, c.slot[i], n);
            sum += sqrt(max(irradiance, vec3<f32>(0.0))) * c.weight[i];
        }
        let root = sum / c.total;
        let share = remaining * confidence;
        if k == 0u { raa_ddgi_diag.x += share; } else { raa_ddgi_diag.y += share; }
        if c.filled { raa_ddgi_diag.z = 1.0; }
        rgb += share * root * root;
        weight += share;
        remaining -= share;
    }
    raa_ddgi_quality.x = select(1.0, grid_quality / max(grid, 1e-6), grid > 1e-3);
    if grid <= 1e-3 && weight <= 0.0 { raa_ddgi_quality.x = 0.0; }
    if weight <= 0.0 { return vec4<f32>(0.0); }
    if clamp_far { return vec4<f32>(rgb / weight * v0.intensity, 1.0); }
    return vec4<f32>(rgb * v0.intensity / weight, weight);
}

// The screen-resolved value at a forward pixel (row 9: first row, width, height, scale): a
// joint-bilateral blend of the four nearest resolve texels, keeping those resolved for a surface
// at this camera distance (within 2%), else any resolved neighbour (`raa_ddgi_fallback` set).
// Forward never runs the eight-probe lookup itself: compiled into the forward shader, even an
// untaken branch to it cost 0.4-2.5 ms through register pressure (Sponza, RTX 3090).
var<private> raa_ddgi_fallback: bool;

fn raa_ddgi_screen(p: vec3<f32>, frag: vec2<f32>, eye: vec3<f32>) -> vec4<f32> {
    raa_ddgi_fallback = false;
    let screen = raa_ddgi_word(9u);
    if screen.w == 0u { return vec4<f32>(0.0); }
    let distance = length(p - eye);
    let at = frag / f32(screen.w) - 0.5;
    let corner = floor(at);
    let f = at - corner;
    let size = vec2<i32>(i32(screen.y), i32(screen.z));
    var sum = vec4<f32>(0.0);
    var total = 0.0;
    var any_sum = vec4<f32>(0.0);
    var any_total = 0.0;
    for (var i = 0u; i < 4u; i++) {
        let o = vec2<u32>(i & 1u, i >> 1u);
        let texel = clamp(vec2<i32>(corner) + vec2<i32>(o), vec2<i32>(0), size - 1);
        let row = raa_ddgi_word(screen.x + u32(texel.y) * screen.y + u32(texel.x));
        if row.w == 0u { continue; }
        let w = select(1.0 - f.x, f.x, o.x == 1u) * select(1.0 - f.y, f.y, o.y == 1u) + 1e-3;
        let ba = unpack2x16float(row.y);
        let value = vec4<f32>(unpack2x16float(row.x), ba.x, ba.y);
        any_sum += w * value;
        any_total += w;
        if abs(bitcast<f32>(row.z) - distance) <= 0.02 * distance + 0.002 {
            sum += w * value;
            total += w;
        }
    }
    if total > 0.0 { return sum / total; }
    raa_ddgi_fallback = true;
    if any_total > 0.0 { return any_sum / any_total; }
    return vec4<f32>(0.0);
}

// Whether the DDGI term takes the screen AO (the pass's combined material and GTAO occlusion).
fn raa_ddgi_screen_ao() -> bool {
    return (raa_ddgi_word(4u).w & RAA_DDGI_SCREEN_AO) != 0u;
}

// Same contract as raa_spherical_evaluate: rgb is irradiance / pi, w the blend weight.
// Forward / VG entry: reads the screen resolve (see raa_ddgi_screen); `n` and `view_dir` are
// unused there and kept for the shared call shape.
fn raa_ddgi_evaluate(p: vec3<f32>, n: vec3<f32>, view_dir: vec3<f32>, frag: vec2<f32>,
                     eye: vec3<f32>) -> RaaSphericalResult {
    if !raa_ddgi_active() { return RaaSphericalResult(); }
    let sample = raa_ddgi_screen(p, frag, eye);
    let debug = (raa_ddgi_word(4u).w >> 8u) & 0xffu;
    if debug == 1u || debug >= 4u { return RaaSphericalResult(sample, vec4<f32>(sample.rgb, 1.0)); }
    if debug == 2u {
        return RaaSphericalResult(sample, vec4<f32>(1.0 - sample.w, sample.w, 0.1, 1.0));
    }
    if debug == 3u {
        let fallback = f32(raa_ddgi_fallback);
        return RaaSphericalResult(sample, vec4<f32>(fallback, 1.0 - fallback, 0.1, 1.0));
    }
    return RaaSphericalResult(sample, vec4<f32>(0.0));
}
