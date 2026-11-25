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
const RAA_DDGI_VALID: u32 = 1u;
const RAA_DDGI_INACTIVE: u32 = 2u;
const RAA_DDGI_INSIDE: u32 = 4u;

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
}

fn raa_ddgi_active() -> bool {
    return raa_ddgi_word(3u).x == RAA_DDGI_MAGIC;
}

fn raa_ddgi_volume() -> RaaDdgiVolume {
    let base = raa_ddgi_word(4u);
    let counts = raa_ddgi_word(5u);
    let tuning = bitcast<vec4<f32>>(raa_ddgi_word(6u));
    let rows = raa_ddgi_word(7u);
    let extra = bitcast<vec4<f32>>(raa_ddgi_word(8u));
    return RaaDdgiVolume(bitcast<vec3<i32>>(base.xyz), base.w, counts.xyz, tuning.x, tuning.y,
        tuning.z, tuning.w, rows.x, rows.y, rows.z, extra.x);
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
    return u32(s.x) + v.counts.x * (u32(s.y) + v.counts.y * u32(s.z));
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

// Irradiance / pi at a surface (the cosine-weighted mean incident radiance) in rgb and a
// confidence in w: the trilinear coverage of usable probes times a fade over the outer cell.
fn raa_ddgi_sample(p: vec3<f32>, n: vec3<f32>, view_dir: vec3<f32>) -> vec4<f32> {
    let v = raa_ddgi_volume();
    if any(v.counts < vec3<u32>(2u)) || v.spacing <= 0.0 { return vec4<f32>(0.0); }
    let biased = p + n * v.normal_bias + view_dir * v.view_bias;
    let local = biased / v.spacing - vec3<f32>(v.base);
    let last = vec3<f32>(v.counts) - 1.0;
    let inside = min(local, last - local);
    let edge = saturate(min(min(inside.x, inside.y), inside.z));
    if edge <= 0.0 { return vec4<f32>(0.0); }
    let cell = clamp(vec3<i32>(floor(local)), vec3<i32>(0), vec3<i32>(v.counts) - 2);
    let alpha = clamp(local - vec3<f32>(cell), vec3<f32>(0.0), vec3<f32>(1.0));
    var sum = vec3<f32>(0.0);
    var total = 0.0;
    var coverage = 0.0;
    for (var i = 0u; i < 8u; i++) {
        let corner = vec3<u32>(i & 1u, (i >> 1u) & 1u, (i >> 2u) & 1u);
        let g = v.base + cell + vec3<i32>(corner);
        let slot = raa_ddgi_slot(v, g);
        let record = raa_ddgi_word(v.meta_row + 2u * slot);
        if any(bitcast<vec3<i32>>(record.xyz) != g) || (record.w & RAA_DDGI_VALID) == 0u
            || (record.w & RAA_DDGI_INACTIVE) != 0u {
            continue;
        }
        let offset = bitcast<vec3<f32>>(raa_ddgi_word(v.meta_row + 2u * slot + 1u).xyz);
        let probe = vec3<f32>(g) * v.spacing + offset;
        let tri = mix(1.0 - alpha, alpha, vec3<f32>(corner));
        let trilinear = tri.x * tri.y * tri.z;
        // Smooth backface: probes behind the surface fade without a hard cut.
        let to_probe = normalize(probe - p);
        var weight = 0.5 * (dot(to_probe, n) + 1.0);
        weight = weight * weight + 0.2;
        // Chebyshev visibility from the probe's mean distance and its square.
        let from_probe = biased - probe;
        let distance = length(from_probe);
        let moments = raa_ddgi_depth(v, slot, from_probe / max(distance, 1e-4));
        if distance > moments.x {
            let variance = abs(moments.x * moments.x - moments.y);
            let gap = distance - moments.x;
            var visible = variance / max(variance + gap * gap, 1e-8);
            visible = visible * visible * visible;
            weight *= max(0.05, visible);
        }
        weight = max(weight, 1e-6);
        if weight < 0.2 { weight *= weight * weight / 0.04; }
        weight *= trilinear;
        let irradiance = raa_ddgi_irradiance(v, slot, n);
        sum += sqrt(max(irradiance, vec3<f32>(0.0))) * weight;
        total += weight;
        coverage += trilinear;
    }
    if total <= 0.0 { return vec4<f32>(0.0); }
    let root = sum / total;
    return vec4<f32>(root * root * v.intensity, edge * saturate(4.0 * coverage));
}

// The screen-resolved value at a forward pixel (row 9: first row, width, height, scale): a
// joint-bilateral blend of the four nearest resolve texels, keeping those resolved for a surface
// at this camera distance (within 2%); the full eight-probe lookup when none was.
fn raa_ddgi_screen_or_sample(p: vec3<f32>, n: vec3<f32>, view_dir: vec3<f32>, frag: vec2<f32>,
                             eye: vec3<f32>) -> vec4<f32> {
    let screen = raa_ddgi_word(9u);
    if screen.w != 0u {
        let distance = length(p - eye);
        let at = frag / f32(screen.w) - 0.5;
        let corner = floor(at);
        let f = at - corner;
        let size = vec2<i32>(i32(screen.y), i32(screen.z));
        var sum = vec4<f32>(0.0);
        var total = 0.0;
        for (var i = 0u; i < 4u; i++) {
            let o = vec2<u32>(i & 1u, i >> 1u);
            let texel = clamp(vec2<i32>(corner) + vec2<i32>(o), vec2<i32>(0), size - 1);
            let row = raa_ddgi_word(screen.x + u32(texel.y) * screen.y + u32(texel.x));
            if row.w == 0u || abs(bitcast<f32>(row.z) - distance) > 0.02 * distance + 0.002 {
                continue;
            }
            let w = select(1.0 - f.x, f.x, o.x == 1u) * select(1.0 - f.y, f.y, o.y == 1u) + 1e-3;
            let ba = unpack2x16float(row.y);
            sum += w * vec4<f32>(unpack2x16float(row.x), ba.x, ba.y);
            total += w;
        }
        if total > 0.0 { return sum / total; }
    }
    return raa_ddgi_sample(p, n, view_dir);
}

// Same contract as raa_spherical_evaluate: rgb is irradiance / pi, w the blend weight.
fn raa_ddgi_evaluate(p: vec3<f32>, n: vec3<f32>, view_dir: vec3<f32>, frag: vec2<f32>,
                     eye: vec3<f32>) -> RaaSphericalResult {
    if !raa_ddgi_active() { return RaaSphericalResult(); }
    let sample = raa_ddgi_screen_or_sample(p, n, view_dir, frag, eye);
    let debug = (raa_ddgi_word(4u).w >> 8u) & 0xffu;
    if debug == 1u { return RaaSphericalResult(sample, vec4<f32>(sample.rgb, 1.0)); }
    if debug == 2u {
        return RaaSphericalResult(sample, vec4<f32>(1.0 - sample.w, sample.w, 0.1, 1.0));
    }
    return RaaSphericalResult(sample, vec4<f32>(0.0));
}
