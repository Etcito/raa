// A CPU-authored sparse lookup and GPU-projected radiance share one read-only
// buffer at shading time. Probes float in free space (cell centre pushed out of
// geometry); a probe becomes visible once six faces were captured and projected,
// then fades in over a few frames. The enclosing shader declares
// raa_spherical_data at its own read-only binding.
//
// Header rows: 0 anchor xyz + intensity; 1 angular n, shells, table mask, debug;
// 2 clipStart, zMax, frame, channel mask; 3 x = table row of the centre probe (0 none);
// 4.. shell inner, centre, outer. Entry rows: key, slot, first publication frame,
// light hash; probe xyz + spacing; reserved xyz + published.

const RAA_SPHERICAL_TABLE: u32 = 68u;
// 21 legacy rows (uniform average, six world-axis faces, three light groups each),
// 24 parallax patches x 3 rows (hit centroid + R², intensity + coherence, miss flux +
// octahedral mean normal), 96 depth-moment bins two per row, one information row.
const RAA_SPHERICAL_SLOT_ROWS: u32 = 142u;
const RAA_SPHERICAL_PATCH_ROW: u32 = 21u;
const RAA_SPHERICAL_MOMENT_ROW: u32 = 93u;
const RAA_SPHERICAL_INFO_ROW: u32 = 141u;
const RAA_SPHERICAL_EMPTY: u32 = 0xffffffffu;
const RAA_SPHERICAL_CENTER: u32 = 0xffffu;
const RAA_SPHERICAL_FADE_FRAMES: f32 = 8.0;

fn raa_spherical_hash(key: u32) -> u32 {
    let x = (key ^ (key >> 16u)) * 0x7feb352du;
    let y = (x ^ (x >> 15u)) * 0x846ca68bu;
    return y ^ (y >> 16u);
}

fn raa_spherical_find(key: u32, mask: u32) -> u32 {
    let start = raa_spherical_hash(key) & mask;
    for (var probe = 0u; probe <= mask; probe++) {
        let entry = RAA_SPHERICAL_TABLE + 3u * ((start + probe) & mask);
        let stored = raa_spherical_data[entry].x;
        if stored == key { return entry; }
        if stored == 0u { break; }
    }
    return RAA_SPHERICAL_EMPTY;
}

fn raa_spherical_key(shell: u32, face: u32, ij: vec2<i32>, n: u32) -> u32 {
    if shell == RAA_SPHERICAL_CENTER { return 1u + (RAA_SPHERICAL_CENTER << 15u); }
    var index = ij;
    var side = face;
    if any(index < vec2<i32>(0)) || any(index >= vec2<i32>(i32(n))) {
        let uv = 2.0 * (vec2<f32>(index) + 0.5) / f32(n) - 1.0;
        let adjacent = raa_spherical_face_uv(raa_spherical_face_direction(face, uv));
        index = clamp(vec2<i32>(floor((adjacent.xy + 1.0) * (0.5 * f32(n)))),
            vec2<i32>(0), vec2<i32>(i32(n) - 1));
        side = u32(adjacent.z);
    }
    return 1u + (shell << 15u) + (side << 12u) + (u32(index.y) << 6u) + u32(index.x);
}

fn raa_spherical_published(entry: u32) -> bool {
    return entry != RAA_SPHERICAL_EMPTY && raa_spherical_data[entry + 2u].w != 0u;
}

fn raa_spherical_supported(entry: u32, p: vec3<f32>, n: vec3<f32>) -> f32 {
    if !raa_spherical_published(entry) { return 0.0; }
    let gates = raa_spherical_support_gates(entry, p, n);
    return gates.x * gates.y * gates.z;
}

fn raa_spherical_oct_decode(e: vec2<f32>) -> vec3<f32> {
    var v = vec3<f32>(e, 1.0 - abs(e.x) - abs(e.y));
    if v.z < 0.0 {
        let s = select(vec2<f32>(-1.0), vec2<f32>(1.0), v.xy >= vec2<f32>(0.0));
        v = vec3<f32>((1.0 - abs(v.yx)) * s, v.z);
    }
    return normalize(v);
}

// Chebyshev visibility of p from the probe on the stored depth moments (DDGI style),
// queried slightly off the surface along its normal.
fn raa_spherical_visibility(entry: u32, mask: u32, p: vec3<f32>, n: vec3<f32>) -> f32 {
    let probe = bitcast<vec4<f32>>(raa_spherical_data[entry + 1u]);
    let biased = p + n * max(0.02, 0.05 * probe.w);
    let offset = biased - probe.xyz;
    let distance = length(offset);
    if distance < 1e-5 { return 1.0; }
    let face_uv = raa_spherical_face_uv(offset);
    let bin = clamp(vec2<u32>((face_uv.xy + 1.0) * 2.0), vec2<u32>(0u), vec2<u32>(3u));
    let index = 16u * u32(face_uv.z) + 4u * bin.y + bin.x;
    let row = bitcast<vec4<f32>>(raa_spherical_data[raa_spherical_slot(entry, mask)
        + RAA_SPHERICAL_MOMENT_ROW + index / 2u]);
    let moments = select(row.xy, row.zw, (index & 1u) != 0u);
    if distance <= moments.x { return 1.0; }
    let variance = max(abs(moments.y - moments.x * moments.x), 1e-4 * probe.w * probe.w);
    let d = distance - moments.x;
    let v = variance / (variance + d * d);
    return v * v * v;
}

// x: probe in front of the receiver, y: visibility, z: fade-in times validity.
fn raa_spherical_support_gates(entry: u32, p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let mask = raa_spherical_data[1].z;
    let probe = bitcast<vec4<f32>>(raa_spherical_data[entry + 1u]);
    let to_probe = probe.xyz - p;
    let facing = dot(to_probe * inverseSqrt(max(dot(to_probe, to_probe), 1e-12)), n);
    let half = 0.5 * (facing + 1.0);
    let front = 0.05 + 0.95 * half * half;
    let visibility = raa_spherical_visibility(entry, mask, p, n);
    let age = f32(raa_spherical_data[2].z - raa_spherical_data[entry].z);
    let fade = smoothstep(0.0, RAA_SPHERICAL_FADE_FRAMES, age + 1.0);
    let validity = bitcast<f32>(raa_spherical_data[raa_spherical_slot(entry, mask)
        + RAA_SPHERICAL_INFO_ROW].x);
    return vec3<f32>(front, visibility, fade * clamp(validity, 0.0, 1.0));
}

fn raa_spherical_slot(entry: u32, mask: u32) -> u32 {
    let slot = raa_spherical_data[entry].y;
    return RAA_SPHERICAL_TABLE + 3u * (mask + 1u) + RAA_SPHERICAL_SLOT_ROWS * slot;
}

fn raa_spherical_incident(entry: u32, mask: u32, channels: u32) -> vec3<f32> {
    return raa_spherical_channel_sum(raa_spherical_slot(entry, mask), channels);
}

fn raa_spherical_channel_sum(base: u32, channels: u32) -> vec3<f32> {
    var rgb = vec3<f32>(0.0);
    for (var i = 0u; i < 3u; i++) {
        if (channels & (1u << i)) != 0u {
            rgb += bitcast<vec3<f32>>(raa_spherical_data[base + i].rgb);
        }
    }
    return rgb;
}

// Parallax-aware diffuse irradiance / pi of one probe at a receiver p with the
// receiver's own normal n. Each patch is a one-sided emitter at its hit centroid,
// scaled so that it reproduces the probe's captured irradiance at the probe; misses
// arrive from infinity along the patch direction.
fn raa_spherical_probe_irradiance(entry: u32, mask: u32, p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let base = raa_spherical_slot(entry, mask) + RAA_SPHERICAL_PATCH_ROW;
    var e = vec3<f32>(0.0);
    for (var k = 0u; k < 24u; k++) {
        let r0 = bitcast<vec4<f32>>(raa_spherical_data[base + 3u * k]);
        let r1 = bitcast<vec4<f32>>(raa_spherical_data[base + 3u * k + 1u]);
        let r2 = raa_spherical_data[base + 3u * k + 2u];
        if r0.w >= 0.0 {
            let delta = r0.xyz - p;
            let d2 = dot(delta, delta);
            let w = delta * inverseSqrt(max(d2, 1e-10));
            let cos_n = dot(n, w);
            if cos_n > 0.0 {
                let normal = raa_spherical_oct_decode(unpack2x16snorm(r2.w));
                let cos_h = mix(1.0, max(0.0, -dot(normal, w)), r1.w);
                e += r1.xyz * (cos_h * cos_n / (d2 + r0.w));
            }
        }
        let miss = bitcast<vec3<f32>>(r2.xyz);
        if any(miss > vec3<f32>(0.0)) {
            let corner = vec2<f32>(f32(k & 1u), f32((k >> 1u) & 1u)) - 0.5;
            let d = normalize(raa_spherical_face_direction(k / 4u, corner));
            e += miss * max(0.0, dot(n, d));
        }
    }
    return e * 0.31830988;
}

struct RaaSphericalCoordinate {
    desc: vec4<u32>,
    shell: u32,
    bounds: vec3<f32>,
    face: u32,
    angular: vec2<f32>,
    radius: f32,
}

// Below clipStart the point lies in the centre probe's sphere: it reports shell
// zero and the interpolation blends toward the centre probe.
fn raa_spherical_coordinate(p: vec3<f32>) -> RaaSphericalCoordinate {
    let anchor = bitcast<vec4<f32>>(raa_spherical_data[0]);
    if anchor.w == 0.0 { return RaaSphericalCoordinate(); }
    let desc = raa_spherical_data[1];
    let config = bitcast<vec2<f32>>(raa_spherical_data[2].xy);
    var delta = p - anchor.xyz;
    let radius = length(delta);
    if radius > config.y { return RaaSphericalCoordinate(); }
    if radius < 1e-6 { delta = vec3<f32>(1e-6, 0.0, 0.0); }
    var first = 0u;
    var last = desc.y - 1u;
    while first < last {
        let mid = (first + last) / 2u;
        if radius >= bitcast<f32>(raa_spherical_data[4u + mid].z) {
            first = mid + 1u;
        } else {
            last = mid;
        }
    }
    let shell = first;
    let shell_bounds = bitcast<vec3<f32>>(raa_spherical_data[4u + shell].xyz);
    let face_uv = raa_spherical_face_uv(delta);
    let face = u32(face_uv.z);
    let angular = (face_uv.xy + 1.0) * (0.5 * f32(desc.x));
    return RaaSphericalCoordinate(desc, shell, shell_bounds, face, angular, radius);
}

struct RaaSphericalNeighbors {
    entries: array<u32, 8>,
    weights: array<f32, 8>,
    trilinear: array<f32, 8>,
    support: array<f32, 8>,
    count: u32,
    total: f32,
    /// Trilinear corner weight of published probes times fade and validity, without
    /// facing or visibility: how well the grid covers p.
    coverage: f32,
}

struct RaaSphericalResult {
    diffuse: vec4<f32>,
    debug: vec4<f32>,
}

fn raa_spherical_evaluate(p: vec3<f32>, normal: vec3<f32>) -> RaaSphericalResult {
    let cell = raa_spherical_coordinate(p);
    if cell.desc.x == 0u { return RaaSphericalResult(); }
    let desc = cell.desc;
    let shell = cell.shell;
    let shell_bounds = cell.bounds;
    let face = cell.face;
    let angular = cell.angular;
    let radius = cell.radius;
    if desc.w != 0u {
        let ij = min(vec2<i32>(i32(desc.x) - 1), vec2<i32>(floor(angular)));
        let inner = radius < bitcast<f32>(raa_spherical_data[2].x);
        let entry = raa_spherical_find(raa_spherical_key(select(shell, RAA_SPHERICAL_CENTER, inner),
            face, ij, desc.x), desc.z);
        if desc.w == 1u {
            let cell = fract(angular);
            let side = min(min(cell.x, 1.0 - cell.x), min(cell.y, 1.0 - cell.y));
            let radial = min(radius - shell_bounds.x, shell_bounds.z - radius) / (shell_bounds.z - shell_bounds.x);
            if min(side, radial) < 0.045 { return RaaSphericalResult(vec4<f32>(0.0), vec4<f32>(0.8, 0.32, 0.02, 1.0)); }
            return RaaSphericalResult();
        }
        var rgb = vec3<f32>(0.06);
        if entry != RAA_SPHERICAL_EMPTY {
            let record = raa_spherical_data[entry];
            let ready = raa_spherical_data[entry + 2u].w != 0u;
            rgb = select(vec3<f32>(0.7, 0.28, 0.03), vec3<f32>(0.06, 0.6, 0.14), ready);
            if ready && desc.w == 3u {
                let age = min(1.0, f32(raa_spherical_data[2].z - record.z) / 180.0);
                rgb = mix(vec3<f32>(0.04, 0.65, 0.25), vec3<f32>(0.62, 0.09, 0.14), age);
            }
            if ready && desc.w == 4u {
                let hash = raa_spherical_hash(record.w);
                rgb = unpack4x8unorm(hash).rgb;
            }
        }
        return RaaSphericalResult(vec4<f32>(0.0), vec4<f32>(rgb, 1.0));
    }

    return RaaSphericalResult(raa_spherical_diffuse_at(p, normal, cell), vec4<f32>(0.0));
}

// Physical irradiance / pi stays available while forward displays a debug overlay.
// The strongest four supported probes are evaluated with parallax; their support
// weights already include visibility, so occluded probes do not leak.
fn raa_spherical_diffuse_at(p: vec3<f32>, normal: vec3<f32>,
                           cell: RaaSphericalCoordinate) -> vec4<f32> {
    let neighbors = raa_spherical_neighbors(p, normal, cell);
    if neighbors.count == 0u { return vec4<f32>(0.0); }
    let mask = cell.desc.z;
    var used = 0u;
    var picked: array<bool, 8>;
    var sum = vec3<f32>(0.0);
    var total = 0.0;
    for (var round = 0u; round < min(neighbors.count, 4u); round++) {
        var best = 0u;
        var best_weight = -1.0;
        for (var i = 0u; i < neighbors.count; i++) {
            if !picked[i] && neighbors.weights[i] > best_weight {
                best = i;
                best_weight = neighbors.weights[i];
            }
        }
        picked[best] = true;
        if best_weight <= 1e-6 { break; }
        sum += best_weight * raa_spherical_probe_irradiance(neighbors.entries[best], mask, p, normal);
        total += best_weight;
    }
    if total <= 0.0 { return vec4<f32>(0.0); }
    // Coverage ignores visibility so an occluded receiver still uses the probes
    // rather than the legacy ambient; it fades out over the outer half of the last shell.
    let coverage = neighbors.coverage;
    let last = bitcast<vec3<f32>>(raa_spherical_data[4u + cell.desc.y - 1u].xyz);
    let edge = 1.0 - smoothstep(last.y, last.z, cell.radius);
    let blend = smoothstep(0.0, 0.25, coverage) * edge;
    let intensity = bitcast<f32>(raa_spherical_data[0].w);
    return vec4<f32>(intensity * sum / total, blend);
}

struct RaaSphericalInterpolation {
    shell0: u32,
    shell1: u32,
    fraction: vec3<f32>,
    base: vec2<i32>,
}

// Radial interpolation runs between probe-shell centres, with the centre probe at
// radius zero as the innermost layer (shell0 = RAA_SPHERICAL_CENTER).
fn raa_spherical_interpolation(cell: RaaSphericalCoordinate) -> RaaSphericalInterpolation {
    let desc = cell.desc;
    let shell = cell.shell;
    let radius = cell.radius;
    var shell0 = shell;
    var shell1 = shell;
    var depth = 0.0;
    let c0 = bitcast<f32>(raa_spherical_data[4u].y);
    if shell == 0u && radius < c0 {
        shell0 = RAA_SPHERICAL_CENTER;
        shell1 = 0u;
        depth = clamp(radius / max(c0, 1e-8), 0.0, 1.0);
    } else {
        if radius < bitcast<f32>(raa_spherical_data[4u + shell].y) && shell > 0u { shell0 = shell - 1u; }
        shell1 = min(shell0 + 1u, desc.y - 1u);
        let r0 = bitcast<f32>(raa_spherical_data[4u + shell0].y);
        let r1 = bitcast<f32>(raa_spherical_data[4u + shell1].y);
        depth = clamp((radius - r0) / max(r1 - r0, 1e-8), 0.0, 1.0);
    }
    let pixel = cell.angular - 0.5;
    return RaaSphericalInterpolation(shell0, shell1, vec3<f32>(fract(pixel), depth),
        vec2<i32>(floor(pixel)));
}

fn raa_spherical_corner(index: u32) -> vec3<bool> {
    return vec3<bool>((index & 1u) != 0u, (index & 2u) != 0u, (index & 4u) != 0u);
}

fn raa_spherical_corner_key(cell: RaaSphericalCoordinate, interpolation: RaaSphericalInterpolation,
                            corner: vec3<bool>) -> u32 {
    let coord = interpolation.base + select(vec2<i32>(0), vec2<i32>(1), corner.xy);
    return raa_spherical_key(select(interpolation.shell0, interpolation.shell1, corner.z),
        cell.face, coord, cell.desc.x);
}

// Published probes around p with weight = trilinear x support (fade, validity,
// facing and depth visibility). Duplicate corners (the centre probe, cube seams)
// accumulate their trilinear weight. With poor corner coverage the immediate ring
// and the centre probe supply the nearest probes instead of an ambient fallback.
fn raa_spherical_neighbors(p: vec3<f32>, normal: vec3<f32>,
                           cell: RaaSphericalCoordinate) -> RaaSphericalNeighbors {
    if cell.desc.x == 0u { return RaaSphericalNeighbors(); }
    let interpolation = raa_spherical_interpolation(cell);
    let desc = cell.desc;
    var result: RaaSphericalNeighbors;
    var coverage = 0.0;
    for (var i = 0u; i < select(8u, 4u, interpolation.shell0 == interpolation.shell1); i++) {
        let corner = raa_spherical_corner(i);
        let key = raa_spherical_corner_key(cell, interpolation, corner);
        let entry = raa_spherical_find(key, desc.z);
        if !raa_spherical_published(entry) { continue; }
        let w = select(1.0 - interpolation.fraction, interpolation.fraction, corner);
        let trilinear = max(w.x * w.y * w.z, 0.001);
        var seen = 8u;
        for (var j = 0u; j < result.count; j++) {
            if result.entries[j] == entry { seen = j; }
        }
        if seen < 8u {
            result.trilinear[seen] += trilinear;
            let weight = trilinear * result.support[seen];
            result.weights[seen] += weight;
            result.total += weight;
            coverage += trilinear;
            continue;
        }
        let gates = raa_spherical_support_gates(entry, p, normal);
        let support = gates.x * gates.y * gates.z;
        coverage += trilinear * gates.z;
        if support <= 0.0 { continue; }
        let weight = support * trilinear;
        result.entries[result.count] = entry;
        result.weights[result.count] = weight;
        result.trilinear[result.count] = trilinear;
        result.support[result.count] = support;
        result.total += weight;
        result.count += 1u;
    }

    if coverage < 0.5 && result.count < 8u {
        let center = vec2<i32>(floor(cell.angular));
        let lo = max(i32(cell.shell) - 1, 0);
        let hi = min(i32(cell.shell) + 1, i32(desc.y) - 1);
        for (var ri = lo - 1; ri <= hi; ri++) {
            for (var y = -1; y <= 1; y++) {
                for (var x = -1; x <= 1; x++) {
                    if result.count >= 8u { break; }
                    // ri == lo - 1 visits the centre probe once.
                    if ri == lo - 1 && (x != 0 || y != 0) { continue; }
                    let shell = select(u32(max(ri, 0)), RAA_SPHERICAL_CENTER, ri == lo - 1);
                    let entry = raa_spherical_find(
                        raa_spherical_key(shell, cell.face, center + vec2<i32>(x, y), desc.x), desc.z);
                    if !raa_spherical_published(entry) { continue; }
                    var seen = false;
                    for (var j = 0u; j < result.count; j++) { seen = seen || result.entries[j] == entry; }
                    if seen { continue; }
                    let probe = bitcast<vec4<f32>>(raa_spherical_data[entry + 1u]);
                    let offset = probe.xyz - p;
                    let spacing2 = probe.w * probe.w;
                    let nearness = spacing2 / (dot(offset, offset) + spacing2);
                    let support = raa_spherical_supported(entry, p, normal);
                    if support <= 0.0 { continue; }
                    let trilinear = 0.25 * nearness;
                    coverage += trilinear;
                    let weight = support * trilinear;
                    result.entries[result.count] = entry;
                    result.weights[result.count] = weight;
                    result.trilinear[result.count] = trilinear;
                    result.support[result.count] = support;
                    result.total += weight;
                    result.count += 1u;
                }
            }
        }
    }
    result.coverage = coverage;
    return result;
}
