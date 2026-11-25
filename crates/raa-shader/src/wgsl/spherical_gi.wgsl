// A CPU-authored sparse lookup and GPU-projected radiance share one read-only
// buffer at shading time. Slots become visible only after six submitted faces.
// The enclosing shader declares raa_spherical_data at its own read-only binding.

const RAA_SPHERICAL_TABLE: u32 = 68u;
const RAA_SPHERICAL_SLOT_ROWS: u32 = 21u; // projected RGB and six directional faces, each with three light groups
const RAA_SPHERICAL_EMPTY: u32 = 0xffffffffu;

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

fn raa_spherical_supported(entry: u32, p: vec3<f32>, n: vec3<f32>) -> f32 {
    if entry == RAA_SPHERICAL_EMPTY || raa_spherical_data[entry + 2u].w == 0u { return 0.0; }
    let gates = raa_spherical_support_gates(entry, p, n);
    return gates.x * gates.y * gates.z;
}

fn raa_spherical_support_gates(entry: u32, p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let surface = bitcast<vec4<f32>>(raa_spherical_data[entry + 1u]);
    let normal = bitcast<vec3<f32>>(raa_spherical_data[entry + 2u].xyz);
    let delta = (p - surface.xyz) / surface.w;
    let facing = dot(normal, n);
    // Reject opposite sides and use a tighter distance gate across the surface.
    let separation = abs(dot(delta, normal));
    return vec3<f32>(smoothstep(0.4, 0.9, facing),
        1.0 - smoothstep(0.18, 0.65, separation),
        1.0 - smoothstep(0.7, 2.5, length(delta)));
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

struct RaaSphericalCoordinate {
    desc: vec4<u32>,
    shell: u32,
    bounds: vec3<f32>,
    face: u32,
    angular: vec2<f32>,
    radius: f32,
}

fn raa_spherical_coordinate(p: vec3<f32>) -> RaaSphericalCoordinate {
    let anchor = bitcast<vec4<f32>>(raa_spherical_data[0]);
    if anchor.w == 0.0 { return RaaSphericalCoordinate(); }
    let desc = raa_spherical_data[1];
    let config = bitcast<vec2<f32>>(raa_spherical_data[2].xy);
    let delta = p - anchor.xyz;
    let radius = length(delta);
    if radius <= 0.0 || radius < config.x || radius > config.y { return RaaSphericalCoordinate(); }
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
        let entry = raa_spherical_find(raa_spherical_key(shell, face, ij, desc.x), desc.z);
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

// Physical irradiance stays available while forward displays a debug overlay.
fn raa_spherical_diffuse_at(p: vec3<f32>, normal: vec3<f32>,
                           cell: RaaSphericalCoordinate) -> vec4<f32> {
    let neighbors = raa_spherical_neighbors(p, normal, cell);
    if neighbors.count < 2u { return vec4<f32>(0.0); }
    let channels = raa_spherical_data[2].w;
    var sum = vec3<f32>(0.0);
    for (var i = 0u; i < neighbors.count; i++) {
        sum += neighbors.weights[i] * raa_spherical_incident(neighbors.entries[i], cell.desc.z, channels);
    }
    let blend = smoothstep(0.0, 0.15, neighbors.total);
    let intensity = bitcast<f32>(raa_spherical_data[0].w);
    return vec4<f32>(intensity * sum / neighbors.total, blend);
}

struct RaaSphericalInterpolation {
    shell0: u32,
    shell1: u32,
    fraction: vec3<f32>,
    base: vec2<i32>,
}

fn raa_spherical_interpolation(cell: RaaSphericalCoordinate) -> RaaSphericalInterpolation {
    let desc = cell.desc;
    let shell = cell.shell;
    let shell_bounds = cell.bounds;
    let radius = cell.radius;
    var shell0 = shell;
    if radius < shell_bounds.y && shell > 0u { shell0 -= 1u; }
    let shell1 = min(shell0 + 1u, desc.y - 1u);
    let r0 = bitcast<f32>(raa_spherical_data[4u + shell0].y);
    let r1 = bitcast<f32>(raa_spherical_data[4u + shell1].y);
    let depth = clamp((radius - r0) / max(r1 - r0, 1e-8), 0.0, 1.0);
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

// Reuse exactly the forward lookup's current, ready, normal-compatible cells.
// Consumers choose how to evaluate each entry's projected or directional data.
fn raa_spherical_neighbors(p: vec3<f32>, normal: vec3<f32>,
                           cell: RaaSphericalCoordinate) -> RaaSphericalNeighbors {
    if cell.desc.x == 0u { return RaaSphericalNeighbors(); }
    let interpolation = raa_spherical_interpolation(cell);
    let desc = cell.desc;
    let shell = cell.shell;
    let face = cell.face;
    let angular = cell.angular;
    var result: RaaSphericalNeighbors;
    for (var i = 0u; i < select(8u, 4u, interpolation.shell0 == interpolation.shell1); i++) {
        let corner = raa_spherical_corner(i);
        let key = raa_spherical_corner_key(cell, interpolation, corner);
        let entry = raa_spherical_find(key, desc.z);
        let support = raa_spherical_supported(entry, p, normal);
        if support <= 0.0 { continue; }
        var seen = false;
        for (var j = 0u; j < result.count; j++) { seen = seen || result.entries[j] == entry; }
        if seen { continue; }
        let w = select(1.0 - interpolation.fraction, interpolation.fraction, corner);
        let trilinear = max(w.x * w.y * w.z, 0.01);
        let weight = support * trilinear;
        result.entries[result.count] = entry;
        result.weights[result.count] = weight;
        result.trilinear[result.count] = trilinear;
        result.support[result.count] = support;
        result.total += weight;
        result.count += 1u;
    }

    // Near cell boundaries, a sparse surface may occupy only one of the
    // trilinear corners. Search the immediate ring for the remaining support.
    if result.count == 1u {
        let center = vec2<i32>(floor(angular));
        let lo = max(i32(shell) - 1, 0);
        let hi = min(i32(shell) + 1, i32(desc.y) - 1);
        var best = RAA_SPHERICAL_EMPTY;
        var best_support = 0.0;
        for (var ri = lo; ri <= hi; ri++) {
            for (var y = -1; y <= 1; y++) {
                for (var x = -1; x <= 1; x++) {
                    let key = raa_spherical_key(u32(ri), face, center + vec2<i32>(x, y), desc.x);
                    let entry = raa_spherical_find(key, desc.z);
                    if entry == result.entries[0] { continue; }
                    let support = raa_spherical_supported(entry, p, normal);
                    if support > best_support {
                        best = entry;
                        best_support = support;
                    }
                }
            }
        }
        if best_support > 0.0 {
            let weight = 0.15 * best_support;
            result.entries[result.count] = best;
            result.weights[result.count] = weight;
            result.trilinear[result.count] = 0.15;
            result.support[result.count] = best_support;
            result.total += weight;
            result.count += 1u;
        }
    }
    return result;
}
