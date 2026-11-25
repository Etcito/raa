// Offline light transport for the same secondary query, before albedo. Uses
// distinct random banks for the cosine paths and for panel/sky samples.
fn grid_diffuse_sky(dir: vec3<f32>) -> vec3<f32> {
    if (options[3].w & 64u) == 0u { return frame.ambient.rgb; }
    return bitcast<f32>(options[2].x) * textureSampleLevel(sky, raa_linear_clamp, dir, 0.0).rgb;
}

fn grid_diffuse_teacher(q: GridDiffuseQuery, seed_value: u32, channels: u32) -> vec3<f32> {
    var seed = seed_value;
    var rgb = vec3<f32>(0.0);
    let surface = basis(q.normal);
    let origin = q.position + q.geometric * (2.0 * EPS);
    let limit = max(frame.clip.y, bitcast<f32>(raa_spherical_data[2].y));
    for (var sample = 0u; sample < 32u; sample++) {
        let u = (f32(sample) + random(&seed)) / 32.0;
        let angle = 2.0 * PI * random(&seed);
        let dir = surface * vec3<f32>(sqrt(u) * cos(angle), sqrt(u) * sin(angle), sqrt(1.0 - u));
        if dot(dir, q.geometric) <= 0.0 { continue; }
        let hit = closest(Ray(origin, dir), limit, false);
        if hit.object == MISS {
            if (channels & 1u) != 0u { rgb += grid_diffuse_sky(dir); }
            continue;
        }
        let tri = triangles[hit.object];
        let material = hit_material(tri, hit.bary);
        let p = origin + dir * hit.t;
        let geom = normalize(cross(tri.b.position - tri.a.position, tri.c.position - tri.a.position));
        let ng = select(geom, -geom, dot(geom, dir) > 0.0);
        let blended = tri.a.normal * (1.0 - hit.bary.x - hit.bary.y)
            + tri.b.normal * hit.bary.x + tri.c.normal * hit.bary.y;
        var n = ng;
        if dot(blended, blended) > 1e-8 {
            let candidate = normalize(blended);
            n = select(-candidate, candidate, dot(candidate, ng) >= 0.0);
        }
        if (channels & 4u) != 0u {
            rgb += material.emission_metallic.rgb;
            if (material.flags.x & 2u) != 0u { rgb += material.base_roughness.rgb; }
        }
        if (material.flags.x & 3u) != 0u { continue; }
        if (channels & 2u) != 0u { rgb += mesh_sun_at_hit(p, ng, n, -dir, material, true); }
        if (channels & 4u) != 0u {
            rgb += mesh_lights_at_hit(p, ng, n, -dir, material, &seed, true, false, vec3<f32>(0.0));
        }
        if (channels & 1u) != 0u {
            let v = random(&seed);
            let phi = 2.0 * PI * random(&seed);
            let bounce = basis(n) * vec3<f32>(sqrt(v) * cos(phi), sqrt(v) * sin(phi), sqrt(1.0 - v));
            if dot(ng, bounce) > 0.0
                && closest(Ray(p + ng * (2.0 * EPS), bounce), limit, true).object == MISS {
                rgb += diffuse(material) * grid_diffuse_sky(bounce);
            }
        }
    }
    return rgb * (bitcast<f32>(raa_spherical_data[0].w) / 32.0);
}

fn grid_diffuse_write_rgb(offset: u32, value: vec3<f32>) {
    let words = bitcast<vec3<u32>>(value);
    for (var i = 0u; i < 3u; i++) { atomicStore(&counters[offset + i], words[i]); }
}

// Four bits per interpolation corner. The first rejection explains why
// that corner cannot contribute, even if another test would also reject it.
fn grid_diffuse_corner_status(entry: u32, q: GridDiffuseQuery) -> u32 {
    if entry == RAA_SPHERICAL_EMPTY { return 1u; }
    if raa_spherical_data[entry + 2u].w == 0u { return 2u; }
    let gates = raa_spherical_support_gates(entry, q.position, q.normal);
    if gates.x == 0.0 { return 3u; }
    if gates.y == 0.0 { return 4u; }
    if gates.z == 0.0 { return 5u; }
    return 6u;
}

fn grid_diffuse_corner_diagnostics(q: GridDiffuseQuery, cell: RaaSphericalCoordinate) -> u32 {
    if cell.desc.x == 0u { return 8u; }
    let interpolation = raa_spherical_interpolation(cell);
    let count = select(8u, 4u, interpolation.shell0 == interpolation.shell1);
    var keys: array<u32, 8>;
    var packed = 0u;
    for (var i = 0u; i < count; i++) {
        let key = raa_spherical_corner_key(cell, interpolation, raa_spherical_corner(i));
        var duplicate = false;
        for (var j = 0u; j < i; j++) { duplicate = duplicate || keys[j] == key; }
        keys[i] = key;
        var status = 7u;
        if !duplicate {
            status = grid_diffuse_corner_status(raa_spherical_find(key, cell.desc.z), q);
        }
        packed |= status << (4u * i);
    }
    return packed;
}

const GRID_DIFFUSE_CAPTURE_MAX_ROWS: u32 = 128u;
const GRID_DIFFUSE_CAPTURE_ROW_WORDS: u32 = 128u;

fn grid_diffuse_capture(q: GridDiffuseQuery, cell: RaaSphericalCoordinate,
                        neighbors: RaaSphericalNeighbors, active_value: vec4<f32>,
                        analytic: vec4<f32>) {
    if (options[3].w & 32u) == 0u { return; }
    let p = bitcast<vec3<u32>>(q.position);
    let hash = raa_spherical_hash(p.x ^ raa_spherical_hash(p.y ^ raa_spherical_hash(p.z ^ q.triangle)));
    if raa_spherical_hash(q.pixel) % options[1].w != 0u { return; }
    let row = atomicAdd(&counters[GRID_DIFFUSE_ROW_COUNT], 1u);
    if row >= GRID_DIFFUSE_CAPTURE_MAX_ROWS { return; }
    let base = REFLECTED_RECEIVER_BASE + row * GRID_DIFFUSE_CAPTURE_ROW_WORDS;
    let diagnostics = REFLECTED_RECEIVER_BASE + GRID_DIFFUSE_CAPTURE_MAX_ROWS * GRID_DIFFUSE_CAPTURE_ROW_WORDS;
    atomicStore(&counters[diagnostics + row], grid_diffuse_corner_diagnostics(q, cell));
    let channels = raa_spherical_data[2].w;
    let seed = raa_spherical_hash(hash ^ options[2].w);
    let legacy = raa_spherical_diffuse_at(q.position, q.normal, cell);
    grid_diffuse_write_rgb(base, q.position);
    atomicStore(&counters[base + 3u], q.triangle);
    grid_diffuse_write_rgb(base + 4u, q.normal);
    atomicStore(&counters[base + 7u], q.material);
    grid_diffuse_write_rgb(base + 8u, q.geometric);
    atomicStore(&counters[base + 11u], neighbors.count);
    grid_diffuse_write_rgb(base + 12u, grid_diffuse_teacher(q, seed ^ 0x9e3779b9u, channels));
    atomicStore(&counters[base + 15u], bitcast<u32>(analytic.w));
    grid_diffuse_write_rgb(base + 16u, grid_diffuse_teacher(q, seed ^ 0x85ebca6bu, channels));
    atomicStore(&counters[base + 19u], bitcast<u32>(legacy.w));
    grid_diffuse_write_rgb(base + 20u, legacy.rgb);
    atomicStore(&counters[base + 23u], bitcast<u32>(neighbors.total));
    grid_diffuse_write_rgb(base + 24u, active_value.rgb);
    atomicStore(&counters[base + 27u], channels);
    grid_diffuse_write_rgb(base + 28u, analytic.rgb);
    atomicStore(&counters[base + 31u], q.pixel);
    for (var i = 0u; i < neighbors.count; i++) {
        let neighbor = grid_diffuse_neighbor(q.position, q.normal, neighbors.entries[i], cell.desc.z,
            neighbors.trilinear[i], neighbors.support[i]);
        let offset = base + 32u + 12u * i;
        for (var j = 0u; j < 8u; j++) {
            atomicStore(&counters[offset + j], bitcast<u32>(neighbor.features[j]));
        }
        grid_diffuse_write_rgb(offset + 8u, neighbor.incident);
        atomicStore(&counters[offset + 11u], bitcast<u32>(neighbor.base_weight));
    }
}
