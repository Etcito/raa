// The legacy, directional and learned paths share the same live lookup and
// hard compatibility masks. The fit only redistributes normalized influence.
struct GridDiffuseNeighbor {
    features: array<f32, 8>,
    incident: vec3<f32>,
    base_weight: f32,
}

fn grid_diffuse_neighbor(p: vec3<f32>, n: vec3<f32>, entry: u32, mask: u32,
                           trilinear: f32, support: f32) -> GridDiffuseNeighbor {
    let surface = bitcast<vec4<f32>>(raa_spherical_data[entry + 1u]);
    let normal = bitcast<vec3<f32>>(raa_spherical_data[entry + 2u].xyz);
    let delta = clamp((p - surface.xyz) / surface.w, vec3<f32>(-2.5), vec3<f32>(2.5));
    let base = raa_spherical_slot(entry, mask) + 3u;
    let channels = raa_spherical_data[2].w & 7u;
    var projected = vec3<f32>(0.0);
    var moment = vec3<f32>(0.0);
    for (var face = 0u; face < 6u; face++) {
        let rgb = raa_spherical_channel_sum(base + 3u * face, channels);
        let axis = raa_spherical_face_direction(face, vec2<f32>(0.0));
        projected += max(0.0, dot(n, axis)) * rgb;
        moment += axis * dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    }
    let magnitude = length(moment);
    let orientation = dot(n, moment) / max(magnitude, 1e-8);
    let incident = projected / max(dot(abs(n), vec3<f32>(1.0)), 1e-8);
    let intensity = bitcast<f32>(raa_spherical_data[0].w);
    return GridDiffuseNeighbor(
        array<f32, 8>(delta.x, delta.y, delta.z, dot(n, normal),
            orientation, trilinear, support, 1.0),
        incident * intensity, trilinear * support);
}

fn grid_diffuse(q: GridDiffuseQuery, cell: RaaSphericalCoordinate) -> vec4<f32> {
    let capture = (options[3].w & 32u) != 0u;
    if !GRID_DIFFUSE_DIRECTIONAL && !capture {
        return raa_spherical_diffuse_at(q.position, q.normal, cell);
    }
    let neighbors = raa_spherical_neighbors(q.position, q.normal, cell);
    var analytic: vec4<f32>;
    var fitted: vec4<f32>;
    if neighbors.count >= 2u {
        var raw_sum = vec3<f32>(0.0);
        var fitted_sum = vec3<f32>(0.0);
        var fitted_total = 0.0;
        for (var i = 0u; i < neighbors.count; i++) {
            let neighbor = grid_diffuse_neighbor(q.position, q.normal, neighbors.entries[i], cell.desc.z,
                neighbors.trilinear[i], neighbors.support[i]);
            if !GRID_DIFFUSE_LEARNED || capture {
                raw_sum += neighbor.base_weight * neighbor.incident;
            }
            if GRID_DIFFUSE_LEARNED {
                let weight = neighbor.base_weight * grid_diffuse_gate(neighbor.features);
                fitted_sum += weight * neighbor.incident;
                fitted_total += weight;
            }
        }
        let blend = smoothstep(0.0, 0.15, neighbors.total);
        analytic = vec4<f32>(raw_sum / neighbors.total, blend);
        if GRID_DIFFUSE_LEARNED {
            fitted = vec4<f32>(fitted_sum / fitted_total, blend);
        }
        if GRID_DIFFUSE_DIRECTIONAL { atomicAdd(&counters[DIRECTIONAL_DIFFUSE_HITS], 1u); }
        if GRID_DIFFUSE_LEARNED { atomicAdd(&counters[LEARNED_DIFFUSE_HITS], 1u); }
    }
    var value = select(analytic, fitted, GRID_DIFFUSE_LEARNED);
    if !GRID_DIFFUSE_DIRECTIONAL { value = raa_spherical_diffuse_at(q.position, q.normal, cell); }
    if capture { grid_diffuse_capture(q, cell, neighbors, value, analytic); }
    return value;
}
