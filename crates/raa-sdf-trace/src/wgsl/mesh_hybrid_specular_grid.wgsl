// The original directional capture ABI. Each ready slot owns six 8x8 faces.
struct MeshSphericalDirectionalTexel {
    position: vec4<f32>,
    sky_rgb: vec2<u32>,
    sun_rgb: vec2<u32>,
    other_rgb: vec2<u32>,
    normal: vec2<u32>,
}
@group(1) @binding(4) var<storage, read> mesh_spherical_faces: array<MeshSphericalDirectionalTexel>;

const ROUGH_GRID_TAPS: u32 = 4u;
const ROUGH_GRID_MIN_ROUGHNESS: f32 = 0.25;

fn rough_unpack_rgb(words: vec2<u32>) -> vec3<f32> {
    return vec3<f32>(unpack2x16float(words.x), unpack2x16float(words.y).x);
}

fn rough_valid_rgb(rgb: vec3<f32>) -> bool {
    return all(rgb >= vec3<f32>(0.0)) && all(rgb <= vec3<f32>(65504.0));
}

// Use world-space hit positions to avoid reusing texels on the wrong side of
// a parallax discontinuity. Each tap either contributes directional HDR or is
// rejected, never converted from the separately projected diffuse terms.
fn rough_cube_sample(slot: u32, p: vec3<f32>, direction: vec3<f32>, channels: u32) -> vec4<f32> {
    if slot >= arrayLength(&mesh_spherical_faces) / 384u { return vec4<f32>(0.0); }
    let face_uv = raa_spherical_face_uv(direction);
    let face = u32(face_uv.z);
    let pixel = (face_uv.xy + 1.0) * 4.0 - 0.5;
    let base = vec2<i32>(floor(pixel));
    let fraction = fract(pixel);
    var sum = vec3<f32>(0.0);
    var total = 0.0;
    for (var i = 0u; i < 4u; i++) {
        let step = vec2<bool>((i & 1u) != 0u, (i & 2u) != 0u);
        let weight = select(1.0 - fraction, fraction, step);
        let bilinear = weight.x * weight.y;
        if bilinear <= 0.0 { continue; }
        let coord = base + vec2<i32>(step);
        let uv = (vec2<f32>(coord) + 0.5) * 0.25 - 1.0;
        // Project taps beyond an edge onto the adjacent cubemap face.
        let adjacent = raa_spherical_face_uv(raa_spherical_face_direction(face, uv));
        let pixel_on_face = clamp(vec2<u32>((adjacent.xy + 1.0) * 4.0), vec2<u32>(0u), vec2<u32>(7u));
        let sample = mesh_spherical_faces[384u * slot + 64u * u32(adjacent.z)
            + 8u * pixel_on_face.y + pixel_on_face.x];
        if sample.position.w > 0.5 {
            // The recorded mirror surface has geometry but no complete outgoing
            // directional lighting in the capture.
            if unpack2x16float(sample.normal.y).y > 0.5 { continue; }
            let delta = sample.position.xyz - p;
            let along = dot(delta, direction);
            // Reject samples behind the secondary receiver or over about one
            // capture texel away from its projected direction.
            if along <= 2.0 * EPS || dot(delta, delta) > 1.09 * along * along { continue; }
            let normal = rough_unpack_rgb(sample.normal);
            if dot(normal, direction) >= -0.025 { continue; }
        }
        var rgb = vec3<f32>(0.0);
        if (channels & 1u) != 0u { rgb += rough_unpack_rgb(sample.sky_rgb); }
        if (channels & 2u) != 0u { rgb += rough_unpack_rgb(sample.sun_rgb); }
        if (channels & 4u) != 0u { rgb += rough_unpack_rgb(sample.other_rgb); }
        if !rough_valid_rgb(rgb) { continue; }
        sum += bilinear * rgb;
        total += bilinear;
    }
    if total < 0.5 { return vec4<f32>(0.0); }
    return vec4<f32>(sum / total, total);
}

fn rough_spherical_direction(neighbors: RaaSphericalNeighbors, p: vec3<f32>,
                             direction: vec3<f32>) -> vec4<f32> {
    if neighbors.count < 2u { return vec4<f32>(0.0); }
    let intensity = bitcast<f32>(raa_spherical_data[0].w);
    let channels = raa_spherical_data[2].w;
    var sum = vec3<f32>(0.0);
    var weight = 0.0;
    var supported = 0u;
    for (var i = 0u; i < neighbors.count; i++) {
        let slot = raa_spherical_data[neighbors.entries[i]].y;
        let incoming = rough_cube_sample(slot, p, direction, channels);
        if incoming.w == 0.0 { continue; }
        let w = incoming.w * neighbors.weights[i];
        sum += w * incoming.rgb;
        weight += w;
        supported += 1u;
    }
    if supported < 2u { return vec4<f32>(0.0); }
    return vec4<f32>(intensity * sum / weight, smoothstep(0.0, 0.15, weight));
}

fn rough_volume_direction(p: vec3<f32>, ng: vec3<f32>, direction: vec3<f32>) -> vec4<f32> {
    let voxel = raa_gi_finest_voxel();
    let hit = raa_gi_trace(p + ng * (1.5 * voxel), direction, voxel,
        min(frame.clip.y, 160.0), 32u, 0.5);
    if !hit.hit || !rough_valid_rgb(hit.radiance) { return vec4<f32>(0.0); }
    return vec4<f32>(hit.radiance * raa_gi_volume_params.intensity, 1.0);
}

// Integrate the actual secondary material's GGX lobe. Horizon-rejected
// taps have zero physical weight and are excluded from support accounting.
fn rough_grid_convolve(p: vec3<f32>, ng: vec3<f32>, n: vec3<f32>, v: vec3<f32>,
                       mat: Material, neighbors: RaaSphericalNeighbors, sphere: bool) -> HybridRoughGrid {
    let roughness = mat.base_roughness.w;
    let a = max(roughness * roughness, 0.01);
    let rotation = fract(dot(p, vec3<f32>(0.73, 1.13, 0.97)));
    let proper_surface = smoothstep(ROUGH_GRID_MIN_ROUGHNESS, 0.45, roughness);
    let nv = max(dot(n, v), 0.001);
    let surface = basis(n);
    var sum = vec3<f32>(0.0);
    var support = 0.0;
    var exposed = 0u;

    for (var i = 0u; i < ROUGH_GRID_TAPS; i++) {
        let u = (f32(i) + 0.5) / f32(ROUGH_GRID_TAPS);
        let phi = 2.0 * PI * (f32(reverseBits(i) >> 30u) / f32(ROUGH_GRID_TAPS) + rotation);
        let nh = sqrt((1.0 - u) / (1.0 + (a * a - 1.0) * u));
        let tangent = sqrt(max(0.0, 1.0 - nh * nh));
        let h = surface * vec3<f32>(tangent * cos(phi), tangent * sin(phi), nh);
        let direction = reflect(-v, h);
        let nl = dot(n, direction);
        let vh = dot(v, h);
        if nl <= 0.0 || vh <= 0.0 || dot(ng, direction) <= 0.0 { continue; }
        exposed += 1u;
        // Cancel GGX D with the half-vector PDF exactly as for the original stochastic ray.
        let brdf = fresnel(mat, vh) * ggx_visibility(a, nv, nl) * (4.0 * nl * vh / max(nh, 1e-5));
        var cached: vec4<f32>;
        if sphere { cached = rough_spherical_direction(neighbors, p, direction); }
        else { cached = rough_volume_direction(p, ng, direction); }
        sum += cached.w * brdf * cached.rgb;
        support += cached.w;
    }

    if exposed < 2u || support < 0.5 * f32(exposed) { return HybridRoughGrid(); }
    let coverage = support / f32(exposed);
    let result = sum * (f32(exposed) / (support * f32(ROUGH_GRID_TAPS)));
    if !rough_valid_rgb(result) { return HybridRoughGrid(); }
    return HybridRoughGrid(result, proper_surface * coverage, select(1u, 2u, sphere));
}

fn hybrid_grid_specular(p: vec3<f32>, ng: vec3<f32>, n: vec3<f32>, v: vec3<f32>,
                        mat: Material) -> HybridRoughGrid {
    if mat.base_roughness.w < ROUGH_GRID_MIN_ROUGHNESS || (mat.flags.x & 3u) != 0u {
        return HybridRoughGrid();
    }
    let coord = raa_spherical_coordinate(p);
    let neighbors = raa_spherical_neighbors(p, n, coord);
    if coord.desc.w == 0u && neighbors.count >= 2u {
        let sphere = rough_grid_convolve(p, ng, n, v, mat, neighbors, true);
        if sphere.trust > 0.0 { return sphere; }
    }
    // Explicit spherical routing supplies no volume role. Auto routing may
    // use the live volume when spherical directional coverage is incomplete.
    if mesh_volume_available(p) {
        return rough_grid_convolve(p, ng, n, v, mat, RaaSphericalNeighbors(), false);
    }
    return HybridRoughGrid();
}
