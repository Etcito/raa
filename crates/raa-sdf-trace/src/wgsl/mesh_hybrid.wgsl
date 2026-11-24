// Separate entry point and resources. The independent path reference never binds a model.
struct HybridOptions { mode: u32, seed: u32, capture: u32, samples: u32 }
@group(1) @binding(0) var<uniform> hybrid: HybridOptions;
@group(1) @binding(1) var<storage, read> path_head: array<f32, 131>;
@group(1) @binding(2) var<storage, read_write> path_capture: array<u32>;
@group(1) @binding(3) var diffuse_ao: texture_2d<f32>;

const PATH_MODEL_PIXELS: u32 = 12u;
const PATH_REFRESH_PIXELS: u32 = 13u;
const DIFFUSE_MISSING_PIXELS: u32 = 14u;
const DIFFUSE_RAYS: u32 = 15u;
const SPECULAR_RAYS: u32 = 16u;
const PATH_FOREGROUND_PIXELS: u32 = 17u;
const PATH_MODEL_REJECTED: u32 = 18u;
const PRIMARY_BVH_PIXELS: u32 = 19u;

fn hybrid_seed(pixel: vec2<u32>, stream: u32) -> u32 {
    var seed = (hybrid.seed ^ stream ^ (pixel.x * 1597334677u) ^ (pixel.y * 3812015801u));
    _ = random(&seed);
    return seed;
}

struct HybridSpecularPath {
    base: vec3<f32>,
    first_position: vec3<f32>,
    first_triangle: u32,
    grid: HybridRoughGrid,
};

fn hybrid_specular(p: vec3<f32>, ng: vec3<f32>, n: vec3<f32>, v: vec3<f32>,
                   mat: Material, steps: u32, use_grid: bool, seed: ptr<function, u32>) -> HybridSpecularPath {
    let a = max(mat.base_roughness.w * mat.base_roughness.w, 0.01);
    let angle = random(seed) * (2.0 * PI);
    let xi = min(random(seed), 0.99999);
    let nh = sqrt((1.0 - xi) / (1.0 + (a * a - 1.0) * xi));
    let sh = sqrt(max(0.0, 1.0 - nh * nh));
    let h = basis(n) * vec3<f32>(sh * cos(angle), sh * sin(angle), nh);
    let l = reflect(-v, h);
    let nl = dot(n, l);
    let vh = dot(v, h);
    if nl <= 0.0 || vh <= 0.0 || dot(ng, l) <= 0.0 { return HybridSpecularPath(vec3<f32>(0.0), vec3<f32>(0.0), MISS, HybridRoughGrid()); }
    let nv = max(dot(n, v), 0.001);
    // Cancel GGX D against the sampled half-vector PDF.
    let weight = fresnel(mat, vh) * ggx_visibility(a, nv, nl) * (4.0 * nl * vh / max(nh, 1e-5));
    atomicAdd(&counters[PATH_RAYS], 1u);
    atomicAdd(&counters[SPECULAR_RAYS], 1u);
    let ray = Ray(p + ng * (2.0 * EPS), normalize(l));
    let hit = closest(ray, frame.clip.y, false);
    if hybrid.mode == 3u && hybrid.capture != 1u { record_mesh_first_hit(hit, ray); }
    // The fitted input uses the material-weighted first bounce captured by its
    // trainer. The grid reads the same exact hit after that estimate.
    let base = weight * trace_path_steps_from_hit(ray, hit, options[0].y, true, steps, seed);
    if hit.object == MISS {
        return HybridSpecularPath(base, vec3<f32>(0.0), MISS, HybridRoughGrid());
    }
    let p1 = ray.origin + ray.direction * hit.t;
    var grid: HybridRoughGrid;
    if use_grid {
        let tri = triangles[hit.object];
        let mat1 = hit_material(tri, hit.bary);
        let face = normalize(cross(tri.b.position - tri.a.position, tri.c.position - tri.a.position));
        let ng1 = select(-face, face, dot(face, ray.direction) < 0.0);
        let interp = normalize(tri.a.normal * (1.0 - hit.bary.x - hit.bary.y)
            + tri.b.normal * hit.bary.x + tri.c.normal * hit.bary.y);
        let facing = select(-interp, interp, dot(interp, ng1) >= 0.0);
        let n1 = select(ng1, facing, dot(facing, -ray.direction) > 0.001);
        grid = hybrid_grid_specular(p1, ng1, n1, -ray.direction, mat1);
        grid.rgb *= weight;
    }
    return HybridSpecularPath(base, p1, hit.object, grid);
}

fn hybrid_diffuse(p: vec3<f32>, ng: vec3<f32>, n: vec3<f32>, mat: Material,
                  seed: ptr<function, u32>) -> vec3<f32> {
    let weight = diffuse(mat);
    if !any(weight > vec3<f32>(0.0)) { return vec3<f32>(0.0); }
    let angle = random(seed) * (2.0 * PI);
    let u = random(seed);
    let l = basis(n) * vec3<f32>(sqrt(u) * cos(angle), sqrt(u) * sin(angle), sqrt(1.0 - u));
    if dot(ng, l) <= 0.0 { return vec3<f32>(0.0); }
    atomicAdd(&counters[PATH_RAYS], 1u);
    atomicAdd(&counters[DIFFUSE_RAYS], 1u);
    let ray = Ray(p + ng * (2.0 * EPS), normalize(l));
    if hybrid.mode == 3u {
        return weight * trace_live_indirect_path(ray, options[0].y, 1u, seed, 0u);
    }
    return weight * trace_path_steps(ray, options[0].y, true, 1u, seed);
}

fn predict_specular(raw: vec3<f32>, normal: vec3<f32>, z: f32, roughness: f32) -> vec3<f32> {
    // Normalized HWC lanes from the CPU indirect-specular fit. The cheap
    // source is a valid freshly traced one-bounce estimate with no confidence value.
    let input = array<f32, 12>(
        raw.x, raw.y, raw.z, normal.x, normal.y, normal.z, z * 0.125, roughness, 1.0, 0.0, 0.0, 1.0,
    );
    var hidden: array<f32, 8>;
    for (var h = 0u; h < 8u; h++) {
        var sum = path_head[96u + h];
        for (var i = 0u; i < 12u; i++) {
            sum = fma(input[i], path_head[h * 12u + i], sum);
        }
        hidden[h] = max(sum, 0.0);
    }
    var rgb = vec3<f32>(0.0);
    for (var channel = 0u; channel < 3u; channel++) {
        var sum = path_head[128u + channel];
        for (var h = 0u; h < 8u; h++) {
            sum = fma(hidden[h], path_head[104u + channel * 8u + h], sum);
        }
        rgb[channel] = max(sum, 0.0) + log(1.0 + exp(-abs(sum)));
    }
    return rgb;
}

fn write_hybrid_capture(index: u32, rgb: vec3<f32>, n: vec3<f32>, z: f32, roughness: f32) {
    if hybrid.capture != 1u { return; }
    let base = index * 8u;
    path_capture[base] = bitcast<u32>(rgb.x);
    path_capture[base + 1u] = bitcast<u32>(rgb.y);
    path_capture[base + 2u] = bitcast<u32>(rgb.z);
    path_capture[base + 3u] = bitcast<u32>(n.x);
    path_capture[base + 4u] = bitcast<u32>(n.y);
    path_capture[base + 5u] = bitcast<u32>(n.z);
    path_capture[base + 6u] = bitcast<u32>(z);
    path_capture[base + 7u] = bitcast<u32>(roughness);
}

// Evidence mode uses the same path capture buffer as training. The renderer
// continues to produce the real fitted image with the explicitly pinned seed.
fn write_rough_evidence(index: u32, sample: HybridSpecularPath) {
    if hybrid.capture != 2u { return; }
    let base = index * 12u;
    path_capture[base] = bitcast<u32>(sample.first_position.x);
    path_capture[base + 1u] = bitcast<u32>(sample.first_position.y);
    path_capture[base + 2u] = bitcast<u32>(sample.first_position.z);
    path_capture[base + 3u] = sample.first_triangle;
    path_capture[base + 4u] = bitcast<u32>(sample.base.x);
    path_capture[base + 5u] = bitcast<u32>(sample.base.y);
    path_capture[base + 6u] = bitcast<u32>(sample.base.z);
    path_capture[base + 7u] = bitcast<u32>(sample.grid.trust);
    path_capture[base + 8u] = bitcast<u32>(sample.grid.rgb.x * sample.grid.trust);
    path_capture[base + 9u] = bitcast<u32>(sample.grid.rgb.y * sample.grid.trust);
    path_capture[base + 10u] = bitcast<u32>(sample.grid.rgb.z * sample.grid.trust);
    path_capture[base + 11u] = sample.grid.source;
}

// Reuse raster GTAO only when the raster and ray hits describe the same surface.
fn hybrid_diffuse_ao(pixel: vec2<i32>, dims: vec2<u32>, z: f32, view_z: f32,
                     view_normal: vec3<f32>) -> f32 {
    if (frame.flags[INDIRECT_DIFFUSE_AO_WORD] & INDIRECT_DIFFUSE_AO_MASK) == 0u || z >= 1.0 {
        return 1.0;
    }
    let raster_position = world_at(uv_at(pixel, dims), z);
    let raster_z = (frame.view * vec4<f32>(raster_position, 1.0)).z;
    let raster_normal = textureLoad(normals, pixel, 0).xyz;
    if abs(raster_z - view_z) > max(0.01, abs(view_z) * 0.01)
        || dot(raster_normal, view_normal) < 0.8 {
        return 1.0;
    }
    let last = vec2<i32>(textureDimensions(diffuse_ao)) - vec2<i32>(1);
    return textureLoad(diffuse_ao, min(pixel, last), 0).r;
}

@compute @workgroup_size(8, 8)
fn trace_hybrid_path(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(radiance);
    if any(id.xy >= size) { return; }
    let sample_pixel = vec2<i32>(id.xy);
    let index = id.y * size.x + id.x;
    write_hybrid_capture(index, vec3<f32>(0.0), vec3<f32>(0.0), 0.0, 0.0);
    write_rough_evidence(index,
        HybridSpecularPath(vec3<f32>(0.0), vec3<f32>(0.0), MISS, HybridRoughGrid()));
    let uv = uv_at(sample_pixel, size);
    let dims = textureDimensions(depth);
    let pixel = pixel_at(uv, dims);
    let z = textureLoad(depth, pixel, 0);
    let origin = world_at(uv, 0.0);
    let far = world_at(uv, 1.0);
    let ray = Ray(origin, normalize(far - origin));
    var spec_seed = hybrid_seed(id.xy, 0u);
    let diffuse_output = options[2].z;
    let lighting_output = options[3].y;
    // Mirror screen evidence uses the same confidence/disocclusion checks as
    // ordinary hybrid reflections, before any BVH query or network work.
    if z < 1.0 && textureLoad(motion, pixel, 0).w > 0.5 {
        if hybrid.capture == 1u || diffuse_output != 0u || lighting_output == 1u || lighting_output >= 16u {
            textureStore(radiance, sample_pixel, vec4<f32>(0.0));
            return;
        }
        atomicAdd(&counters[ELIGIBLE], 1u);
        let screen = textureLoad(confidence, pixel, 0);
        let status = screen_status(screen, pixel, dims);
        atomicAdd(&counters[status], 1u);
        if status == ACCEPTED {
            let rgb = textureLoad(screen_radiance, pixel_at(screen.xy, dims), 0).rgb;
            textureStore(radiance, sample_pixel, vec4<f32>(0.92 * rgb, 1.0));
            return;
        }
        atomicAdd(&counters[TRACED], 1u);
        atomicAdd(&counters[PRIMARY_BVH_PIXELS], 1u);
        var value: vec3<f32>;
        if hybrid.mode == 3u {
            value = trace_live_indirect_path(ray, options[0].y, 2u, &spec_seed, 1u);
        } else {
            value = trace_path_steps(ray, options[0].y, true, 2u, &spec_seed);
        }
        textureStore(radiance, sample_pixel, vec4<f32>(value, 1.0));
        return;
    }
    atomicAdd(&counters[PRIMARY_BVH_PIXELS], 1u);
    let hit = closest(ray, frame.clip.y, false);
    if hit.object == MISS {
        if diffuse_output != 0u || lighting_output == 2u || lighting_output >= 16u {
            textureStore(radiance, sample_pixel, vec4<f32>(0.0));
            return;
        }
        textureStore(radiance, sample_pixel, vec4<f32>(textureSampleLevel(sky, raa_linear_clamp, ray.direction, 0).rgb, 1.0));
        return;
    }
    let tri = triangles[hit.object];
    let mat = hit_material(tri, hit.bary);
    if (mat.flags.x & 3u) != 0u {
        var depth = 0u;
        if (mat.flags.x & 1u) != 0u { depth = 1u; }
        if diffuse_output != 0u || lighting_output >= 16u || !select_lighting_depth(lighting_output, depth) {
            textureStore(radiance, sample_pixel, vec4<f32>(0.0));
            return;
        }
        var value: vec3<f32>;
        if hybrid.mode == 3u && (mat.flags.x & 1u) != 0u {
            value = trace_live_indirect_path(ray, options[0].y, 2u, &spec_seed, 1u);
        } else {
            value = trace_path_steps(ray, options[0].y, true, 2u, &spec_seed);
        }
        textureStore(radiance, sample_pixel, vec4<f32>(value, 1.0));
        return;
    }
    let p = ray.origin + ray.direction * hit.t;
    let ng0 = normalize(cross(tri.b.position - tri.a.position, tri.c.position - tri.a.position));
    let ng = select(-ng0, ng0, dot(ng0, ray.direction) < 0.0);
    let ns = normalize(tri.a.normal * (1.0 - hit.bary.x - hit.bary.y)
        + tri.b.normal * hit.bary.x + tri.c.normal * hit.bary.y);
    let oriented = select(-ns, ns, dot(ns, ng) >= 0.0);
    let v = -ray.direction;
    let n = select(ng, oriented, dot(oriented, v) > 0.001);
    let pos = frame.view * vec4<f32>(p, 1.0);
    let view_n = clamp((frame.view * vec4<f32>(n, 0.0)).xyz, vec3<f32>(-1.0), vec3<f32>(1.0));
    let roughness = clamp(mat.base_roughness.w, 0.0, 1.0);
    let refresh = hybrid.mode == 2u || (hybrid.mode == 3u && ((id.x + 2u * id.y + hybrid.seed) & 3u) == 0u);
    let steps = select(0u, 1u, refresh);
    let try_grid = hybrid.mode == 3u && !refresh && hybrid.capture != 1u
        && roughness >= ROUGH_GRID_MIN_ROUGHNESS;
    var spec = vec3<f32>(0.0);
    var grid_tail = vec3<f32>(0.0);
    var grid_trust = 0.0;
    for (var i = 0u; i < hybrid.samples; i++) {
        let sample = hybrid_specular(p, ng, n, v, mat, steps, try_grid, &spec_seed);
        spec += sample.base;
        if sample.grid.trust > 0.0 {
            grid_tail += sample.grid.rgb * sample.grid.trust;
            grid_trust += sample.grid.trust;
            let counter = select(ROUGH_VOLUME_ACCEPTED, ROUGH_SPHERICAL_ACCEPTED, sample.grid.source == 2u);
            atomicAdd(&counters[counter], 1u);
        }
        if i == 0u { write_rough_evidence(index, sample); }
    }
    spec /= f32(hybrid.samples);
    grid_tail /= f32(hybrid.samples);
    grid_trust /= f32(hybrid.samples);
    if hybrid.capture == 1u {
        write_hybrid_capture(index, spec, view_n, abs(pos.z), roughness);
        textureStore(radiance, sample_pixel, vec4<f32>(0.0));
        return;
    }
    atomicAdd(&counters[PATH_FOREGROUND_PIXELS], 1u);
    if refresh { atomicAdd(&counters[PATH_REFRESH_PIXELS], 1u); }
    if hybrid.mode == 3u && !refresh {
        let predicted = predict_specular(spec, view_n, abs(pos.z), roughness);
        if all(predicted >= vec3<f32>(0.0)) && all(predicted <= vec3<f32>(65504.0)) {
            // Interpolate the two estimates of the remainder. The secondary
            // direct term is evaluated once and is already part of the raw source.
            spec = select(predicted, mix(predicted, spec, grid_trust) + grid_tail, grid_trust > 0.0);
            atomicAdd(&counters[PATH_MODEL_PIXELS], 1u);
        } else {
            spec += grid_tail;
            atomicAdd(&counters[PATH_MODEL_REJECTED], 1u);
        }
    }
    var diffuse_seed = hybrid_seed(id.xy, 0x93ed12a3u);
    // Incident diffuse starts with its existing real ray. A routed cache may
    // estimate the tail only after that ray reaches a secondary surface.
    atomicAdd(&counters[DIFFUSE_MISSING_PIXELS], 1u);
    let indirect_diffuse = hybrid_diffuse(p, ng, n, mat, &diffuse_seed)
        * hybrid_diffuse_ao(pixel, dims, z, pos.z, view_n);
    var direct_seed = hybrid_seed(id.xy, 0x217c6407u);
    let direct = sun_and_lights(p, ng, n, v, mat, &direct_seed, diffuse_output != 0u);
    if diffuse_output != 0u {
        let rgb = direct + select(vec3<f32>(0.0), indirect_diffuse, diffuse_output == 2u);
        textureStore(radiance, sample_pixel, vec4<f32>(rgb, 1.0));
        return;
    }
    var value = mat.emission_metallic.rgb + direct + indirect_diffuse + spec;
    if lighting_output == 1u { value = mat.emission_metallic.rgb + direct; }
    if lighting_output == 2u { value = indirect_diffuse + spec; }
    if lighting_output == 16u { value = indirect_diffuse; }
    if lighting_output == 17u { value = spec; }
    if lighting_output == 18u { value = grid_tail; }
    textureStore(radiance, sample_pixel, vec4<f32>(value, 1.0));
}
