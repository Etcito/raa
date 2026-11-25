// Neural radiance cache on the shared world-space mesh BVH: training rays (self-trained
// one-bounce suffixes) and the per-pixel use of the cache. The cache stores R(x, n) = indirect diffuse
// irradiance / pi. Composed after the mesh trace module and `nrc_mlp.wgsl`.

struct NrcParams {
    // training samples, seed, bounces of the render pass, local-light count sampled exhaustively
    counts: vec4<u32>,
    // sky scale, ray limit (m), target clamp, unused
    tuning: vec4<f32>,
    // low-resolution width and height, divisor, unused
    size: vec4<u32>,
}
@group(0) @binding(30) var<uniform> nrc_params: NrcParams;
@group(0) @binding(31) var<storage, read> nrc_weights: array<vec4<f32>>;
// Three rows per sample: position.xyz + valid, normal.xyz, target.rgb.
@group(0) @binding(32) var<storage, read_write> nrc_samples: array<vec4<f32>>;
@group(0) @binding(33) var nrc_noisy: texture_storage_2d<rgba16float, write>;
// Two rows per pair of training threads: hit position + valid, hit normal.
@group(0) @binding(34) var<storage, read> nrc_origins_in: array<vec4<f32>>;
@group(0) @binding(35) var<storage, read_write> nrc_origins_out: array<vec4<f32>>;

//!include nrc_mlp

struct NrcSurface {
    hit: bool,
    unlit: bool,
    t: f32,
    p: vec3<f32>,
    ng: vec3<f32>,
    n: vec3<f32>,
    mat: Material,
}

fn nrc_sky(direction: vec3<f32>) -> vec3<f32> {
    if options[0].z == 0u { return frame.ambient.rgb; }
    return nrc_params.tuning.x * textureSampleLevel(sky, raa_linear_clamp, direction, 0.0).rgb;
}

// Position normalized in the bounds of the BVH root.
fn nrc_position(p: vec3<f32>) -> vec3<f32> {
    let lo = nodes[0].lo;
    let extent = max(nodes[0].hi - lo, vec3<f32>(1e-3));
    return clamp((p - lo) / extent, vec3<f32>(0.0), vec3<f32>(1.0));
}

// Closest surface along a ray; geometry is treated as two-sided.
fn nrc_surface(ray: Ray, limit: f32) -> NrcSurface {
    var s: NrcSurface;
    let hit = closest(ray, limit, false);
    s.hit = hit.object != MISS;
    if !s.hit { return s; }
    let tri = mesh_triangle(hit.object);
    let geom = normalize(cross(tri.b.position - tri.a.position, tri.c.position - tri.a.position));
    s.t = hit.t;
    s.p = ray.origin + ray.direction * hit.t;
    s.ng = select(geom, -geom, dot(geom, ray.direction) > 0.0);
    let blended = tri.a.normal * (1.0 - hit.bary.x - hit.bary.y)
        + tri.b.normal * hit.bary.x + tri.c.normal * hit.bary.y;
    s.n = s.ng;
    if dot(blended, blended) > 1e-8 {
        let candidate = normalize(blended);
        s.n = select(-candidate, candidate, dot(candidate, s.ng) >= 0.0);
    }
    s.mat = hit_material(tri, hit.bary);
    s.unlit = (s.mat.flags.x & 2u) != 0u;
    return s;
}

// Outgoing diffuse radiance of a surface from emission, the sun and the local lights.
fn nrc_direct(s: NrcSurface, v: vec3<f32>, seed: ptr<function, u32>) -> vec3<f32> {
    var rgb = s.mat.emission_metallic.rgb;
    if s.unlit { return rgb + s.mat.base_roughness.rgb; }
    rgb += mesh_sun_at_hit(s.p, s.ng, s.n, v, s.mat, true);
    let origin = s.p + s.ng * (2.0 * EPS);
    let count = min(256u, frame.light_info.x);
    if count <= nrc_params.counts.w {
        for (var i = 0u; i < count; i++) {
            var light_seed = *seed;
            rgb += mesh_local_at_hit(i, origin, s.p, s.ng, s.n, v, s.mat, &light_seed, true, false);
        }
    } else {
        // One uniformly chosen light, weighted by the light count.
        let i = min(u32(random(seed) * f32(count)), count - 1u);
        var light_seed = *seed;
        rgb += f32(count)
            * mesh_local_at_hit(i, origin, s.p, s.ng, s.n, v, s.mat, &light_seed, true, false);
    }
    return rgb;
}

// Cosine-distributed direction around `n`, kept on the outer side of the geometric normal.
fn nrc_bounce(n: vec3<f32>, ng: vec3<f32>, seed: ptr<function, u32>) -> vec3<f32> {
    let u = vec2<f32>(random(seed), random(seed));
    let r = sqrt(u.x);
    let phi = 2.0 * PI * u.y;
    let local = vec3<f32>(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u.x)));
    var d = normalize(basis(n) * local);
    let below = dot(d, ng);
    if below <= 0.0 { d = normalize(d - 2.0 * below * ng + 1e-3 * ng); }
    return d;
}

fn nrc_cached(s: NrcSurface) -> vec3<f32> {
    return max(nrc_query(nrc_position(s.p), s.n), vec3<f32>(0.0));
}

fn nrc_store(index: u32, p: vec3<f32>, n: vec3<f32>, wanted: vec3<f32>, valid: bool) {
    let clamped = clamp(wanted, vec3<f32>(0.0), vec3<f32>(nrc_params.tuning.z));
    nrc_samples[index * 3u] = vec4<f32>(nrc_position(p), select(0.0, 1.0, valid && all(wanted == wanted)));
    nrc_samples[index * 3u + 1u] = vec4<f32>(n, 0.0);
    nrc_samples[index * 3u + 2u] = vec4<f32>(clamped, 0.0);
}

// One training sample per invocation: an origin x (a random visible pixel for even threads, a
// surface reached by last frame's rays for odd threads, so the cache also learns what the
// tails query off screen), one cosine bounce to y, direct light at y and the cache at y as the
// tail (self-training). The sample learns R(x) = L_o(y). Each pair of threads hands one hit to
// the next frame. A thread traces one closest-hit ray and its shadow rays: the pass costs about
// one ray of latency.
@compute @workgroup_size(64)
fn nrc_paths(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= nrc_params.counts.x { return; }
    var seed = (id.x * 747796405u + nrc_params.counts.y * 2891336453u) ^ 0x85ebca6bu;
    let limit = nrc_params.tuning.y;
    let slot = id.x >> 1u;
    var p = vec3<f32>(0.0);
    var n = vec3<f32>(0.0, 1.0, 0.0);
    var bias = 2.0 * EPS;
    var valid = false;
    if (id.x & 1u) == 1u {
        let stored = nrc_origins_in[slot * 2u];
        p = stored.xyz;
        n = nrc_origins_in[slot * 2u + 1u].xyz;
        valid = stored.w > 0.5;
    }
    if !valid {
        let dims = textureDimensions(depth);
        let pixel = min(vec2<i32>(vec2<f32>(random(&seed), random(&seed)) * vec2<f32>(dims)),
            vec2<i32>(dims) - 1);
        let z = textureLoad(depth, pixel, 0);
        let normal_view = textureLoad(normals, pixel, 0).xyz;
        if z < 1.0 && dot(normal_view, normal_view) > 0.0 {
            p = world_at(uv_at(pixel, dims), z);
            n = normalize((frame.inv_view * vec4<f32>(normal_view, 0.0)).xyz);
            bias = max(4.0 * EPS, 2e-3 * distance(p, frame.camera_pos.xyz));
            valid = true;
        }
    }
    // Which thread of the pair hands its hit over: the off-screen walk continues half the time.
    var pair_seed = (slot * 2654435761u) ^ (nrc_params.counts.y * 2246822519u);
    let keeper = u32(random(&pair_seed) < 0.5);
    let keeps = (id.x & 1u) == keeper;
    if !valid {
        nrc_store(id.x, p, n, vec3<f32>(0.0), false);
        if keeps { nrc_origins_out[slot * 2u] = vec4<f32>(0.0); }
        return;
    }
    let d = nrc_bounce(n, n, &seed);
    let y = nrc_surface(Ray(p + n * bias, d), limit);
    var wanted = nrc_sky(d);
    if y.hit {
        wanted = nrc_direct(y, -d, &seed);
        if !y.unlit { wanted += diffuse(y.mat) * nrc_cached(y); }
    }
    nrc_store(id.x, p, n, wanted, true);
    if keeps {
        let next = y.hit && !y.unlit;
        nrc_origins_out[slot * 2u] = vec4<f32>(y.p, select(0.0, 1.0, next));
        nrc_origins_out[slot * 2u + 1u] = vec4<f32>(y.n, 0.0);
    }
}

// The cache at reduced resolution: rgb = R at the pixel, alpha = bounce ray length (0 without a
// bounce), negative where the pixel has no surface.
@compute @workgroup_size(8, 8)
fn nrc_render(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= nrc_params.size.xy) { return; }
    let dims = textureDimensions(depth);
    let divisor = nrc_params.size.z;
    let pixel = min(vec2<i32>(id.xy * divisor + divisor / 2u), vec2<i32>(dims) - 1);
    let z = textureLoad(depth, pixel, 0);
    let normal_view = textureLoad(normals, pixel, 0).xyz;
    if z >= 1.0 || dot(normal_view, normal_view) <= 0.0 {
        textureStore(nrc_noisy, vec2<i32>(id.xy), vec4<f32>(0.0, 0.0, 0.0, -1.0));
        return;
    }
    let p = world_at(uv_at(pixel, dims), z);
    let n = normalize((frame.inv_view * vec4<f32>(normal_view, 0.0)).xyz);
    if nrc_params.counts.z == 0u {
        let value = max(nrc_query(nrc_position(p), n), vec3<f32>(0.0));
        textureStore(nrc_noisy, vec2<i32>(id.xy), vec4<f32>(value, 0.0));
        return;
    }
    var seed = ((id.x + id.y * nrc_params.size.x) * 747796405u
        + nrc_params.counts.y * 2891336453u) ^ 0x27d4eb2fu;
    let d = nrc_bounce(n, n, &seed);
    let bias = max(4.0 * EPS, 2e-3 * distance(p, frame.camera_pos.xyz));
    let limit = nrc_params.tuning.y;
    let s = nrc_surface(Ray(p + n * bias, d), limit);
    var value = nrc_sky(d);
    var length = limit;
    if s.hit {
        value = nrc_direct(s, -d, &seed);
        if !s.unlit { value += diffuse(s.mat) * nrc_cached(s); }
        length = s.t;
    }
    value = clamp(value, vec3<f32>(0.0), vec3<f32>(nrc_params.tuning.z));
    textureStore(nrc_noisy, vec2<i32>(id.xy), vec4<f32>(value, length));
}
