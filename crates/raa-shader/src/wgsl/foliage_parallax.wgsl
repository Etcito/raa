// M7 convex-hull parallax sampling (contract m7-impostor-sampling): SPOM (mode 0) and POM
// (mode 1) view-space ray traversal of the baked convex-hull layers of group 2
// (m7-foliage-resources). This file declares no binding and no group-2 type: the composing
// pass declares `RaaFoliageParams` (64 bytes), `RaaFoliageView` (mat4x4 `clip` + vec4
// `near_far`, 80 bytes), `RaaFoliageViewTable` (`views: array<RaaFoliageView, 128>`) and
// `raa_foliage_params` (binding 0), `raa_foliage_sampler` (binding 2), `raa_foliage_albedo`
// (binding 3, linear RGB + coverage A), `raa_foliage_normal_depth` (binding 4, source-root
// normal XYZ + standard-Z view depth W) and `raa_foliage_views` (binding 5).

const FOLIAGE_PARALLAX_MIN_STEPS: u32 = 8u;
const FOLIAGE_PARALLAX_MAX_STEPS: u32 = 64u;
const FOLIAGE_PARALLAX_REFINE: u32 = 4u;
const FOLIAGE_PARALLAX_BOUNDARY: u32 = 6u;
const FOLIAGE_PARALLAX_EPS: f32 = 1e-6;
const FOLIAGE_PARALLAX_GRAZING: f32 = 1e-3;

struct FoliageHit {
    position: vec3<f32>,
    normal: vec3<f32>,
    albedo: vec3<f32>,
    coverage: f32,
    depth: f32,
}

// One layer sample: `depth` is the clear depth 1 and `coverage` 0 outside the silhouette,
// outside [0, 1] layer coordinates or on a malformed texel.
struct FoliageParallaxSample {
    albedo: vec3<f32>,
    coverage: f32,
    normal: vec3<f32>,
    depth: f32,
}

fn foliage_parallax_miss(position: vec3<f32>, direction: vec3<f32>) -> FoliageHit {
    return FoliageHit(position, -direction, vec3<f32>(0.0), 0.0, 1.0);
}

// Closed-form slab intersection of the ray with the source-root bounds: (near, far), empty
// when near > far. A zero direction component keeps its sign through a clamped inverse.
fn foliage_parallax_box(origin: vec3<f32>, direction: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>) -> vec2<f32> {
    let s = select(vec3<f32>(1.0), vec3<f32>(-1.0), direction < vec3<f32>(0.0));
    let inv = s / max(abs(direction), vec3<f32>(FOLIAGE_PARALLAX_EPS));
    let a = (lo - origin) * inv;
    let b = (hi - origin) * inv;
    let near = min(a, b);
    let far = max(a, b);
    return vec2<f32>(max(max(near.x, near.y), near.z), min(min(far.x, far.y), far.z));
}

// Layer coordinates (u, v), standard-Z depth and validity (w = 1) of `p` in a source view.
fn foliage_parallax_project(clip: mat4x4<f32>, p: vec3<f32>) -> vec4<f32> {
    let c = clip * vec4<f32>(p, 1.0);
    if (c.w <= FOLIAGE_PARALLAX_EPS) {
        return vec4<f32>(-1.0, -1.0, 1.0, 0.0);
    }
    let ndc = c.xyz / c.w;
    return vec4<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, ndc.z, 1.0);
}

// The unit source-view ray direction through `p` (layer coordinates constant along it, depth
// increasing), for orthographic and perspective clip matrices; zero when degenerate.
fn foliage_parallax_forward(clip: mat4x4<f32>, p: vec3<f32>) -> vec3<f32> {
    let m = transpose(clip);
    let q = vec4<f32>(p, 1.0);
    let w = dot(m[3], q);
    let a = m[0].xyz * w - m[3].xyz * dot(m[0], q);
    let b = m[1].xyz * w - m[3].xyz * dot(m[1], q);
    let f = cross(a, b);
    let len = length(f);
    if (!(len > FOLIAGE_PARALLAX_EPS)) {
        return vec3<f32>(0.0);
    }
    let dz = dot(m[2].xyz, f) * w - dot(m[3].xyz, f) * dot(m[2], q);
    return select(-f, f, dz >= 0.0) / len;
}

// Covered: positive coverage at or above `threshold` (the cutoff for SPOM, 0 for POM).
fn foliage_parallax_covered(coverage: f32, threshold: f32) -> bool {
    return coverage > 0.0 && coverage >= threshold;
}

fn foliage_parallax_sample(layer: u32, uv: vec2<f32>, threshold: f32) -> FoliageParallaxSample {
    var s = FoliageParallaxSample(vec3<f32>(0.0), 0.0, vec3<f32>(0.0), 1.0);
    if (!(all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0)))) {
        return s;
    }
    let ac = textureSampleLevel(raa_foliage_albedo, raa_foliage_sampler, uv, layer, 0.0);
    let nd = textureSampleLevel(raa_foliage_normal_depth, raa_foliage_sampler, uv, layer, 0.0);
    if (foliage_parallax_covered(ac.a, threshold) && nd.w >= 0.0 && nd.w <= 1.0) {
        s = FoliageParallaxSample(clamp(ac.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), min(ac.a, 1.0), nd.xyz, nd.w);
    }
    return s;
}

// Secant step of the bracket fa < 0 <= fb; returns b when the bracket is flat.
fn foliage_parallax_secant(a: f32, fa: f32, b: f32, fb: f32) -> f32 {
    let df = fb - fa;
    if (!(df > FOLIAGE_PARALLAX_EPS)) {
        return b;
    }
    return a + (b - a) * clamp(-fa / df, 0.0, 1.0);
}

// Number of march steps for a ray spanning `texels` layer texels.
fn foliage_parallax_steps(texels: f32) -> u32 {
    if (!(texels >= 0.0 && texels < 1e30)) {
        return FOLIAGE_PARALLAX_MAX_STEPS;
    }
    return u32(clamp(ceil(texels), f32(FOLIAGE_PARALLAX_MIN_STEPS), f32(FOLIAGE_PARALLAX_MAX_STEPS)));
}

// The ray sample at distance t: `f = side * (ray depth - layer depth)`, at or behind the
// layer when `f >= 0`.
struct FoliageParallaxProbe {
    s: FoliageParallaxSample,
    z: f32,
    f: f32,
    covered: bool,
}

fn foliage_parallax_probe(clip: mat4x4<f32>, layer: u32, origin: vec3<f32>, ray: vec3<f32>, t: f32, side: f32, threshold: f32) -> FoliageParallaxProbe {
    let pr = foliage_parallax_project(clip, origin + ray * t);
    var s = FoliageParallaxSample(vec3<f32>(0.0), 0.0, vec3<f32>(0.0), 1.0);
    if (pr.w > 0.5) {
        s = foliage_parallax_sample(layer, pr.xy, threshold);
    }
    return FoliageParallaxProbe(s, pr.z, side * (pr.z - s.depth), s.coverage > 0.0);
}

fn foliage_parallax_result(p: FoliageParallaxProbe, position: vec3<f32>, ray: vec3<f32>) -> FoliageHit {
    let len = length(p.s.normal);
    let normal = select(-ray, p.s.normal / max(len, FOLIAGE_PARALLAX_EPS), len > FOLIAGE_PARALLAX_EPS && len < 1e30);
    return FoliageHit(position, normal, p.s.albedo, p.s.coverage, p.z);
}

// Handoff: 'Compose relightable foliage materials' composes this file and calls
// `foliage_parallax_hit`; today it marches the hull ray `position + t * direction`
// (source-root, entered at the hull fragment) through the bounds in the best-facing source
// view of the Rust table order. POM (mode 1) refines the depth crossing by secant steps;
// SPOM (mode 0) first resolves the coverage boundary against the cutoff and returns a miss
// outside the baked silhouette. The result holds the reconstructed source-root position,
// the baked source-root normal (independent of the hull face), linear unlit albedo,
// coverage (0 = miss, the cutout input of forward and prepass alike) and the standard-Z
// depth of the selected view. Any other mode, a zero or non-finite ray, an empty view table,
// a ray leaving the bounds at once or a grazing view returns a miss.
fn foliage_parallax_hit(position: vec3<f32>, direction: vec3<f32>, mode: u32) -> FoliageHit {
    let len = length(direction);
    let ray = direction / max(len, FOLIAGE_PARALLAX_EPS);
    let count = min(raa_foliage_params.view_count, 128u);
    if (mode > 1u || count == 0u || !(len > FOLIAGE_PARALLAX_EPS && len < 1e30) || !all(abs(position) < vec3<f32>(1e30))) {
        return foliage_parallax_miss(position, ray);
    }
    let span = foliage_parallax_box(position, ray, raa_foliage_params.min, raa_foliage_params.max);
    let t0 = max(span.x, 0.0);
    let t1 = span.y;
    if (!(t1 - t0 > FOLIAGE_PARALLAX_EPS)) {
        return foliage_parallax_miss(position, ray);
    }
    var layer = 0u;
    var best = -2.0;
    for (var i = 0u; i < count; i++) {
        let score = dot(foliage_parallax_forward(raa_foliage_views.views[i].clip, position), ray);
        if (score > best) {
            best = score;
            layer = i;
        }
    }
    if (!(abs(best) > FOLIAGE_PARALLAX_GRAZING)) {
        return foliage_parallax_miss(position, ray);
    }
    let side = select(-1.0, 1.0, best > 0.0);
    let clip = raa_foliage_views.views[layer].clip;
    let threshold = select(0.0, raa_foliage_params.cutoff, mode == 0u);
    let a = foliage_parallax_project(clip, position + ray * t0);
    let b = foliage_parallax_project(clip, position + ray * t1);
    var steps = FOLIAGE_PARALLAX_MAX_STEPS;
    if (a.w > 0.5 && b.w > 0.5) {
        steps = foliage_parallax_steps(distance(a.xy, b.xy) * f32(raa_foliage_params.texture_resolution));
    }
    var prev = foliage_parallax_probe(clip, layer, position, ray, t0, side, threshold);
    if (prev.covered && prev.f >= 0.0) {
        return foliage_parallax_result(prev, position + ray * t0, ray);
    }
    var tp = t0;
    var tc = t0;
    var cur = prev;
    var found = false;
    for (var i = 1u; i <= steps; i++) {
        tc = mix(t0, t1, f32(i) / f32(steps));
        cur = foliage_parallax_probe(clip, layer, position, ray, tc, side, threshold);
        if (cur.covered && cur.f >= 0.0) {
            found = true;
            break;
        }
        tp = tc;
        prev = cur;
    }
    if (!found) {
        return foliage_parallax_miss(position, ray);
    }
    var ta = tp;
    var fa = min(prev.f, -FOLIAGE_PARALLAX_EPS);
    if (!prev.covered && mode == 0u) {
        // SPOM: bisect the silhouette boundary between the outside and the inside sample.
        var lo = tp;
        var hi = tc;
        var inside = cur;
        for (var k = 0u; k < FOLIAGE_PARALLAX_BOUNDARY; k++) {
            let tm = 0.5 * (lo + hi);
            let pm = foliage_parallax_probe(clip, layer, position, ray, tm, side, threshold);
            if (pm.covered) {
                hi = tm;
                inside = pm;
            } else {
                lo = tm;
            }
        }
        if (inside.f >= 0.0) {
            return foliage_parallax_result(inside, position + ray * hi, ray);
        }
        ta = hi;
        fa = min(inside.f, -FOLIAGE_PARALLAX_EPS);
    }
    var tb = tc;
    var hit = cur;
    for (var k = 0u; k < FOLIAGE_PARALLAX_REFINE; k++) {
        let tm = foliage_parallax_secant(ta, fa, tb, hit.f);
        let pm = foliage_parallax_probe(clip, layer, position, ray, tm, side, threshold);
        if (pm.covered && pm.f >= 0.0) {
            tb = tm;
            hit = pm;
        } else {
            ta = tm;
            fa = min(pm.f, -FOLIAGE_PARALLAX_EPS);
        }
    }
    return foliage_parallax_result(hit, position + ray * tb, ray);
}
