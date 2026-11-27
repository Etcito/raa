//! M7 convex-hull SPOM/POM sampling (`m7-impostor-sampling`): `foliage_parallax.wgsl`
//! validates with the group-2 declarations of `m7-foliage-resources` and forward/prepass
//! callers, keeps its contract names and constants, and its traversal is checked through a
//! CPU twin of the same helpers: closed-form slab and plane intersections, step bracketing,
//! the SPOM silhouette boundary and malformed or empty samples.

use raa_shader::Composer;

const WGSL: &str = include_str!("../src/wgsl/foliage_parallax.wgsl");

const DECLARATIONS: &str = "
struct RaaFoliageParams {
    min: vec3<f32>, resolution: u32, max: vec3<f32>, mode: u32, translucency: f32,
    generation: u32, cutoff: f32, view_count: u32, texture_resolution: u32,
    zero0: u32, zero1: u32, zero2: u32,
}
struct RaaFoliageView { clip: mat4x4<f32>, near_far: vec4<f32> }
struct RaaFoliageViewTable { views: array<RaaFoliageView, 128> }
@group(2) @binding(0) var<uniform> raa_foliage_params: RaaFoliageParams;
@group(2) @binding(1) var raa_foliage_volume: texture_3d<f32>;
@group(2) @binding(2) var raa_foliage_sampler: sampler;
@group(2) @binding(3) var raa_foliage_albedo: texture_2d_array<f32>;
@group(2) @binding(4) var raa_foliage_normal_depth: texture_2d_array<f32>;
@group(2) @binding(5) var<uniform> raa_foliage_views: RaaFoliageViewTable;
";

const CALLERS: &str = "
@fragment
fn fs_forward(@location(0) p: vec3<f32>, @location(1) d: vec3<f32>) -> @location(0) vec4<f32> {
    let hit = foliage_parallax_hit(p, d, 1u);
    if (hit.coverage <= 0.0) {
        discard;
    }
    return vec4<f32>(hit.albedo * max(hit.normal.y, 0.0) + hit.position * 0.0, hit.coverage);
}

struct PrepassOut { @builtin(frag_depth) depth: f32, @location(0) normal: vec4<f32> }

@fragment
fn fs_prepass(@location(0) p: vec3<f32>, @location(1) d: vec3<f32>) -> PrepassOut {
    let hit = foliage_parallax_hit(p, d, 0u);
    if (hit.coverage <= 0.0) {
        discard;
    }
    return PrepassOut(hit.depth, vec4<f32>(hit.normal, hit.coverage));
}
";

fn module() -> naga::Module {
    naga::front::wgsl::parse_str(&format!("{DECLARATIONS}\n{WGSL}\n{CALLERS}")).expect("parse")
}

#[test]
fn validates_with_group2_declarations_and_callers() {
    let wgsl = format!("{DECLARATIONS}\n{WGSL}\n{CALLERS}");
    if let Err(e) = Composer::validate(&wgsl) {
        panic!("foliage_parallax.wgsl with forward and prepass callers: {e}");
    }
}

#[test]
fn foliage_hit_and_group2_layouts_match_the_contracts() {
    let module = module();
    let mut layouter = naga::proc::Layouter::default();
    layouter.update(module.to_ctx()).expect("layout");
    let members = |name: &str| {
        let (handle, ty) = module
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("missing {name}"));
        let naga::TypeInner::Struct { members, .. } = &ty.inner else {
            panic!("{name} is not a struct");
        };
        let list: Vec<(String, u32)> = members
            .iter()
            .map(|m| (m.name.clone().unwrap(), m.offset))
            .collect();
        (layouter[handle].size, list)
    };
    let (_, hit) = members("FoliageHit");
    let names: Vec<&str> = hit.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["position", "normal", "albedo", "coverage", "depth"]);
    let (size, params) = members("RaaFoliageParams");
    assert_eq!(size, 64);
    let offsets: Vec<u32> = params.iter().map(|(_, o)| *o).collect();
    assert_eq!(offsets, [0, 12, 16, 28, 32, 36, 40, 44, 48, 52, 56, 60]);
    let (size, view) = members("RaaFoliageView");
    assert_eq!((size, view[1].1), (80, 64));
}

#[test]
fn text_keeps_contract_names_and_twin_constants() {
    assert!(WGSL.contains(
        "fn foliage_parallax_hit(position: vec3<f32>, direction: vec3<f32>, mode: u32) -> FoliageHit"
    ));
    assert!(WGSL.contains("// Handoff: 'Compose relightable foliage materials'"));
    assert!(!WGSL.contains("@group") && !WGSL.contains("@binding"));
    assert!(!WGSL.contains("textureSample(") && WGSL.contains("textureSampleLevel("));
    for line in [
        format!("const FOLIAGE_PARALLAX_MIN_STEPS: u32 = {MIN_STEPS}u;"),
        format!("const FOLIAGE_PARALLAX_MAX_STEPS: u32 = {MAX_STEPS}u;"),
        format!("const FOLIAGE_PARALLAX_REFINE: u32 = {REFINE}u;"),
        format!("const FOLIAGE_PARALLAX_BOUNDARY: u32 = {BOUNDARY}u;"),
        "const FOLIAGE_PARALLAX_EPS: f32 = 1e-6;".into(),
        "const FOLIAGE_PARALLAX_GRAZING: f32 = 1e-3;".into(),
        "return vec4<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, ndc.z, 1.0);".into(),
        "return a + (b - a) * clamp(-fa / df, 0.0, 1.0);".into(),
        "return coverage > 0.0 && coverage >= threshold;".into(),
        "let threshold = select(0.0, raa_foliage_params.cutoff, mode == 0u);".into(),
    ] {
        assert!(WGSL.contains(&line), "missing {line}");
    }
}

// ---- CPU twin of the WGSL helpers (same names without the prefix, same constants) ----

const MIN_STEPS: u32 = 8;
const MAX_STEPS: u32 = 64;
const REFINE: u32 = 4;
const BOUNDARY: u32 = 6;
const EPS: f32 = 1e-6;
const GRAZING: f32 = 1e-3;

type V3 = [f32; 3];
/// Column-major root-to-clip matrix, as the view table stores it.
type Clip = [f32; 16];
/// Layer texels at (layer, uv): albedo/coverage and normal/depth.
type Texels = dyn Fn(usize, [f32; 2]) -> ([f32; 4], [f32; 4]);

fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: V3, s: f32) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn length(a: V3) -> f32 {
    dot(a, a).sqrt()
}
fn row(m: &Clip, j: usize) -> [f32; 4] {
    [m[j], m[4 + j], m[8 + j], m[12 + j]]
}
fn dot4(r: [f32; 4], p: V3) -> f32 {
    r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + r[3]
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Hit {
    position: V3,
    normal: V3,
    albedo: V3,
    coverage: f32,
    depth: f32,
}

#[derive(Clone, Copy)]
struct Sample {
    albedo: V3,
    coverage: f32,
    normal: V3,
    depth: f32,
}

const EMPTY: Sample = Sample {
    albedo: [0.0; 3],
    coverage: 0.0,
    normal: [0.0; 3],
    depth: 1.0,
};

#[derive(Clone, Copy)]
struct Probe {
    s: Sample,
    z: f32,
    f: f32,
    covered: bool,
}

struct Fixture {
    min: V3,
    max: V3,
    cutoff: f32,
    texture_resolution: u32,
    views: Vec<Clip>,
    texels: Box<Texels>,
}

fn miss(position: V3, ray: V3) -> Hit {
    Hit {
        position,
        normal: scale(ray, -1.0),
        albedo: [0.0; 3],
        coverage: 0.0,
        depth: 1.0,
    }
}

fn box_span(origin: V3, direction: V3, lo: V3, hi: V3) -> [f32; 2] {
    let (mut near, mut far) = (f32::MIN, f32::MAX);
    for i in 0..3 {
        let s = if direction[i] < 0.0 { -1.0 } else { 1.0 };
        let inv = s / direction[i].abs().max(EPS);
        let (a, b) = ((lo[i] - origin[i]) * inv, (hi[i] - origin[i]) * inv);
        near = near.max(a.min(b));
        far = far.min(a.max(b));
    }
    [near, far]
}

fn project(clip: &Clip, p: V3) -> [f32; 4] {
    let w = dot4(row(clip, 3), p);
    if w <= EPS {
        return [-1.0, -1.0, 1.0, 0.0];
    }
    let (x, y, z) = (
        dot4(row(clip, 0), p) / w,
        dot4(row(clip, 1), p) / w,
        dot4(row(clip, 2), p) / w,
    );
    [x * 0.5 + 0.5, 0.5 - y * 0.5, z, 1.0]
}

fn forward(clip: &Clip, p: V3) -> V3 {
    let r: Vec<[f32; 4]> = (0..4).map(|j| row(clip, j)).collect();
    let xyz = |v: [f32; 4]| [v[0], v[1], v[2]];
    let w = dot4(r[3], p);
    let a = add(scale(xyz(r[0]), w), scale(xyz(r[3]), -dot4(r[0], p)));
    let b = add(scale(xyz(r[1]), w), scale(xyz(r[3]), -dot4(r[1], p)));
    let f = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    let len = length(f);
    if len.is_nan() || len <= EPS {
        return [0.0; 3];
    }
    let dz = dot(xyz(r[2]), f) * w - dot(xyz(r[3]), f) * dot4(r[2], p);
    let sign = if dz >= 0.0 { 1.0 } else { -1.0 };
    scale(f, sign / len)
}

fn covered(coverage: f32, threshold: f32) -> bool {
    coverage > 0.0 && coverage >= threshold
}

fn sample(fx: &Fixture, layer: usize, uv: [f32; 2], threshold: f32) -> Sample {
    if !(uv.iter().all(|c| *c >= 0.0 && *c <= 1.0)) {
        return EMPTY;
    }
    let (ac, nd) = (fx.texels)(layer, uv);
    if covered(ac[3], threshold) && nd[3] >= 0.0 && nd[3] <= 1.0 {
        let c = |v: f32| v.clamp(0.0, 1.0);
        return Sample {
            albedo: [c(ac[0]), c(ac[1]), c(ac[2])],
            coverage: ac[3].min(1.0),
            normal: [nd[0], nd[1], nd[2]],
            depth: nd[3],
        };
    }
    EMPTY
}

fn secant(a: f32, fa: f32, b: f32, fb: f32) -> f32 {
    let df = fb - fa;
    if df.is_nan() || df <= EPS {
        return b;
    }
    a + (b - a) * (-fa / df).clamp(0.0, 1.0)
}

fn steps(texels: f32) -> u32 {
    if !(0.0..1e30).contains(&texels) {
        return MAX_STEPS;
    }
    texels.ceil().clamp(MIN_STEPS as f32, MAX_STEPS as f32) as u32
}

struct Ray<'a> {
    fx: &'a Fixture,
    clip: Clip,
    layer: usize,
    origin: V3,
    ray: V3,
    side: f32,
    threshold: f32,
}

impl Ray<'_> {
    fn probe(&self, t: f32) -> Probe {
        let pr = project(&self.clip, add(self.origin, scale(self.ray, t)));
        let s = if pr[3] > 0.5 {
            sample(self.fx, self.layer, [pr[0], pr[1]], self.threshold)
        } else {
            EMPTY
        };
        Probe {
            s,
            z: pr[2],
            f: self.side * (pr[2] - s.depth),
            covered: s.coverage > 0.0,
        }
    }

    fn result(&self, p: Probe, t: f32) -> Hit {
        let len = length(p.s.normal);
        let normal = if len > EPS && len < 1e30 {
            scale(p.s.normal, 1.0 / len)
        } else {
            scale(self.ray, -1.0)
        };
        Hit {
            position: add(self.origin, scale(self.ray, t)),
            normal,
            albedo: p.s.albedo,
            coverage: p.s.coverage,
            depth: p.z,
        }
    }
}

/// The twin of `foliage_parallax_hit`; also returns the march bracket (t before, t at the
/// first step at or behind the layer) and the step count when the march ran.
fn parallax_hit(
    fx: &Fixture,
    position: V3,
    direction: V3,
    mode: u32,
) -> (Hit, Option<[f32; 2]>, u32) {
    let len = length(direction);
    let ray = scale(direction, 1.0 / len.max(EPS));
    let count = fx.views.len().min(128);
    let finite = position.iter().all(|c| c.abs() < 1e30);
    if mode > 1 || count == 0 || !(len > EPS && len < 1e30) || !finite {
        return (miss(position, ray), None, 0);
    }
    let span = box_span(position, ray, fx.min, fx.max);
    let (t0, t1) = (span[0].max(0.0), span[1]);
    if (t1 - t0).partial_cmp(&EPS) != Some(std::cmp::Ordering::Greater) {
        return (miss(position, ray), None, 0);
    }
    let (mut layer, mut best) = (0, -2.0f32);
    for (i, clip) in fx.views.iter().take(count).enumerate() {
        let score = dot(forward(clip, position), ray);
        if score > best {
            (best, layer) = (score, i);
        }
    }
    if best.abs().partial_cmp(&GRAZING) != Some(std::cmp::Ordering::Greater) {
        return (miss(position, ray), None, 0);
    }
    let r = Ray {
        fx,
        clip: fx.views[layer],
        layer,
        origin: position,
        ray,
        side: if best > 0.0 { 1.0 } else { -1.0 },
        threshold: if mode == 0 { fx.cutoff } else { 0.0 },
    };
    let a = project(&r.clip, add(position, scale(ray, t0)));
    let b = project(&r.clip, add(position, scale(ray, t1)));
    let n = if a[3] > 0.5 && b[3] > 0.5 {
        let texels = (a[0] - b[0]).hypot(a[1] - b[1]) * fx.texture_resolution as f32;
        steps(texels)
    } else {
        MAX_STEPS
    };
    let mut prev = r.probe(t0);
    if prev.covered && prev.f >= 0.0 {
        return (r.result(prev, t0), Some([t0, t0]), n);
    }
    let (mut tp, mut tc, mut cur, mut found) = (t0, t0, prev, false);
    for i in 1..=n {
        tc = t0 + (t1 - t0) * (i as f32 / n as f32);
        cur = r.probe(tc);
        if cur.covered && cur.f >= 0.0 {
            found = true;
            break;
        }
        (tp, prev) = (tc, cur);
    }
    if !found {
        return (miss(position, ray), None, n);
    }
    let bracket = Some([tp, tc]);
    let (mut ta, mut fa) = (tp, prev.f.min(-EPS));
    if !prev.covered && mode == 0 {
        let (mut lo, mut hi, mut inside) = (tp, tc, cur);
        for _ in 0..BOUNDARY {
            let tm = 0.5 * (lo + hi);
            let pm = r.probe(tm);
            if pm.covered {
                (hi, inside) = (tm, pm);
            } else {
                lo = tm;
            }
        }
        if inside.f >= 0.0 {
            return (r.result(inside, hi), bracket, n);
        }
        (ta, fa) = (hi, inside.f.min(-EPS));
    }
    let (mut tb, mut hit) = (tc, cur);
    for _ in 0..REFINE {
        let tm = secant(ta, fa, tb, hit.f);
        let pm = r.probe(tm);
        if pm.covered && pm.f >= 0.0 {
            (tb, hit) = (tm, pm);
        } else {
            (ta, fa) = (tm, pm.f.min(-EPS));
        }
    }
    (r.result(hit, tb), bracket, n)
}

// ---- Fixtures ----

fn from_rows(r: [[f32; 4]; 4]) -> Clip {
    let mut m = [0.0; 16];
    for (j, row) in r.iter().enumerate() {
        for (c, v) in row.iter().enumerate() {
            m[4 * c + j] = *v;
        }
    }
    m
}

/// Orthographic view down -Z over [-1, 1]^3: ndc = (x, y), depth = (1 - z) / 2.
fn ortho() -> Clip {
    from_rows([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -0.5, 0.5],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

/// Layer of the plane z = z0 + kx over the whole orthographic layer, coverage `alpha`
/// inside the disk |xy| < radius.
fn plane(z0: f32, k: f32, alpha: f32, radius: f32, cutoff: f32) -> Fixture {
    Fixture {
        min: [-1.0; 3],
        max: [1.0; 3],
        cutoff,
        texture_resolution: 64,
        views: vec![ortho()],
        texels: Box::new(move |_, uv| {
            let (x, y) = (2.0 * uv[0] - 1.0, 1.0 - 2.0 * uv[1]);
            let a = if x.hypot(y) < radius { alpha } else { 0.0 };
            (
                [0.4, 0.6, 0.2, a],
                [0.1, 0.2, 0.9, (1.0 - (z0 + k * x)) / 2.0],
            )
        }),
    }
}

fn normalize(v: V3) -> V3 {
    scale(v, 1.0 / length(v))
}

fn close(a: V3, b: V3, tol: f32) -> bool {
    (0..3).all(|i| (a[i] - b[i]).abs() <= tol)
}

// ---- Tests of the twin ----

#[test]
fn slab_intersection_is_closed_form() {
    let (lo, hi) = ([0.0; 3], [1.0; 3]);
    assert_eq!(
        box_span([-2.0, 0.5, 0.5], [1.0, 0.0, 0.0], lo, hi),
        [2.0, 3.0]
    );
    let d = normalize([1.0, 1.0, 1.0]);
    let s = box_span([-1.0; 3], d, lo, hi);
    assert!((s[0] - 3f32.sqrt()).abs() < 1e-5 && (s[1] - 2.0 * 3f32.sqrt()).abs() < 1e-5);
    let inside = box_span([0.25, 0.5, 0.5], [-1.0, 0.0, 0.0], lo, hi);
    assert!(inside[0] < 0.0 && (inside[1] - 0.25).abs() < 1e-6);
    // A parallel ray keeps an unbounded slab inside and an empty one outside.
    let parallel = box_span([0.5, 0.5, -1.0], [0.0, 0.0, 1.0], lo, hi);
    assert!((parallel[0] - 1.0).abs() < 1e-5 && (parallel[1] - 2.0).abs() < 1e-5);
    let outside = box_span([2.0, 0.5, -1.0], [0.0, 0.0, 1.0], lo, hi);
    assert!(outside[0] > outside[1]);
}

#[test]
fn view_forward_keeps_layer_coordinates_and_increases_depth() {
    assert!(close(
        forward(&ortho(), [0.3, -0.2, 0.1]),
        [0.0, 0.0, -1.0],
        1e-6
    ));
    let (n, f) = (1.0f32, 5.0f32);
    let c = f / (f - n);
    let persp = from_rows([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -c, 3.0 * c - c * n],
        [0.0, 0.0, -1.0, 3.0],
    ]);
    let p = [0.4, -0.3, 0.5];
    let fwd = forward(&persp, p);
    assert!(
        close(fwd, normalize(add(p, [0.0, 0.0, -3.0])), 1e-5),
        "{fwd:?}"
    );
    let (a, b) = (project(&persp, p), project(&persp, add(p, scale(fwd, 0.7))));
    assert!((a[0] - b[0]).abs() < 1e-5 && (a[1] - b[1]).abs() < 1e-5 && b[2] > a[2]);
    assert_eq!(project(&persp, [0.0, 0.0, 3.5])[3], 0.0, "behind the eye");
}

#[test]
fn pom_and_spom_find_the_closed_form_plane_crossing() {
    let fx = plane(0.2, 0.3, 1.0, 10.0, 0.5);
    let (o, d) = ([-0.5, 0.1, 1.0], normalize([0.3, 0.05, -1.0]));
    // z0 + t dz = 0.2 + 0.3 (x0 + t dx)
    let t = (0.2 + 0.3 * o[0] - o[2]) / (d[2] - 0.3 * d[0]);
    let expected = add(o, scale(d, t));
    for mode in [0, 1] {
        let (hit, bracket, n) = parallax_hit(&fx, o, d, mode);
        assert!(close(hit.position, expected, 1e-4), "mode {mode}: {hit:?}");
        assert!((hit.depth - (1.0 - expected[2]) / 2.0).abs() < 1e-4);
        assert!(
            close(hit.normal, normalize([0.1, 0.2, 0.9]), 1e-6),
            "baked normal"
        );
        assert_eq!((hit.albedo, hit.coverage), ([0.4, 0.6, 0.2], 1.0));
        let [tp, tc] = bracket.unwrap();
        assert!((MIN_STEPS..=MAX_STEPS).contains(&n));
        let span = box_span(o, d, fx.min, fx.max);
        let step = (span[1] - span[0].max(0.0)) / n as f32;
        assert!(tp < t && t <= tc + 1e-5 && (tc - tp - step).abs() < 1e-5);
    }
}

#[test]
fn spom_resolves_the_silhouette_wall_and_pom_crosses_depth() {
    // Plane z = 0 inside the disk r < 0.5; the ray passes z = 0 at x = -0.7, outside it.
    let fx = plane(0.0, 0.0, 1.0, 0.5, 0.5);
    let (o, d) = ([-0.9, 0.0, 0.2], normalize([1.0, 0.0, -1.0]));
    let (spom, bracket, n) = parallax_hit(&fx, o, d, 0);
    let [tp, tc] = bracket.unwrap();
    let tol = (tc - tp) / (1u32 << BOUNDARY) as f32 + 1e-4;
    assert!((spom.position[0] + 0.5).abs() <= tol, "{spom:?} n {n}");
    assert!((spom.position[2] + 0.2).abs() <= tol && spom.coverage == 1.0);
    let (pom, _, _) = parallax_hit(&fx, o, d, 1);
    assert!(pom.coverage == 1.0 && pom.position[2] < 0.0);
    assert!(pom.depth >= 0.5, "POM hit lies at or behind the layer");
}

#[test]
fn spom_discards_outside_the_cutoff_silhouette() {
    let fx = plane(0.0, 0.0, 1.0, 0.5, 0.5);
    let (o, d) = ([-0.9, 0.8, 0.5], normalize([1.0, 0.0, -0.5]));
    assert_eq!(parallax_hit(&fx, o, d, 0).0.coverage, 0.0);
    assert_eq!(parallax_hit(&fx, o, d, 1).0.coverage, 0.0);
    let thin = plane(0.0, 0.0, 0.3, 10.0, 0.5);
    let (o, d) = ([0.0, 0.0, 0.8], [0.0, 0.0, -1.0]);
    assert_eq!(parallax_hit(&thin, o, d, 0).0, miss(o, d));
    let pom = parallax_hit(&thin, o, d, 1).0;
    assert!((pom.coverage - 0.3).abs() < 1e-6 && (pom.position[2]).abs() < 1e-4);
}

#[test]
fn a_ray_towards_the_source_view_meets_the_layer_from_behind() {
    let fx = plane(0.0, 0.0, 1.0, 10.0, 0.5);
    let (hit, _, _) = parallax_hit(&fx, [0.0, 0.0, -0.8], normalize([0.2, 0.0, 1.0]), 1);
    assert!(
        hit.coverage == 1.0 && hit.position[2].abs() < 1e-4,
        "{hit:?}"
    );
}

#[test]
fn malformed_and_empty_inputs_return_misses() {
    let fx = plane(0.0, 0.0, 1.0, 10.0, 0.5);
    let o = [0.0, 0.0, 0.8];
    let down = [0.0, 0.0, -1.0];
    for (p, d, mode) in [
        (o, [0.0; 3], 1),
        (o, [f32::NAN, 0.0, -1.0], 1),
        ([f32::NAN, 0.0, 0.0], down, 0),
        (o, down, 2),
        (o, down, 5),
        ([0.0, 0.0, 1.5], [0.0, 0.0, 1.0], 1), // leaves the bounds at once
        (o, [1.0, 0.0, 0.0], 0),               // grazing: perpendicular to the view
    ] {
        assert_eq!(
            parallax_hit(&fx, p, d, mode).0.coverage,
            0.0,
            "{p:?} {d:?} {mode}"
        );
    }
    let no_views = Fixture {
        views: Vec::new(),
        ..plane(0.0, 0.0, 1.0, 10.0, 0.5)
    };
    assert_eq!(parallax_hit(&no_views, o, down, 1).0, miss(o, down));
    let empty = plane(0.0, 0.0, 0.0, 10.0, 0.0);
    assert_eq!(parallax_hit(&empty, o, down, 0).0, miss(o, down));
    for depth in [f32::NAN, 1.5, -0.1] {
        let bad = Fixture {
            texels: Box::new(move |_, _| ([1.0; 4], [0.0, 1.0, 0.0, depth])),
            ..plane(0.0, 0.0, 1.0, 10.0, 0.5)
        };
        assert_eq!(
            parallax_hit(&bad, o, down, 1).0.coverage,
            0.0,
            "depth {depth}"
        );
    }
    for uv in [[1.2, 0.5], [-0.01, 0.5], [0.5, f32::NAN]] {
        assert_eq!(sample(&fx, 0, uv, 0.0).coverage, 0.0, "{uv:?}");
    }
    // A zero normal falls back to the reversed ray, never to the hull face.
    let flat = Fixture {
        texels: Box::new(|_, _| ([1.0; 4], [0.0, 0.0, 0.0, 0.5])),
        ..plane(0.0, 0.0, 1.0, 10.0, 0.5)
    };
    assert_eq!(parallax_hit(&flat, o, down, 1).0.normal, [0.0, 0.0, 1.0]);
}

#[test]
fn step_count_and_secant_are_bounded() {
    assert_eq!(steps(0.0), MIN_STEPS);
    assert_eq!(steps(20.2), 21);
    assert_eq!(steps(1e3), MAX_STEPS);
    assert_eq!(steps(f32::NAN), MAX_STEPS);
    assert_eq!(steps(f32::INFINITY), MAX_STEPS);
    assert_eq!(secant(1.0, -0.5, 3.0, 1.5), 1.5);
    assert_eq!(secant(1.0, 0.5, 3.0, 0.5), 3.0, "flat bracket keeps b");
    assert!(covered(0.5, 0.5) && !covered(0.0, 0.0) && !covered(0.4, 0.5));
}
