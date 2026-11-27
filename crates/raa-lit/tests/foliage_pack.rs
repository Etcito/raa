//! Packed convex-hull layers (M7, contract m7-reference-views) on the CPU: a synthetic
//! reference set of one leaf card, rasterized per texel as `pass_foliage_reference.wgsl`
//! writes it (the fragment position z as standard-Z depth), packed through
//! [`pack_texel`] (the mirror of `foliage_pack.wgsl`), has coverage and a depth below the
//! clear depth inside the silhouette; and a CPU mirror of `foliage_parallax_hit` for one
//! view finds the card from a hull fragment. The depth the reference pass wrote before
//! (z divided by the fragment w, that is z times clip w) lies above 1 on every covered
//! texel, so frag_depth clamped to the clear depth and the Less test dropped every layer.
//! A texel without a reference fragment packs coverage 0 and the clear depth 1: with the
//! empty depth 0 the linear sampler blended a near depth into every silhouette edge, the
//! march hit at the hull entry and the whole hull was filled.

#![cfg(not(target_arch = "wasm32"))]

use raa_lit::foliage_reference::{
    ReferenceCamera, pack_texel, reference_cameras, world_to_root_normal,
};
use raa_math::{Mat4, Quat, Vec3};

const RESOLUTION: usize = 64;
const VIEWS: u32 = 8;
const MIN: [f32; 3] = [-2.0, 0.0, -2.0];
const MAX: [f32; 3] = [2.0, 6.0, 2.0];
/// The leaf card: the root plane z = 0 over x in -1.5..1.5 and y in 1..5.
const CARD: [f32; 4] = [-1.5, 1.5, 1.0, 5.0];
const CUTOFF: f32 = 0.5;
/// What the reference targets hold where no fragment landed: the normal/coverage target is
/// cleared to zero and the standard-Z depth target to the clear depth 1.
const EMPTY_NORMAL_COVERAGE: [f32; 4] = [0.0; 4];
const EMPTY_REFERENCE_DEPTH: f32 = 1.0;

fn root_world() -> Mat4 {
    Mat4::from_trs(
        Vec3::new(3.0, 0.0, -2.0),
        Quat::from_rotation_y(0.6),
        Vec3::new(1.5, 1.5, 1.5),
    )
}

fn mul4(m: &[f32; 16], v: [f32; 4]) -> [f32; 4] {
    [0, 1, 2, 3].map(|r| (0..4).map(|c| m[c * 4 + r] * v[c]).sum())
}

fn row(m: &[f32; 16], r: usize) -> [f32; 4] {
    [m[r], m[4 + r], m[8 + r], m[12 + r]]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn dot4(a: [f32; 4], b: [f32; 4]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

fn add(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [a[0] + b[0] * t, a[1] + b[1] * t, a[2] + b[2] * t]
}

/// One packed layer: the copied albedo/coverage alpha and the packed normal/depth texels.
struct Layer {
    clip: [f32; 16],
    coverage: Vec<f32>,
    normal_depth: Vec<[f32; 4]>,
}

/// Rasterizes the card for one reference view at texel centres. Returns the packed layer
/// and, per covered texel, the depth the earlier reference pass wrote (z times clip w).
fn reference_layer(camera: &ReferenceCamera, world_to_root: &[[f32; 4]; 3]) -> (Layer, Vec<f32>) {
    let clip = camera.root_to_clip.to_cols_array();
    let inverse = camera.root_to_clip.inverse().unwrap().to_cols_array();
    let world_normal = root_world().transform_vector(Vec3::new(0.0, 0.0, 1.0));
    let mut coverage = vec![0.0; RESOLUTION * RESOLUTION];
    let empty = pack_texel(world_to_root, EMPTY_NORMAL_COVERAGE, EMPTY_REFERENCE_DEPTH);
    let mut normal_depth = vec![empty; RESOLUTION * RESOLUTION];
    let mut old_depths = Vec::new();
    for ty in 0..RESOLUTION {
        for tx in 0..RESOLUTION {
            let u = (tx as f32 + 0.5) / RESOLUTION as f32;
            let v = (ty as f32 + 0.5) / RESOLUTION as f32;
            let ndc = [2.0 * u - 1.0, 1.0 - 2.0 * v];
            let unproject = |z: f32| {
                let p = mul4(&inverse, [ndc[0], ndc[1], z, 1.0]);
                [p[0] / p[3], p[1] / p[3], p[2] / p[3]]
            };
            let (a, b) = (unproject(0.0), unproject(1.0));
            let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            if d[2].abs() < 1e-9 {
                continue;
            }
            let p = add(a, d, -a[2] / d[2]);
            let t = -a[2] / d[2];
            if !(0.0..=1.0).contains(&t)
                || p[0] < CARD[0]
                || p[0] > CARD[1]
                || p[1] < CARD[2]
                || p[1] > CARD[3]
            {
                continue;
            }
            let c = mul4(&clip, [p[0], p[1], p[2], 1.0]);
            // The fragment position of the rasterized card: z / w as z and 1 / w as w.
            let (position_z, position_w) = (c[2] / c[3], 1.0 / c[3]);
            old_depths.push(position_z / position_w);
            let facing = if d[2] > 0.0 { -1.0 } else { 1.0 };
            let n = world_normal * facing;
            let index = ty * RESOLUTION + tx;
            coverage[index] = 1.0;
            normal_depth[index] = pack_texel(world_to_root, [n.x, n.y, n.z, 1.0], position_z);
        }
    }
    let layer = Layer {
        clip,
        coverage,
        normal_depth,
    };
    (layer, old_depths)
}

fn reference_set() -> Vec<(Layer, Vec<f32>)> {
    let world_to_root = world_to_root_normal(root_world()).unwrap();
    reference_cameras(MIN, MAX, root_world(), VIEWS)
        .unwrap()
        .iter()
        .map(|camera| reference_layer(camera, &world_to_root))
        .collect()
}

#[test]
fn packed_layers_cover_the_silhouette_with_a_depth_below_the_clear_depth() {
    let set = reference_set();
    let mut covered_views = 0;
    for (view, (layer, old_depths)) in set.iter().enumerate() {
        let covered: Vec<usize> = (0..layer.coverage.len())
            .filter(|&i| layer.coverage[i] > 0.0)
            .collect();
        if covered.is_empty() {
            continue;
        }
        covered_views += 1;
        for &i in &covered {
            let [x, y, z, depth] = layer.normal_depth[i];
            assert!(
                depth > 0.0 && depth < 1.0,
                "view {view} texel {i}: packed depth {depth} must lie inside the frustum"
            );
            assert!(
                x.abs() < 1e-4 && y.abs() < 1e-4 && (z.abs() - 1.0).abs() < 1e-4,
                "view {view} texel {i}: the card normal packs to the root z axis, got {x} {y} {z}"
            );
        }
        assert!(
            old_depths.iter().all(|&d| d > 1.0),
            "view {view}: the depth z times clip w lies above the clear depth 1"
        );
    }
    assert!(
        covered_views * 2 >= VIEWS as usize,
        "the card covers texels in {covered_views} of {VIEWS} views"
    );
}

#[test]
fn pack_texel_rotates_normals_into_the_root_and_clamps_depth() {
    let world_to_root = world_to_root_normal(root_world()).unwrap();
    let n = root_world()
        .transform_vector(Vec3::new(1.0, 0.0, 0.0))
        .normalize_or_zero();
    let packed = pack_texel(&world_to_root, [n.x, n.y, n.z, 1.0], 1.5);
    assert!((packed[0] - 1.0).abs() < 1e-4 && packed[1].abs() < 1e-4 && packed[2].abs() < 1e-4);
    assert_eq!(packed[3], 1.0);
    assert_eq!(
        pack_texel(&world_to_root, [0.0; 4], -0.5),
        [0.0, 0.0, 0.0, 1.0]
    );
}

#[test]
fn texel_without_a_reference_fragment_packs_coverage_zero_and_the_clear_depth() {
    let world_to_root = world_to_root_normal(root_world()).unwrap();
    let n = root_world()
        .transform_vector(Vec3::new(0.0, 1.0, 0.0))
        .normalize_or_zero();
    // The cleared targets (depth 1), the earlier depth clear 0, and a stray depth or normal
    // under zero coverage all pack the empty texel: no normal and the clear depth 1.
    for (normal_coverage, depth) in [
        (EMPTY_NORMAL_COVERAGE, EMPTY_REFERENCE_DEPTH),
        (EMPTY_NORMAL_COVERAGE, 0.0),
        ([n.x, n.y, n.z, 0.0], 0.4),
    ] {
        assert_eq!(
            pack_texel(&world_to_root, normal_coverage, depth),
            [0.0, 0.0, 0.0, 1.0],
            "reference {normal_coverage:?} depth {depth}"
        );
    }
    let (layer, _) = reference_layer(
        &reference_cameras(MIN, MAX, root_world(), VIEWS).unwrap()[0],
        &world_to_root,
    );
    for (i, texel) in layer.normal_depth.iter().enumerate() {
        if layer.coverage[i] == 0.0 {
            assert_eq!(*texel, [0.0, 0.0, 0.0, 1.0], "empty texel {i}");
        }
    }
}

// ---- CPU mirror of foliage_parallax.wgsl (nearest texel instead of the linear sampler) ----

const EPS: f32 = 1e-6;

struct Sample {
    coverage: f32,
    depth: f32,
}

struct Probe {
    s: Sample,
    z: f32,
    f: f32,
    covered: bool,
}

fn parallax_box(origin: [f32; 3], direction: [f32; 3]) -> (f32, f32) {
    let (mut near, mut far) = (f32::MIN, f32::MAX);
    for axis in 0..3 {
        let s = if direction[axis] < 0.0 { -1.0 } else { 1.0 };
        let inv = s / direction[axis].abs().max(EPS);
        let a = (MIN[axis] - origin[axis]) * inv;
        let b = (MAX[axis] - origin[axis]) * inv;
        near = near.max(a.min(b));
        far = far.min(a.max(b));
    }
    (near, far)
}

fn project(clip: &[f32; 16], p: [f32; 3]) -> [f32; 4] {
    let c = mul4(clip, [p[0], p[1], p[2], 1.0]);
    if c[3] <= EPS {
        return [-1.0, -1.0, 1.0, 0.0];
    }
    let ndc = [c[0] / c[3], c[1] / c[3], c[2] / c[3]];
    [ndc[0] * 0.5 + 0.5, 0.5 - ndc[1] * 0.5, ndc[2], 1.0]
}

fn forward(clip: &[f32; 16], p: [f32; 3]) -> [f32; 3] {
    let (m0, m1, m2, m3) = (row(clip, 0), row(clip, 1), row(clip, 2), row(clip, 3));
    let q = [p[0], p[1], p[2], 1.0];
    let w = dot4(m3, q);
    let xyz = |m: [f32; 4]| [m[0], m[1], m[2]];
    let (d0, d1) = (dot4(m0, q), dot4(m1, q));
    let a = [0, 1, 2].map(|i| m0[i] * w - m3[i] * d0);
    let b = [0, 1, 2].map(|i| m1[i] * w - m3[i] * d1);
    let f = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    let len = dot3(f, f).sqrt();
    if len.is_nan() || len <= EPS {
        return [0.0; 3];
    }
    let dz = dot3(xyz(m2), f) * w - dot3(xyz(m3), f) * dot4(m2, q);
    let s = if dz >= 0.0 { 1.0 } else { -1.0 } / len;
    f.map(|v| v * s)
}

/// The GPU linear sampler at `uv` (clamp to edge): bilinear coverage and packed depth,
/// accepted like `foliage_parallax_sample`.
fn sample_linear(layer: &Layer, uv: [f32; 2], threshold: f32) -> Sample {
    let miss = Sample {
        coverage: 0.0,
        depth: 1.0,
    };
    if !(uv.iter().all(|&c| (0.0..=1.0).contains(&c))) {
        return miss;
    }
    let r = RESOLUTION as f32;
    let (x, y) = (uv[0] * r - 0.5, uv[1] * r - 0.5);
    let (fx, fy) = (x - x.floor(), y - y.floor());
    let at = |c: f32| (c.max(0.0) as usize).min(RESOLUTION - 1);
    let (x0, y0) = (at(x.floor()), at(y.floor()));
    let (x1, y1) = (at(x.floor() + 1.0), at(y.floor() + 1.0));
    let (mut a, mut depth) = (0.0, 0.0);
    for (tx, ty, w) in [
        (x0, y0, (1.0 - fx) * (1.0 - fy)),
        (x1, y0, fx * (1.0 - fy)),
        (x0, y1, (1.0 - fx) * fy),
        (x1, y1, fx * fy),
    ] {
        a += w * layer.coverage[ty * RESOLUTION + tx];
        depth += w * layer.normal_depth[ty * RESOLUTION + tx][3];
    }
    if a > 0.0 && a >= threshold && (0.0..=1.0).contains(&depth) {
        return Sample {
            coverage: a.min(1.0),
            depth,
        };
    }
    miss
}

fn sample(layer: &Layer, uv: [f32; 2], threshold: f32) -> Sample {
    let miss = Sample {
        coverage: 0.0,
        depth: 1.0,
    };
    if !(uv.iter().all(|&c| (0.0..=1.0).contains(&c))) {
        return miss;
    }
    let texel = |c: f32| ((c * RESOLUTION as f32) as usize).min(RESOLUTION - 1);
    let index = texel(uv[1]) * RESOLUTION + texel(uv[0]);
    let (a, nd) = (layer.coverage[index], layer.normal_depth[index]);
    if a > 0.0 && a >= threshold && (0.0..=1.0).contains(&nd[3]) {
        return Sample {
            coverage: a.min(1.0),
            depth: nd[3],
        };
    }
    miss
}

fn probe(
    layer: &Layer,
    origin: [f32; 3],
    ray: [f32; 3],
    t: f32,
    side: f32,
    threshold: f32,
    linear: bool,
) -> Probe {
    let pr = project(&layer.clip, add(origin, ray, t));
    let s = if pr[3] > 0.5 && linear {
        sample_linear(layer, [pr[0], pr[1]], threshold)
    } else if pr[3] > 0.5 {
        sample(layer, [pr[0], pr[1]], threshold)
    } else {
        Sample {
            coverage: 0.0,
            depth: 1.0,
        }
    };
    let covered = s.coverage > 0.0;
    Probe {
        f: side * (pr[2] - s.depth),
        z: pr[2],
        s,
        covered,
    }
}

/// `foliage_parallax_hit` without refinement: SPOM (mode 0) with `threshold` the cutoff,
/// POM (mode 1) with 0, nearest or linear texels; (hit point, coverage, depth), coverage 0
/// on a miss.
fn parallax_march(
    layers: &[Layer],
    position: [f32; 3],
    ray: [f32; 3],
    threshold: f32,
    linear: bool,
) -> ([f32; 3], f32, f32) {
    let miss = (position, 0.0, 1.0);
    let (near, t1) = parallax_box(position, ray);
    let t0 = near.max(0.0);
    if (t1 - t0).is_nan() || t1 - t0 <= EPS {
        return miss;
    }
    let (mut index, mut best) = (0, -2.0);
    for (i, layer) in layers.iter().enumerate() {
        let score = dot3(forward(&layer.clip, position), ray);
        if score > best {
            best = score;
            index = i;
        }
    }
    if best.abs() <= 1e-3 {
        return miss;
    }
    let side = if best > 0.0 { 1.0 } else { -1.0 };
    let layer = &layers[index];
    let a = project(&layer.clip, add(position, ray, t0));
    let b = project(&layer.clip, add(position, ray, t1));
    let texels = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt() * RESOLUTION as f32;
    let steps = texels.ceil().clamp(8.0, 64.0) as u32;
    let first = probe(layer, position, ray, t0, side, threshold, linear);
    if first.covered && first.f >= 0.0 {
        return (add(position, ray, t0), first.s.coverage, first.z);
    }
    for i in 1..=steps {
        let t = t0 + (t1 - t0) * i as f32 / steps as f32;
        let cur = probe(layer, position, ray, t, side, threshold, linear);
        if cur.covered && cur.f >= 0.0 {
            return (add(position, ray, t), cur.s.coverage, cur.z);
        }
    }
    miss
}

/// SPOM (mode 0) with the nearest texel.
fn parallax_hit(layers: &[Layer], position: [f32; 3], ray: [f32; 3]) -> ([f32; 3], f32, f32) {
    parallax_march(layers, position, ray, CUTOFF, false)
}

#[test]
fn parallax_mirror_finds_the_card_from_a_hull_fragment() {
    let layers: Vec<Layer> = reference_set().into_iter().map(|(l, _)| l).collect();
    let world = root_world();
    // A game camera in front of the card, a little above and to the side, in world space.
    let eye_world = world.transform_point(Vec3::new(1.0, 4.0, 14.0));
    let target_world = world.transform_point(Vec3::new(0.2, 3.0, 0.0));
    let normal = world.inverse().unwrap().transpose();
    // transpose(normal matrix) of the fragment shader: the world ray in source-root space.
    let to_root = normal.transpose();
    let ray = to_root
        .transform_vector(target_world - eye_world)
        .normalize_or_zero();
    let ray = [ray.x, ray.y, ray.z];
    let eye = world.inverse().unwrap().transform_point(eye_world);
    let eye = [eye.x, eye.y, eye.z];
    // The hull fragment: where the camera ray enters the bounds the hull lies within.
    let (enter, _) = parallax_box(eye, ray);
    let position = add(eye, ray, enter + 1e-4);
    let (hit, coverage, depth) = parallax_hit(&layers, position, ray);
    assert!(
        coverage >= CUTOFF,
        "the hull fragment sees the card, coverage {coverage}"
    );
    assert!(depth > 0.0 && depth < 1.0, "hit depth {depth}");
    assert!(
        hit[2].abs() < 0.25,
        "the hit lies on the card plane z = 0 within a depth texel, got {hit:?}"
    );

    // A ray that passes beside the card misses.
    let beside = world.transform_point(Vec3::new(1.95, 0.2, 0.0));
    let ray = to_root
        .transform_vector(beside - eye_world)
        .normalize_or_zero();
    let ray = [ray.x, ray.y, ray.z];
    let (enter, _) = parallax_box(eye, ray);
    let (_, coverage, _) = parallax_hit(&layers, add(eye, ray, enter + 1e-4), ray);
    assert_eq!(coverage, 0.0, "beside the card is a miss");
}

#[test]
fn parallax_mirror_misses_through_empty_texels_beside_the_silhouette() {
    let world_to_root = world_to_root_normal(root_world()).unwrap();
    let cameras = reference_cameras(MIN, MAX, root_world(), VIEWS).unwrap();
    let mut probed = 0;
    for camera in &cameras {
        let (layer, _) = reference_layer(camera, &world_to_root);
        // The same layer with the earlier empty texel: depth 0 (the cleared depth target).
        let mut old = Layer {
            clip: layer.clip,
            coverage: layer.coverage.clone(),
            normal_depth: layer.normal_depth.clone(),
        };
        for (i, texel) in old.normal_depth.iter_mut().enumerate() {
            if old.coverage[i] == 0.0 {
                *texel = [0.0; 4];
            }
        }
        // A covered texel whose right neighbour has no reference fragment: the linear sample
        // halfway between them, at a ray depth just in front of the card.
        let edge = (0..RESOLUTION * RESOLUTION).find(|&i| {
            i % RESOLUTION + 1 < RESOLUTION
                && layer.coverage[i] > 0.0
                && layer.coverage[i + 1] == 0.0
        });
        let Some(i) = edge else { continue };
        probed += 1;
        let uv = [
            (i % RESOLUTION + 1) as f32 / RESOLUTION as f32,
            ((i / RESOLUTION) as f32 + 0.5) / RESOLUTION as f32,
        ];
        let z = layer.normal_depth[i][3] - 1e-4;
        let s = sample_linear(&layer, uv, 0.0);
        assert!(
            s.coverage > 0.0 && z - s.depth < 0.0,
            "a ray in front of the card misses beside its edge: coverage {}, depth {} at z {z}",
            s.coverage,
            s.depth
        );
        let s = sample_linear(&old, uv, 0.0);
        assert!(
            s.coverage > 0.0 && z - s.depth >= 0.0,
            "the empty depth 0 blends a near depth into the edge, so the same ray hit there"
        );
    }
    assert!(probed > 0, "no view shows a card edge");

    // A POM march (threshold 0) on linear texels through empty texels beside the card misses.
    let layers: Vec<Layer> = cameras
        .iter()
        .map(|camera| reference_layer(camera, &world_to_root).0)
        .collect();
    let world = root_world();
    let eye_world = world.transform_point(Vec3::new(1.0, 4.0, 14.0));
    let to_root = world.inverse().unwrap();
    let eye = to_root.transform_point(eye_world);
    let eye = [eye.x, eye.y, eye.z];
    let beside = world.transform_point(Vec3::new(1.95, 0.2, 0.0));
    let ray = to_root
        .transform_vector(beside - eye_world)
        .normalize_or_zero();
    let ray = [ray.x, ray.y, ray.z];
    let (enter, _) = parallax_box(eye, ray);
    let (_, coverage, depth) = parallax_march(&layers, add(eye, ray, enter + 1e-4), ray, 0.0, true);
    assert_eq!(
        (coverage, depth),
        (0.0, 1.0),
        "beside the card the POM march misses"
    );
}
