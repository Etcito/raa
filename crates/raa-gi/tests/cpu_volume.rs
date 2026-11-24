//! Hand-derived fixtures for the CPU twins of share 1 (S37): triangle coverage, `clip_span`
//! and the emissive, radiance and octahedral codecs. Every expected value is derived by hand
//! from the widened-edge rule and the WGSL pack/unpack definitions, never from a GPU run.

use raa_gi::cpu::{
    CpuGrid, CpuHit, CpuVolume, clip_span, cone_smooth, decode_emissive, decode_radiance,
    encode_emissive, encode_radiance, jump_flood, oct_decode, oct_encode, ray_origin,
    seed_distance, voxelize_triangle,
};
use raa_loader::json::{JsonValue, parse};

const EMISSIVE_RANGE: f32 = 128.0;
const RADIANCE_RANGE: f32 = 512.0;

fn fixture(name: &str) -> JsonValue {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    parse(&text).unwrap_or_else(|e| panic!("{path}: {e}"))
}

fn field<'a>(value: &'a JsonValue, key: &str) -> &'a JsonValue {
    let JsonValue::Object(entries) = value else {
        panic!("expected an object holding {key}");
    };
    entries
        .iter()
        .find_map(|(name, value)| (name == key).then_some(value))
        .unwrap_or_else(|| panic!("missing key {key}"))
}

fn array(value: &JsonValue) -> &[JsonValue] {
    let JsonValue::Array(items) = value else {
        panic!("expected an array");
    };
    items
}

fn number(value: &JsonValue) -> f64 {
    let JsonValue::Number(n) = value else {
        panic!("expected a number");
    };
    *n
}

fn string(value: &JsonValue) -> &str {
    let JsonValue::String(text) = value else {
        panic!("expected a string");
    };
    text
}

fn floats<const N: usize>(value: &JsonValue) -> [f32; N] {
    let items = array(value);
    assert_eq!(items.len(), N, "array length");
    std::array::from_fn(|i| number(&items[i]) as f32)
}

fn words(value: &JsonValue) -> [u32; 2] {
    let items = array(value);
    assert_eq!(items.len(), 2, "word count");
    [number(&items[0]) as u32, number(&items[1]) as u32]
}

fn cell(value: &JsonValue) -> [i32; 3] {
    floats::<3>(value).map(|x| x as i32)
}

fn cells(value: &JsonValue) -> Vec<[i32; 3]> {
    array(value).iter().map(cell).collect()
}

fn triangle(value: &JsonValue) -> [[f32; 3]; 3] {
    let items = array(value);
    assert_eq!(items.len(), 3, "triangle vertices");
    [
        floats::<3>(&items[0]),
        floats::<3>(&items[1]),
        floats::<3>(&items[2]),
    ]
}

fn covered(case: &JsonValue) -> Vec<[i32; 3]> {
    let resolution = number(field(case, "resolution")) as u32;
    let mut out = Vec::new();
    let mut covered = Vec::new();
    for value in array(field(case, "triangles")) {
        voxelize_triangle(triangle(value), resolution, &mut out);
        covered.extend_from_slice(&out);
    }
    covered.sort_unstable();
    covered.dedup();
    covered
}

fn check_covered(case: &JsonValue) {
    assert_eq!(
        covered(case),
        cells(field(case, "expected")),
        "{}",
        string(field(case, "why"))
    );
}

#[test]
fn voxelize_quad() {
    let root = fixture("voxelize.json");
    check_covered(field(&root, "quad"));
}

#[test]
fn voxelize_sliver() {
    let root = fixture("voxelize.json");
    check_covered(field(&root, "sliver"));
}

#[test]
fn voxelize_outside() {
    let root = fixture("voxelize.json");
    check_covered(field(&root, "outside"));
}

#[test]
fn clip_span_cases() {
    let root = fixture("voxelize.json");
    for case in array(field(&root, "clip_span")) {
        let span = floats::<2>(field(case, "span"));
        let slope = number(field(case, "slope")) as f32;
        let offset = number(field(case, "offset")) as f32;
        assert_eq!(
            clip_span(span, slope, offset),
            floats::<2>(field(case, "expected")),
            "{}",
            string(field(case, "why"))
        );
    }
}

fn emissive_step(color: [f32; 3]) -> f32 {
    let peak = (color[0].max(color[1]).max(color[2]) / EMISSIVE_RANGE).clamp(1.0e-6, 1.0);
    let scale = (peak * 255.0).ceil() / 255.0;
    EMISSIVE_RANGE * scale / 255.0
}

fn radiance_bound(component: f32) -> f32 {
    let e = (component / RADIANCE_RANGE).clamp(0.0, 1.0).sqrt();
    RADIANCE_RANGE * (0.5 / 65535.0) * (2.0 * e + 0.5 / 65535.0)
}

#[test]
fn codec_round_trips() {
    let root = fixture("codecs.json");

    let emissive = field(&root, "emissive");
    for case in array(field(emissive, "round_trips")) {
        let color = floats::<3>(field(case, "color"));
        let decoded = decode_emissive(encode_emissive(color));
        let bound = emissive_step(color) + 1.0e-6;
        for i in 0..3 {
            assert!(
                (decoded[i] - color[i]).abs() <= bound,
                "emissive {color:?} -> {decoded:?}: {}",
                string(field(case, "why"))
            );
        }
    }
    for case in array(field(emissive, "packed")) {
        let color = floats::<3>(field(case, "color"));
        assert_eq!(
            encode_emissive(color),
            number(field(case, "word")) as u32,
            "emissive word {color:?}: {}",
            string(field(case, "why"))
        );
    }

    let radiance = field(&root, "radiance");
    for case in array(field(radiance, "round_trips")) {
        let color = floats::<3>(field(case, "color"));
        let decoded = decode_radiance(encode_radiance(color));
        for i in 0..3 {
            let bound = radiance_bound(color[i]) + 1.0e-6;
            assert!(
                (decoded[i] - color[i]).abs() <= bound,
                "radiance {color:?} -> {decoded:?}: {}",
                string(field(case, "why"))
            );
        }
    }
    for case in array(field(radiance, "exact")) {
        let color = floats::<3>(field(case, "color"));
        assert_eq!(
            encode_radiance(color),
            words(field(case, "words")),
            "radiance words {color:?}: {}",
            string(field(case, "why"))
        );
    }

    let oct = field(&root, "oct");
    for case in array(field(oct, "cases")) {
        let normal = floats::<3>(field(case, "normal"));
        let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        let decoded = oct_decode(oct_encode(normal));
        for i in 0..3 {
            assert!(
                (decoded[i] - normal[i] / length).abs() <= 1.0e-3,
                "oct {normal:?} -> {decoded:?}: {}",
                string(field(case, "why"))
            );
        }
    }
}

fn index_of(cell: [i32; 3], r: i32) -> usize {
    (cell[0] + r * (cell[1] + r * cell[2])) as usize
}

#[test]
fn jump_flood_two_seeds() {
    let root = fixture("flood.json");
    let r = number(field(&root, "resolution")) as i32;
    let mut occupied = vec![false; (r * r * r) as usize];
    for seed in cells(field(&root, "seeds")) {
        occupied[index_of(seed, r)] = true;
    }
    let grid = CpuGrid {
        resolution: r as u32,
        occupied,
    };
    let steps: Vec<i32> = array(field(&root, "steps"))
        .iter()
        .map(|v| number(v) as i32)
        .collect();
    let nearest = jump_flood(&grid, &steps);
    for case in array(field(&root, "cells")) {
        let at = cell(field(case, "cell"));
        assert_eq!(
            nearest[index_of(at, r)],
            Some(cell(field(case, "seed"))),
            "{at:?}: {}",
            string(field(case, "why"))
        );
    }
}

fn check_metres(case: &JsonValue, got: f32) {
    let want = number(field(case, "metres")) as f32;
    assert!(
        (got - want).abs() <= 1.0e-4,
        "{:?}: {got} m, want {want} m: {}",
        cell(field(case, "cell")),
        string(field(case, "why"))
    );
}

#[test]
fn seed_distances() {
    let root = fixture("flood.json");
    let voxel = number(field(&root, "voxel")) as f32;
    for case in array(field(&root, "cells")) {
        let seed = Some(cell(field(case, "seed")));
        check_metres(case, seed_distance(cell(field(case, "cell")), seed, voxel));
    }
    let near = field(&root, "seed_distance");
    let seed = Some(cell(field(near, "seed")));
    check_metres(near, seed_distance(cell(field(near, "cell")), seed, voxel));
    let far = field(&root, "unseeded");
    check_metres(far, seed_distance(cell(field(far, "cell")), None, voxel));
}

/// One level of R cells at `origin`, filled from the plane `z = plane_z`: every texel holds
/// `a = |z_centre - plane_z|` and the band colour where `a` is within `band_voxels` voxels.
fn plane_volume(root: &JsonValue) -> CpuVolume {
    let r = number(field(root, "resolution")) as usize;
    let origin = floats::<3>(field(root, "origin"));
    let voxel = number(field(root, "voxel")) as f32;
    let plane = number(field(root, "plane_z")) as f32;
    let band = number(field(root, "band_voxels")) as f32 * voxel;
    let colour = floats::<3>(field(root, "colour"));
    let mut texels = Vec::with_capacity(r * r * r);
    for z in 0..r {
        let centre = origin[2] + (z as f32 + 0.5) * voxel;
        let a = (centre - plane).abs();
        let rgb = if a <= band { colour } else { [0.0; 3] };
        for _ in 0..r * r {
            texels.push([rgb[0], rgb[1], rgb[2], a]);
        }
    }
    CpuVolume {
        levels: vec![[origin[0], origin[1], origin[2], voxel]],
        resolution: r as u32,
        texels,
    }
}

fn cast(volume: &CpuVolume, ray: &JsonValue, hit_voxels: f32) -> CpuHit {
    volume.trace(
        floats::<3>(field(ray, "origin")),
        floats::<3>(field(ray, "dir")),
        number(field(ray, "start")) as f32,
        number(field(ray, "max_t")) as f32,
        number(field(ray, "max_steps")) as u32,
        hit_voxels,
    )
}

fn ray_visibility(volume: &CpuVolume, ray: &JsonValue) -> f32 {
    volume.visibility(
        floats::<3>(field(ray, "origin")),
        floats::<3>(field(ray, "dir")),
        number(field(ray, "start")) as f32,
        number(field(ray, "max_t")) as f32,
        number(field(ray, "max_steps")) as u32,
    )
}

#[test]
fn trace_against_plane() {
    let root = fixture("trace.json");
    let volume = plane_volume(&root);
    let hit_voxels = number(field(&root, "hit_voxels")) as f32;
    for ray in array(field(&root, "rays")) {
        let name = string(field(ray, "name"));
        let got = cast(&volume, ray, hit_voxels);
        let JsonValue::Bool(hit) = field(ray, "hit") else {
            panic!("{name}: hit must be a boolean");
        };
        assert_eq!(got.hit, *hit, "{name}");
        if !*hit {
            continue;
        }
        let (t_min, t_max) = (
            number(field(ray, "t_min")) as f32,
            number(field(ray, "t_max")) as f32,
        );
        assert!(got.t >= t_min && got.t < t_max, "{name}: t {}", got.t);
        let radiance = floats::<3>(field(ray, "radiance"));
        for (g, w) in got.radiance.iter().zip(radiance) {
            assert!((g - w).abs() <= 1.0e-3, "{name}: {:?}", got.radiance);
        }
    }
}

#[test]
fn visibility_and_level_lookup() {
    let root = fixture("trace.json");
    let volume = plane_volume(&root);
    let rays = array(field(&root, "rays"));
    for ray in rays {
        let want = number(field(ray, "visibility")) as f32;
        assert_eq!(
            ray_visibility(&volume, ray),
            want,
            "{}",
            string(field(ray, "name"))
        );
    }
    let empty = CpuVolume {
        levels: Vec::new(),
        resolution: volume.resolution,
        texels: Vec::new(),
    };
    assert_eq!(ray_visibility(&empty, &rays[0]), 1.0);

    let lookup = field(&root, "lookup");
    let inside = floats::<3>(field(lookup, "inside"));
    assert_eq!(
        volume.level_at(inside, 0),
        number(field(lookup, "inside_level")) as usize
    );
    let edge = floats::<3>(field(lookup, "edge"));
    assert_eq!(
        volume.level_at(edge, 0),
        number(field(lookup, "edge_level")) as usize
    );
    assert_eq!(volume.level_at(edge, 0), volume.levels.len());
}

#[test]
fn cone_visibility_against_plane() {
    let root = fixture("cone.json");
    let volume = plane_volume(&root);
    let empty = CpuVolume {
        levels: Vec::new(),
        resolution: volume.resolution,
        texels: Vec::new(),
    };
    for case in array(field(&root, "cases")) {
        let name = string(field(case, "name"));
        let JsonValue::Bool(is_empty) = field(case, "empty") else {
            panic!("{name}: empty must be a boolean");
        };
        let subject = if *is_empty { &empty } else { &volume };
        let got = subject.cone_visibility(
            floats::<3>(field(case, "origin")),
            floats::<3>(field(case, "dir")),
            number(field(case, "start")) as f32,
            number(field(case, "max_t")) as f32,
            number(field(case, "max_steps")) as u32,
            number(field(case, "tan_half")) as f32,
        );
        let want = number(field(case, "visibility")) as f32;
        assert!(
            (got - want).abs() <= 1.0e-4,
            "{name}: got {got}, want {want}"
        );
    }
}

/// The cone under the revised rule, walked by hand on the `cone.json` plane (h = |z - 2.125|,
/// voxel 0.25 m, so dd = h - 0.125 m and the radius floor is 0.125 m). A near miss: the one
/// sample at t 0.25 (z 2.025, h = 0.1 = 0.4 voxel) is the first of its level, so approaching;
/// y = 0, dd = 0.1 - 0.125 = -0.025, radius = max(0.5 x 0.25, 0.125) = 0.125, res = -0.2. The
/// step clamp(0.1, 0.125, 1.0) takes t to 0.375 > max_t 0.3, and 0.25 x (0.8)^2 x (2.2) = 0.352.
/// On the plane at t 0.25 (z 2.125, h = 0): dd = -0.125, radius = 0.125, res = -1, so 0.
#[test]
fn cone_visibility_near_miss_stays_soft() {
    let volume = plane_volume(&fixture("cone.json"));
    let near = volume.cone_visibility([2.0, 2.0, 1.775], [0.0, 0.0, 1.0], 0.25, 0.3, 32, 0.5);
    assert!((near - 0.352).abs() <= 1.0e-4, "near miss: got {near}");
    let touch = volume.cone_visibility([2.0, 2.0, 1.875], [0.0, 0.0, 1.0], 0.25, 0.3, 32, 0.5);
    assert_eq!(touch, 0.0);
}

/// `cone_smooth` is Quilez's 0.25 (1 + r)^2 (2 - r) for r = clamp(res, -1, 1), derived by hand:
/// 1 -> 0.25 x 4 x 1 = 1; 0.5 -> 0.25 x 2.25 x 1.5 = 0.84375; 0 -> 0.25 x 1 x 2 = 0.5;
/// -0.5 -> 0.25 x 0.25 x 2.5 = 0.15625; -1 -> 0 and -3 clamps to -1 -> 0.
#[test]
fn cone_smooth_values() {
    for (res, want) in [
        (1.0, 1.0),
        (0.5, 0.84375),
        (0.0, 0.5),
        (-0.5, 0.15625),
        (-1.0, 0.0),
        (-3.0, 0.0),
    ] {
        let got = cone_smooth(res);
        assert!(
            (got - want).abs() <= 1.0e-6,
            "cone_smooth({res}) = {got}, want {want}"
        );
    }
}

fn assert_close3(got: [f32; 3], want: [f32; 3]) {
    for i in 0..3 {
        assert!((got[i] - want[i]).abs() <= 1.0e-5, "{got:?}, want {want:?}");
    }
}

/// `ray_origin` with normal (0, 1, 0), push 0.6 and max_along 1.6, derived by hand. dir
/// (0, 1, 0): n_dot_l = 1, along = min(0.6 / 1, 1.6) = 0.6, rest = 0.6 - 0.6 x 1 = 0, so the
/// origin is world + (0, 0.6, 0). dir (1, 0, 0): n_dot_l = 0, along = 0, rest = 0.6, so the
/// origin is world + (0, 0.6, 0). dir (cos 30, sin 30, 0): n_dot_l = 0.5, along =
/// min(0.6 / 0.5, 1.6) = 1.2, rest = 0.6 - 1.2 x 0.5 = 0, so the origin is world + 1.2 dir.
#[test]
fn ray_origin_clears_the_surface_along_the_ray() {
    let world = [1.0, 2.0, 3.0];
    let normal = [0.0, 1.0, 0.0];

    let (origin, along) = ray_origin(world, normal, [0.0, 1.0, 0.0], 0.6, 1.6);
    assert_close3(origin, [1.0, 2.6, 3.0]);
    assert!((along - 0.6).abs() <= 1.0e-6, "along {along}");

    let (origin, along) = ray_origin(world, normal, [1.0, 0.0, 0.0], 0.6, 1.6);
    assert_close3(origin, [1.0, 2.6, 3.0]);
    assert_eq!(along, 0.0);

    let cos30 = 3.0_f32.sqrt() * 0.5;
    let dir = [cos30, 0.5, 0.0];
    let (origin, along) = ray_origin(world, normal, dir, 0.6, 1.6);
    assert_close3(
        origin,
        [
            world[0] + 1.2 * dir[0],
            world[1] + 1.2 * dir[1],
            world[2] + 1.2 * dir[2],
        ],
    );
    assert!((along - 1.2).abs() <= 1.0e-6, "along {along}");
}

/// The revised cone on the `cone.json` plane, walked by hand (voxel 0.25 m, dd = h - 0.125 m,
/// radius = max(tan_half (t - y), 0.125 m)). A parallel ray 1.25 m from the plane with start
/// 2.0: y = 0, dd = 1.125, radius = max(0.5 x 2.0, 0.125) = 1.0, res = 1 (1.125 >= 1), so 1.0.
/// A ray from z 1.75 along +z with start 0.25, tan_half 0.25: sample 1 (z 2.0, h 0.125) has
/// dd = 0, radius = max(0.25 x 0.25, 0.125) = 0.125, res = 0; sample 2 (t 0.375, z 2.125, h 0)
/// has dd = -0.125, radius = max(0.25 x 0.375, 0.125) = 0.125, res = -1, so 0.0. A sweep of
/// parallel rays (origin y 0, start 2.0, tan_half 0.5) whose offsets from the plane step down
/// by a quarter voxel, h = 0.25, 0.1875, 0.125, 0.0625 and 0 m: the first sample has y = 0,
/// dd = h - 0.125 and radius = max(0.5 x 2.0, 0.125) = 1.0, so res = h - 0.125 and the values
/// are 0.593262, 0.546814, 0.5, 0.453186 and 0.406738 (hand-evaluated 0.25 (1 + r)^2 (2 - r));
/// each step down is at most 0.047 < 0.25 and never upward.
#[test]
fn cone_visibility_soft_rule_on_plane() {
    let volume = plane_volume(&fixture("cone.json"));

    let parallel = volume.cone_visibility([2.0, 0.5, 0.875], [0.0, 1.0, 0.0], 2.0, 4.0, 32, 0.5);
    assert!((parallel - 1.0).abs() <= 1.0e-4, "parallel: {parallel}");

    let crossing = volume.cone_visibility([2.0, 2.0, 1.75], [0.0, 0.0, 1.0], 0.25, 1.0, 32, 0.25);
    assert!((crossing - 0.0).abs() <= 1.0e-4, "crossing: {crossing}");

    let mut previous: Option<f32> = None;
    for (offset, want) in [
        (0.25, 0.593262),
        (0.1875, 0.546814),
        (0.125, 0.5),
        (0.0625, 0.453186),
        (0.0, 0.406738),
    ] {
        let got = volume.cone_visibility(
            [2.0, 0.0, 2.125 - offset],
            [0.0, 1.0, 0.0],
            2.0,
            4.0,
            32,
            0.5,
        );
        assert!(
            (got - want).abs() <= 1.0e-4,
            "offset {offset}: got {got}, want {want}"
        );
        if let Some(previous) = previous {
            assert!(
                got <= previous + 1.0e-6,
                "offset {offset}: {got} rose above {previous}"
            );
            assert!(
                (previous - got).abs() <= 0.25,
                "offset {offset}: jumped from {previous} to {got}"
            );
        }
        previous = Some(got);
    }
}

/// `gather` on the `trace.json` plane (z 2.0 m, voxel 0.25 m, texel a = |z_centre - 2.0| with
/// centres at 0.25 k + 0.125 m, colour (1, 0.5, 0.25) where a <= 3 voxels), walked by hand under
/// the trace rule (step max(d - 0.125, 0.15), hit d < 0.2125 while approaching) and the gather
/// band d <= 3 voxels = 0.75 m at the point where the trace stopped.
/// - Toward the plane from z 0.5 along +z, start 0.25, max_steps 1: the one sample (t 0.25,
///   z 0.75, d = mix(1.375, 1.125, 0.5) = 1.25) is the first of its level, so not approaching,
///   and steps 1.125 to t 1.375, where the budget ends: `trace` misses at t 1.375. z 1.875 is
///   the texel centre k = 7 (local 7.5 voxels, inside the level) with d 0.125 <= 0.75 and the
///   band colour, so `gather` hits at t 1.375 with (1, 0.5, 0.25).
/// - Parallel to the plane at z 0.75 from x 0.5 along +x, max_steps 1: the same d 1.25 and step
///   stop it at t 1.375 (x 1.875, inside) 5 voxels from the plane, so `gather` misses.
/// - Parallel at z 1.5 from x 3.0 along +x, max_steps 64: the sample at x 3.25 (local 13,
///   d = mix(0.625, 0.375, 0.5) = 0.5, not approaching) steps 0.375 to t 0.625, x 3.625 =
///   local 14.5, outside (1.5, 14.5), so the ray leaves the volume; d 0.5 there lies inside the
///   band, yet `gather` misses at t 0.625.
/// - Toward the plane with max_steps 64: the second sample (t 1.375, d 0.125 <= 1.25 and
///   < 0.2125) is a trace hit, and `gather` returns it unchanged.
#[test]
fn gather_counts_exhausted_rays_near_a_surface_as_hits() {
    let volume = plane_volume(&fixture("trace.json"));
    let colour = [1.0, 0.5, 0.25];
    let (origin, dir) = ([2.0, 2.0, 0.5], [0.0, 0.0, 1.0]);

    let traced = volume.trace(origin, dir, 0.25, 10.0, 1, 0.85);
    assert!(
        !traced.hit && !traced.escaped,
        "the step budget ends inside the volume"
    );
    assert_eq!(traced.t, 1.375);
    let gathered = volume.gather(origin, dir, 0.25, 10.0, 1, 0.85);
    assert!(
        gathered.hit,
        "an exhausted ray beside the plane takes its radiance"
    );
    assert_eq!(gathered.t, traced.t);
    assert_close3(gathered.radiance, colour);

    let far = volume.gather([0.5, 2.0, 0.75], [1.0, 0.0, 0.0], 0.25, 10.0, 1, 0.85);
    assert!(!far.hit && !far.escaped, "stopped 5 voxels from the plane");
    assert_eq!(far.t, 1.375);

    let leaving = volume.gather([3.0, 2.0, 1.5], [1.0, 0.0, 0.0], 0.25, 10.0, 64, 0.85);
    assert!(
        !leaving.hit && leaving.escaped,
        "left the volume inside the band"
    );
    let last_step = volume.trace([3.0, 2.0, 1.5], [1.0, 0.0, 0.0], 0.25, 10.0, 1, 0.85);
    assert!(last_step.escaped, "the final step reached the clipmap edge");
    let distance_limit = volume.trace([3.0, 2.0, 1.5], [1.0, 0.0, 0.0], 0.25, 0.5, 1, 0.85);
    assert!(
        !distance_limit.escaped,
        "the distance limit is inside the clipmap"
    );
    assert_eq!(leaving.t, 0.625);

    let traced = volume.trace(origin, dir, 0.25, 10.0, 64, 0.85);
    let gathered = volume.gather(origin, dir, 0.25, 10.0, 64, 0.85);
    assert!(traced.hit && gathered.hit, "a normal hit");
    assert_eq!(gathered.t, traced.t);
    assert_eq!(gathered.radiance, traced.radiance);
    assert_close3(gathered.radiance, colour);
}
