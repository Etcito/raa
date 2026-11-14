//! Hand-derived fixtures for the CPU twins of share 1 (S37): triangle coverage, `clip_span`
//! and the emissive, radiance and octahedral codecs. Every expected value is derived by hand
//! from the widened-edge rule and the WGSL pack/unpack definitions, never from a GPU run.

use raa_gi::cpu::{
    CpuGrid, CpuHit, CpuVolume, clip_span, decode_emissive, decode_radiance, encode_emissive,
    encode_radiance, jump_flood, oct_decode, oct_encode, seed_distance, voxelize_triangle,
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
