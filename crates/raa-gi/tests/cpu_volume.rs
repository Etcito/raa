//! Hand-derived fixtures for the CPU twins of share 1 (S37): triangle coverage, `clip_span`
//! and the emissive, radiance and octahedral codecs. Every expected value is derived by hand
//! from the widened-edge rule and the WGSL pack/unpack definitions, never from a GPU run.

use raa_gi::cpu::{
    clip_span, decode_emissive, decode_radiance, encode_emissive, encode_radiance, oct_decode,
    oct_encode, voxelize_triangle,
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
