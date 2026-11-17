//! P5-Q1: the exposure compute module validates with naga, defines exactly the two compute
//! entries, binds group 0 bindings 0..3 only, and its structs match `PARAMS_SIZE` and
//! `RESULT_SIZE`; `pack_params` and `read_result` follow the S46 layouts.

use raa_exposure::{
    AutoExposureDesc, BUILD_HISTOGRAM, EXPOSURE_COMPUTE_WGSL, ExposureResult, PARAMS_SIZE, RESOLVE,
    RESULT_SIZE, pack_params, read_result,
};
use raa_shader::Composer;

/// Lines of `wgsl` without `//` comments.
fn code_lines(wgsl: &str) -> impl Iterator<Item = &str> {
    wgsl.lines()
        .map(|l| l.split("//").next().unwrap_or("").trim())
        .filter(|l| !l.is_empty())
}

/// `(stage attribute, fn name)` of every entry point.
fn entry_points(wgsl: &str) -> Vec<(&str, &str)> {
    let lines: Vec<&str> = code_lines(wgsl).collect();
    lines
        .windows(2)
        .filter(|w| matches!(w[0], "@vertex" | "@fragment") || w[0].starts_with("@compute"))
        .filter_map(|w| {
            let name = w[1].strip_prefix("fn ")?.split('(').next()?;
            Some((w[0], name))
        })
        .collect()
}

/// `(name, type)` of the fields of `struct <name>`.
fn fields<'a>(wgsl: &'a str, name: &str) -> Vec<(&'a str, &'a str)> {
    let head = format!("struct {name}");
    code_lines(wgsl)
        .skip_while(|l| !l.starts_with(&head))
        .skip(1)
        .take_while(|l| !l.starts_with('}'))
        .filter_map(|l| l.split_once(':'))
        .map(|(n, t)| (n.trim(), t.trim().trim_end_matches(',')))
        .collect()
}

fn word(bytes: &[u8], at: usize) -> [u8; 4] {
    let mut w = [0; 4];
    w.copy_from_slice(&bytes[at..at + 4]);
    w
}

#[test]
fn compute_module_validates() {
    if let Err(e) = Composer::validate(EXPOSURE_COMPUTE_WGSL) {
        panic!("exposure compute: {e}");
    }
}

#[test]
fn two_compute_entries() {
    let entries = entry_points(EXPOSURE_COMPUTE_WGSL);
    assert_eq!(
        entries,
        [
            ("@compute @workgroup_size(16, 16)", BUILD_HISTOGRAM),
            ("@compute @workgroup_size(1)", RESOLVE),
        ]
    );
    assert_eq!(EXPOSURE_COMPUTE_WGSL.matches("@compute").count(), 2);
    assert_eq!(EXPOSURE_COMPUTE_WGSL.matches("@vertex").count(), 0);
    assert_eq!(EXPOSURE_COMPUTE_WGSL.matches("@fragment").count(), 0);
}

#[test]
fn only_group_zero_and_no_prelude() {
    let mut bindings = Vec::new();
    for l in code_lines(EXPOSURE_COMPUTE_WGSL) {
        assert!(!l.contains("raa_"), "PRELUDE identifier: {l}");
        for (i, _) in l.match_indices("@group(") {
            let rest = &l[i..];
            assert!(rest.starts_with("@group(0) @binding("), "{l}");
            let binding = rest["@group(0) @binding(".len()..]
                .split(')')
                .next()
                .and_then(|b| b.parse::<u32>().ok())
                .unwrap_or_else(|| panic!("binding index: {l}"));
            bindings.push(binding);
        }
    }
    assert_eq!(bindings, [0, 1, 2, 3]);
}

#[test]
fn struct_field_order_matches_sizes() {
    let params = [
        ("dt_s", "f32"),
        ("first", "u32"),
        ("compensation", "f32"),
        ("low_percent", "f32"),
        ("high_percent", "f32"),
        ("speed_up", "f32"),
        ("speed_down", "f32"),
        ("min_exposure", "f32"),
        ("max_exposure", "f32"),
        ("frame", "u32"),
        ("_pad0", "u32"),
        ("_pad1", "u32"),
    ];
    assert_eq!(fields(EXPOSURE_COMPUTE_WGSL, "ExposureParams"), params);
    assert_eq!(params.len() * 4, PARAMS_SIZE);
    let state = [
        ("average_luminance", "f32"),
        ("exposure", "f32"),
        ("valid", "u32"),
        ("frame", "u32"),
    ];
    assert_eq!(fields(EXPOSURE_COMPUTE_WGSL, "ExposureState"), state);
    assert_eq!(state.len() * 4, RESULT_SIZE);
}

#[test]
fn attribution_line_present() {
    let first = EXPOSURE_COMPUTE_WGSL.lines().next().unwrap_or("");
    assert_eq!(
        first,
        "// Adapted from solstice@7e465b5 crates/solstice-render/shaders/post/exposure.wgsl, \
         MIT licence, Copyright (c) 2026 SamG-Coder"
    );
}

#[test]
fn pack_params_layout() {
    let mut out = [0xffu8; PARAMS_SIZE];
    pack_params(
        &mut out,
        &AutoExposureDesc::default(),
        0.016,
        true,
        0x1_0000_0005,
    );
    let f = |at| f32::from_le_bytes(word(&out, at));
    assert_eq!(f(0), 0.016);
    assert_eq!(u32::from_le_bytes(word(&out, 4)), 1);
    assert_eq!(f(8), 0.263f32.exp2());
    assert_eq!(f(12), 0.2);
    assert_eq!(f(16), 0.7);
    assert_eq!(f(20), 2.4);
    assert_eq!(f(24), 1.1);
    assert_eq!(f(28), 0.015625);
    assert_eq!(f(32), 64.0);
    assert_eq!(u32::from_le_bytes(word(&out, 36)), 5);
    assert_eq!(out[40..48], [0; 8]);
}

#[test]
fn read_result_layout() {
    let mut bytes = [0u8; RESULT_SIZE];
    bytes[0..4].copy_from_slice(&0.5f32.to_le_bytes());
    bytes[4..8].copy_from_slice(&2.0f32.to_le_bytes());
    bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
    bytes[12..16].copy_from_slice(&7u32.to_le_bytes());
    let want = ExposureResult {
        average_luminance: 0.5,
        exposure: 2.0,
        valid: true,
        frame: 7,
    };
    assert_eq!(read_result(&bytes), want);
    bytes[8..12].copy_from_slice(&0u32.to_le_bytes());
    assert_eq!(
        read_result(&bytes),
        ExposureResult {
            valid: false,
            ..want
        }
    );
}
