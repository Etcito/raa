//! Hand-derived fixtures for the CPU twins of the M6 area-shadow kernels (S44, contract
//! `area-stochastic-cpu`): the stratum and R2 jitter, the rectangle point, the sample weight, the
//! temporal and depth tests, the bilateral weight, the ratio estimator and the slot tables. The
//! expected values live in `tests/fixtures/area_stochastic.json`, derived by hand; nothing is
//! taken from running the code under test. Two analytic properties of the hashed functions (a
//! stratum cycle and the R2 step) have no hand value and are checked directly.

use std::collections::BTreeSet;

use raa_gi::parts::area_shadow::cpu::{
    R2_ALPHA, bilateral_weight, cell_side, depth_consistent, groups_for, history_slot, mask_slot,
    r2_jitter, ratio_visibility, raw_layers, ray_origin, ray_view_depth, rect_point, sample_weight,
    stratum_cell, stratum_uv, temporal_blend,
};
use raa_gi::parts::area_shadow::restir_cpu::{
    REMAP_INVALID, RESTIR_SPATIAL_RADIUS, Reservoir, Surface, area_pdf, balance_weight,
    boiling_keep, clamp_confidence, contribution_weight, depth_similar, luminance, oct_decode,
    oct_encode, pack_reservoir, pack2x16float, rect_solid_angle, remap_slot, remap_table,
    sample_rect, spatial_offset, srgb_decode, target_luminance, unpack_reservoir, unpack2x16float,
    wrs_accept,
};
use raa_loader::json::{JsonValue, parse};

fn fixture() -> JsonValue {
    let path = format!(
        "{}/tests/fixtures/area_stochastic.json",
        env!("CARGO_MANIFEST_DIR")
    );
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

fn items(value: &JsonValue) -> &[JsonValue] {
    let JsonValue::Array(items) = value else {
        panic!("expected an array");
    };
    items
}

fn number(value: &JsonValue) -> f32 {
    let JsonValue::Number(n) = value else {
        panic!("expected a number");
    };
    *n as f32
}

fn floats<const N: usize>(value: &JsonValue) -> [f32; N] {
    let list = items(value);
    assert_eq!(list.len(), N, "expected {N} numbers");
    std::array::from_fn(|i| number(&list[i]))
}

fn rows<const N: usize>(value: &JsonValue) -> Vec<[f32; N]> {
    items(value).iter().map(floats::<N>).collect()
}

fn restir_fixture() -> JsonValue {
    let path = format!(
        "{}/tests/fixtures/area_restir.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    parse(&text).unwrap_or_else(|e| panic!("{path}: {e}"))
}

fn word(value: &JsonValue) -> u32 {
    let JsonValue::Number(n) = value else {
        panic!("expected a number");
    };
    *n as u32
}

/// A JSON number, or the string `"nan"` for a NaN input (JSON has no NaN literal).
fn text(value: &JsonValue) -> &str {
    let JsonValue::String(s) = value else {
        panic!("expected a string");
    };
    s
}

fn boolean(value: &JsonValue) -> bool {
    let JsonValue::Bool(flag) = value else {
        panic!("expected a boolean");
    };
    *flag
}

/// A JSON number, or null, as an optional u32 (the remap fixture's `mapped` lane).
fn optional_word(value: &JsonValue) -> Option<u32> {
    match value {
        JsonValue::Null => None,
        _ => Some(word(value)),
    }
}

fn scalar(value: &JsonValue) -> f32 {
    match value {
        JsonValue::String(s) if s == "nan" => f32::NAN,
        _ => number(value),
    }
}

/// Relative tolerance, with an absolute floor of `tol` for expected values below 1.
fn close(got: f32, want: f32, tol: f32) -> bool {
    (got - want).abs() <= tol * want.abs().max(1.0)
}

fn assert_close<const N: usize>(got: [f32; N], want: [f32; N], tol: f32, what: &str) {
    for (g, w) in got.into_iter().zip(want) {
        assert!(close(g, w, tol), "{what}: got {got:?}, want {want:?}");
    }
}

fn pair(row: [f32; 2]) -> (u32, u32) {
    (row[0] as u32, row[1] as u32)
}

#[test]
fn stratum_uv_and_rect_point_tables() {
    let root = fixture();
    let tol = number(field(&root, "tolerance"));
    for [cells, cell, jx, jy, u, v] in rows::<6>(field(&root, "stratum_uv")) {
        let got = stratum_uv(cells as u32, cell as u32, [jx, jy]);
        assert_close(got, [u, v], tol, "stratum_uv");
    }
    for row in rows::<16>(field(&root, "rect_point")) {
        let got = rect_point(
            [row[0], row[1], row[2]],
            [row[3], row[4], row[5]],
            [row[6], row[7], row[8]],
            [row[9], row[10]],
            [row[11], row[12]],
        );
        assert_close(got, [row[13], row[14], row[15]], tol, "rect_point");
    }
}

#[test]
fn sample_weight_table() {
    let root = fixture();
    let tol = number(field(&root, "tolerance"));
    for row in rows::<13>(field(&root, "sample_weight")) {
        let got = sample_weight(
            [row[0], row[1], row[2]],
            [row[3], row[4], row[5]],
            [row[6], row[7], row[8]],
            [row[9], row[10], row[11]],
        );
        assert!(close(got, row[12], tol), "sample_weight {row:?}: got {got}");
    }
}

#[test]
fn temporal_and_depth_tables() {
    let root = fixture();
    let tol = number(field(&root, "tolerance"));
    for [history, current, usable, out] in rows::<4>(field(&root, "temporal_blend")) {
        let got = temporal_blend(history, current, usable >= 0.5);
        assert!(
            close(got, out, tol),
            "temporal {history} {current}: got {got}"
        );
    }
    for [previous, current, ok] in rows::<3>(field(&root, "depth_consistent")) {
        let got = depth_consistent(previous, current);
        assert_eq!(got, ok >= 0.5, "depth {previous} {current}");
    }
}

#[test]
fn bilateral_and_ratio_tables() {
    let root = fixture();
    let tol = number(field(&root, "tolerance"));
    for row in rows::<10>(field(&root, "bilateral_weight")) {
        let got = bilateral_weight(
            row[0] as usize,
            row[1],
            row[2],
            [row[3], row[4], row[5]],
            [row[6], row[7], row[8]],
        );
        assert!(close(got, row[9], tol), "bilateral {row:?}: got {got}");
    }
    for [s, u, want] in rows::<3>(field(&root, "ratio_visibility")) {
        let got = ratio_visibility(s, u);
        assert!(close(got, want, tol), "ratio {s} {u}: got {got}");
    }
}

#[test]
fn slot_and_group_tables() {
    let root = fixture();
    for [slot, layer, channel] in rows::<3>(field(&root, "mask_slot")) {
        let got = mask_slot(slot as u32);
        assert_eq!(got, pair([layer, channel]), "mask_slot {slot}");
    }
    for [slot, group, channel] in rows::<3>(field(&root, "history_slot")) {
        let got = history_slot(slot as u32);
        assert_eq!(got, pair([group, channel]), "history_slot {slot}");
    }
    for [limit, groups] in rows::<2>(field(&root, "groups_for")) {
        assert_eq!(
            groups_for(limit as u32),
            groups as u32,
            "groups_for {limit}"
        );
    }
    for [limit, layers] in rows::<2>(field(&root, "raw_layers")) {
        assert_eq!(
            raw_layers(limit as u32),
            layers as u32,
            "raw_layers {limit}"
        );
    }
    for [cells, side] in rows::<2>(field(&root, "cell_side")) {
        assert_eq!(cell_side(cells as u32), side as u32, "cell_side {cells}");
    }
}

#[test]
fn stratum_cell_visits_every_cell_once_per_cycle() {
    for cells in [4u32, 9] {
        let count = cell_side(cells) * cell_side(cells);
        for pixel in [[0u32, 0], [17, 5]] {
            let visited: BTreeSet<u32> = (0..count)
                .map(|frame| stratum_cell(cells, frame, pixel))
                .collect();
            assert_eq!(visited.len() as u32, count, "cells {cells} pixel {pixel:?}");
            assert!(visited.iter().all(|&cell| cell < count));
        }
    }
}

#[test]
fn r2_jitter_stays_in_unit_square_and_steps_by_alpha() {
    let pixel = [3u32, 9];
    for frame in 0..64 {
        let now = r2_jitter(frame, pixel);
        let next = r2_jitter(frame + 1, pixel);
        for ((a, b), alpha) in now.into_iter().zip(next).zip(R2_ALPHA) {
            assert!((0.0..1.0).contains(&a), "frame {frame}: {now:?}");
            let excess = (b - a) - alpha;
            assert!(
                (excess - excess.round()).abs() <= 1e-4,
                "frame {frame}: step {} against alpha {alpha}",
                b - a
            );
        }
    }
}

#[test]
fn half_float_pair_codec_table() {
    let root = restir_fixture();
    for case in items(field(&root, "pack2x16float")) {
        let v = [scalar(field(case, "x")), scalar(field(case, "y"))];
        assert_eq!(
            pack2x16float(v),
            word(field(case, "bits")),
            "pack {v:?} ({})",
            text(field(case, "note"))
        );
    }
    for case in items(field(&root, "unpack2x16float")) {
        let got = unpack2x16float(word(field(case, "bits")));
        assert_close(got, floats::<2>(field(case, "v")), 0.0, "unpack");
    }
}

#[test]
fn octahedral_codec_table() {
    let root = restir_fixture();
    for case in items(field(&root, "oct_encode")) {
        let got = oct_encode(floats::<3>(field(case, "n")));
        assert_close(got, floats::<2>(field(case, "e")), 0.000001, "oct_encode");
    }
    for case in items(field(&root, "oct_roundtrip")) {
        let n = floats::<3>(field(case, "n"));
        assert_close(oct_decode(oct_encode(n)), n, 0.001, "oct_roundtrip");
    }
}

#[test]
fn reservoir_record_table() {
    let root = restir_fixture();
    for case in items(field(&root, "pack_reservoir")) {
        let r = Reservoir {
            slot: word(field(case, "slot")),
            uv: floats::<2>(field(case, "uv")),
            weight: number(field(case, "weight")),
            confidence: number(field(case, "confidence")),
            normal_oct: floats::<2>(field(case, "normal_oct")),
        };
        let got = pack_reservoir(&r);
        let want: Vec<u32> = items(field(case, "words")).iter().map(word).collect();
        assert_eq!(
            got.to_vec(),
            want,
            "pack {r:?} ({})",
            text(field(case, "note"))
        );
    }
    for case in items(field(&root, "unpack_reservoir")) {
        let words: [u32; 4] = std::array::from_fn(|i| word(&items(field(case, "words"))[i]));
        let got = unpack_reservoir(words);
        assert_eq!(got.slot, word(field(case, "slot")), "unpack slot");
        assert_close(got.uv, floats::<2>(field(case, "uv")), 0.001, "unpack uv");
        assert_close(
            [got.weight, got.confidence],
            floats::<2>(field(case, "weight_confidence")),
            0.001,
            "unpack weight and confidence",
        );
        assert_close(
            got.normal_oct,
            floats::<2>(field(case, "normal_oct")),
            0.001,
            "unpack normal",
        );
    }
}

#[test]
fn empty_and_saturated_reservoir_words() {
    let root = restir_fixture();
    let empty: Vec<u32> = items(field(&root, "empty")).iter().map(word).collect();
    assert_eq!(pack_reservoir(&Reservoir::EMPTY).to_vec(), empty);
    assert_eq!(pack_reservoir(&Reservoir::EMPTY), [0xFF, 0, 0, 0]);
    let heavy = Reservoir {
        weight: 1e6,
        ..Reservoir::EMPTY
    };
    assert_eq!(pack_reservoir(&heavy)[2] & 0xFFFF, 0x7BFF);
}

#[test]
fn reservoir_acceptance_and_weight_tables() {
    let root = restir_fixture();
    for case in items(field(&root, "wrs_accept")) {
        let got = wrs_accept(
            number(field(case, "w_sum")),
            number(field(case, "w")),
            number(field(case, "xi")),
        );
        assert_eq!(
            got,
            boolean(field(case, "accept")),
            "wrs_accept ({})",
            text(field(case, "note"))
        );
    }
    for case in items(field(&root, "contribution_weight")) {
        let got = contribution_weight(number(field(case, "w_sum")), number(field(case, "target")));
        assert!(
            close(got, number(field(case, "weight")), 1e-6),
            "contribution_weight ({})",
            text(field(case, "note"))
        );
    }
    for case in items(field(&root, "clamp_confidence")) {
        let got = clamp_confidence(
            number(field(case, "previous")),
            number(field(case, "current")),
            number(field(case, "cap")),
        );
        assert!(
            close(got, number(field(case, "confidence")), 1e-6),
            "clamp_confidence ({})",
            text(field(case, "note"))
        );
    }
}

#[test]
fn balance_weight_table() {
    let root = restir_fixture();
    for case in items(field(&root, "balance_weight")) {
        let terms: Vec<(f32, f32)> = items(field(case, "terms"))
            .iter()
            .map(|row| {
                let [m, target] = floats::<2>(row);
                (m, target)
            })
            .collect();
        let got = balance_weight(&terms, word(field(case, "index")) as usize);
        assert!(
            close(got, number(field(case, "weight")), 1e-6),
            "balance_weight ({})",
            text(field(case, "note"))
        );
    }
}

#[test]
fn slot_remap_tables() {
    let root = restir_fixture();
    for case in items(field(&root, "remap_slot")) {
        let remap: [u32; 8] = std::array::from_fn(|i| word(&items(field(case, "remap"))[i]));
        let got = remap_slot(&remap, word(field(case, "slot")));
        assert_eq!(
            got,
            optional_word(field(case, "mapped")),
            "remap_slot ({})",
            text(field(case, "note"))
        );
    }
    for case in items(field(&root, "remap_table")) {
        let previous: [u32; 8] = std::array::from_fn(|i| word(&items(field(case, "previous"))[i]));
        let current: [u32; 8] = std::array::from_fn(|i| word(&items(field(case, "current"))[i]));
        let got = remap_table(&previous, &current, boolean(field(case, "history_valid")));
        let want: [u32; 8] = std::array::from_fn(|i| word(&items(field(case, "lanes"))[i]));
        assert_eq!(got, want, "remap_table ({})", text(field(case, "note")));
    }
    assert_eq!(
        remap_slot(&[0, REMAP_INVALID, 2, 3, 4, 5, 6, 7], 2),
        Some(2)
    );
}

#[test]
fn boiling_and_depth_tables() {
    let root = restir_fixture();
    for case in items(field(&root, "boiling_keep")) {
        let got = boiling_keep(
            number(field(case, "weight")),
            number(field(case, "mean")),
            number(field(case, "factor")),
        );
        assert_eq!(
            got,
            boolean(field(case, "keep")),
            "boiling_keep ({})",
            text(field(case, "note"))
        );
    }
    for case in items(field(&root, "depth_similar")) {
        let got = depth_similar(
            number(field(case, "neighbour")),
            number(field(case, "centre")),
        );
        assert_eq!(
            got,
            boolean(field(case, "similar")),
            "depth_similar ({})",
            text(field(case, "note"))
        );
    }
}

#[test]
fn spatial_offsets_stay_in_the_disk_and_reach_both_signs() {
    let radius = RESTIR_SPATIAL_RADIUS;
    let mut x_signs = BTreeSet::new();
    let mut y_signs = BTreeSet::new();
    for frame in 0..4u32 {
        for k in 0..64u32 {
            let offset = spatial_offset([17, 9], frame, k, radius);
            let [x, y] = offset.map(f64::from);
            assert!(
                (x * x + y * y).sqrt() <= f64::from(radius) + 1.0,
                "frame {frame} k {k}: {offset:?}"
            );
            if x != 0.0 {
                x_signs.insert(x.signum() as i32);
            }
            if y != 0.0 {
                y_signs.insert(y.signum() as i32);
            }
        }
    }
    assert_eq!(x_signs, BTreeSet::from([-1, 1]), "x axis signs");
    assert_eq!(y_signs, BTreeSet::from([-1, 1]), "y axis signs");
}

#[test]
fn spherical_rectangle_tables() {
    let root = restir_fixture();
    let tol = number(field(&root, "tolerance"));
    for case in items(field(&root, "rect_solid_angle")) {
        let got = rect_solid_angle(
            floats::<3>(field(case, "origin")),
            floats::<3>(field(case, "corner")),
            floats::<3>(field(case, "edge_x")),
            floats::<3>(field(case, "edge_y")),
        );
        assert!(
            close(got, number(field(case, "solid_angle")), tol),
            "rect_solid_angle ({})",
            text(field(case, "note"))
        );
    }
    for case in items(field(&root, "sample_rect")) {
        let (uv, solid) = sample_rect(
            floats::<3>(field(case, "origin")),
            floats::<3>(field(case, "corner")),
            floats::<3>(field(case, "edge_x")),
            floats::<3>(field(case, "edge_y")),
            floats::<2>(field(case, "xi")),
        );
        assert_close(uv, floats::<2>(field(case, "uv")), tol, "sample_rect uv");
        assert!(
            close(solid, number(field(case, "solid_angle")), tol),
            "sample_rect solid angle ({})",
            text(field(case, "note"))
        );
    }
    let origin = [0.0, 0.0, 0.0];
    let corner = [-1.0, -1.0, 1.0];
    let edge_y = [0.0, 2.0, 0.0];
    assert_eq!(
        sample_rect(origin, corner, [0.0; 3], edge_y, [0.5, 0.5]),
        ([0.5, 0.5], 0.0),
        "a degenerate rectangle falls back to the centre"
    );
    assert_eq!(
        sample_rect([0.0, 0.0, 1.0], corner, [2.0, 0.0, 0.0], edge_y, [0.5, 0.5]),
        ([0.5, 0.5], 0.0),
        "an origin in the rectangle's plane falls back to the centre"
    );
    assert_eq!(
        rect_solid_angle(origin, corner, [0.0; 3], edge_y),
        0.0,
        "a degenerate rectangle has no solid angle"
    );
}

#[test]
fn area_pdf_srgb_and_luminance_tables() {
    let root = restir_fixture();
    let tol = number(field(&root, "tolerance"));
    for case in items(field(&root, "area_pdf")) {
        let got = area_pdf(
            number(field(case, "solid_angle")),
            number(field(case, "cos_light")),
            number(field(case, "distance")),
        );
        assert!(
            close(got, number(field(case, "pdf")), tol),
            "area_pdf ({})",
            text(field(case, "note"))
        );
    }
    for case in items(field(&root, "srgb_decode")) {
        let got = srgb_decode(number(field(case, "c")));
        assert!(
            close(got, number(field(case, "linear")), tol),
            "srgb_decode ({})",
            text(field(case, "note"))
        );
    }
    for case in items(field(&root, "luminance")) {
        let got = luminance(floats::<3>(field(case, "c")));
        assert!(
            close(got, number(field(case, "value")), tol),
            "luminance ({})",
            text(field(case, "note"))
        );
    }
}

#[test]
fn target_luminance_table() {
    let root = restir_fixture();
    let tol = number(field(&root, "tolerance"));
    for case in items(field(&root, "target_luminance")) {
        let surface = Surface {
            position: floats::<3>(field(case, "position")),
            normal: floats::<3>(field(case, "normal")),
            view: floats::<3>(field(case, "view")),
            roughness: number(field(case, "roughness")),
            f0: floats::<3>(field(case, "f0")),
            metallic: number(field(case, "metallic")),
        };
        let got = target_luminance(
            &surface,
            floats::<3>(field(case, "point")),
            floats::<3>(field(case, "travel")),
            floats::<3>(field(case, "radiance")),
            number(field(case, "window")),
        );
        assert!(
            close(got, number(field(case, "value")), tol),
            "target_luminance ({})",
            text(field(case, "note"))
        );
    }
}

#[test]
fn sampled_rectangle_area_is_analytic() {
    let origin = [0.0, 0.0, 0.0];
    let corner = [-1.0, -1.0, 1.0];
    let edge_x = [2.0, 0.0, 0.0];
    let edge_y = [0.0, 2.0, 0.0];
    let normal = [0.0, 0.0, 1.0];
    let mut sum = 0.0;
    let mut count = 0.0;
    for i in 0..16u32 {
        for j in 0..16u32 {
            let xi = [(i as f32 + 0.5) / 16.0, (j as f32 + 0.5) / 16.0];
            let (uv, solid) = sample_rect(origin, corner, edge_x, edge_y, xi);
            let point = [
                corner[0] + uv[0] * edge_x[0] + uv[1] * edge_y[0],
                corner[1] + uv[0] * edge_x[1] + uv[1] * edge_y[1],
                corner[2] + uv[0] * edge_x[2] + uv[1] * edge_y[2],
            ];
            let d = [
                point[0] - origin[0],
                point[1] - origin[1],
                point[2] - origin[2],
            ];
            let distance = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let cos_light =
                (normal[0] * d[0] + normal[1] * d[1] + normal[2] * d[2]).abs() / distance;
            let pdf = area_pdf(solid, cos_light, distance);
            assert!(pdf > 0.0, "sample {i},{j} has zero pdf");
            sum += 1.0 / pdf;
            count += 1.0;
        }
    }
    let mean = sum / count;
    assert!(
        (mean - 4.0).abs() <= 0.02 * 4.0,
        "mean sampled area {mean} is not within 2% of 4"
    );
}

const AREA_TRACE_WGSL: &str = include_str!("../src/wgsl/area_shadow/trace.wgsl");

#[test]
fn ray_bias_view_depth_identity_camera() {
    let inv_view = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 2.0, 5.0, 1.0],
    ];
    let world = [1.0, 1.0, -3.0];
    let depth = ray_view_depth(inv_view, world);
    assert!(close(depth, 8.0, 1e-5), "depth {depth}");
    assert!(!close(depth, -2.0, 1e-5), "old formula value");
    assert_close(
        ray_origin(world, [0.0, 1.0, 0.0], 8.0),
        [1.0, 1.026, -3.0],
        1e-5,
        "origin",
    );
}

#[test]
fn ray_bias_view_depth_rotated_camera() {
    let inv_view = [
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 1.0],
    ];
    let world = [-4.0, 1.0, 3.0];
    let depth = ray_view_depth(inv_view, world);
    assert!(close(depth, 6.0, 1e-5), "depth {depth}");
    let o = ray_origin(world, [0.0, 1.0, 0.0], depth);
    assert!(
        close(o[1] - world[1], 0.022, 1e-5),
        "bias {}",
        o[1] - world[1]
    );
}

#[test]
fn area_trace_wgsl_ray_bias_pin() {
    assert!(!AREA_TRACE_WGSL.contains("inv_view * vec4<f32>(world, 1.0)"));
    assert!(
        AREA_TRACE_WGSL
            .contains("-dot(raa_gi_frame.inv_view[2].xyz, world - raa_gi_frame.inv_view[3].xyz)")
    );
    assert!(AREA_TRACE_WGSL.contains("0.01 + 0.002 * view_depth"));
}
