//! Hand-derived fixtures for the CPU twins of the M6 area-shadow kernels (S44, contract
//! `area-stochastic-cpu`): the stratum and R2 jitter, the rectangle point, the sample weight, the
//! temporal and depth tests, the bilateral weight, the ratio estimator and the slot tables. The
//! expected values live in `tests/fixtures/area_stochastic.json`, derived by hand; nothing is
//! taken from running the code under test. Two analytic properties of the hashed functions (a
//! stratum cycle and the R2 step) have no hand value and are checked directly.

use std::collections::BTreeSet;

use raa_gi::parts::area_shadow::cpu::{
    R2_ALPHA, bilateral_weight, cell_side, depth_consistent, groups_for, history_slot, mask_slot,
    r2_jitter, ratio_visibility, raw_layers, rect_point, sample_weight, stratum_cell, stratum_uv,
    temporal_blend,
};
use raa_gi::parts::area_shadow::restir_cpu::{
    Reservoir, oct_decode, oct_encode, pack_reservoir, pack2x16float, unpack_reservoir,
    unpack2x16float,
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
