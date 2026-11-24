//! Hand-derived fixtures for the CPU twins of the M8 screen-space indirect light (contract
//! `ssgi-cpu`, `crates/raa-gi/src/parts/ssgi/cpu.rs`): the sector coordinate, the sector mask,
//! the sample bits, the accumulation, the slice resolve, the probe composition and the temporal
//! alpha. The expected values live in `tests/fixtures/ssgi.json`, derived by hand; nothing is
//! taken from running the code under test.

use raa_gi::parts::ssgi::cpu::{
    accumulate, compose, lit_mask, resolve, sample_bits, sector_u, temporal_alpha, update_sectors,
};
use raa_loader::json::{JsonValue, parse};

/// The absolute tolerance of every float comparison. Masks have none: they compare exactly.
const TOLERANCE: f32 = 1e-5;

fn fixture() -> JsonValue {
    parse(include_str!("fixtures/ssgi.json")).unwrap_or_else(|e| panic!("ssgi.json: {e}"))
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
        panic!("expected a number, got {value:?}");
    };
    *n as f32
}

/// A mask is an integer up to 2^32 - 1, which an f32 cannot hold exactly, so it is read as a u32.
fn word(value: &JsonValue) -> u32 {
    let JsonValue::Number(n) = value else {
        panic!("expected a number, got {value:?}");
    };
    *n as u32
}

fn boolean(value: &JsonValue) -> bool {
    let JsonValue::Bool(flag) = value else {
        panic!("expected a boolean, got {value:?}");
    };
    *flag
}

fn floats<const N: usize>(value: &JsonValue) -> [f32; N] {
    let list = items(value);
    assert_eq!(list.len(), N, "expected {N} numbers");
    std::array::from_fn(|i| number(&list[i]))
}

fn assert_close<const N: usize>(got: [f32; N], want: [f32; N], what: &str) {
    for (g, w) in got.into_iter().zip(want) {
        assert!(
            (g - w).abs() <= TOLERANCE,
            "{what}: got {got:?}, want {want:?}"
        );
    }
}

/// Checks `update_sectors` against its doc comment: `start` is `min(floor(min_u x 32), 31)`,
/// `count` is capped at `32 - start`, and the mask is 0 when the count is 0.
#[test]
fn update_sectors_matches_the_fixture() {
    let root = fixture();
    for case in items(field(&root, "update_sectors")) {
        let got = update_sectors(number(field(case, "min_u")), number(field(case, "max_u")));
        assert_eq!(got, word(field(case, "mask")), "update_sectors {case:?}");
    }
}

/// Checks `sector_u` against its doc comment: `(sin(clamp(theta - gamma, -PI/2, PI/2)) + 1) / 2`.
#[test]
fn sector_u_matches_the_fixture() {
    let root = fixture();
    for case in items(field(&root, "sector_u")) {
        let got = sector_u(number(field(case, "theta")), number(field(case, "gamma")));
        assert_close([got], [number(field(case, "u"))], "sector_u");
    }
}

/// Checks `sample_bits` against its doc comment: the sectors between the front and back faces,
/// each mapped through `sector_u` and merged by `update_sectors`.
#[test]
fn sample_bits_matches_the_fixture() {
    let root = fixture();
    for case in items(field(&root, "sample_bits")) {
        let got = sample_bits(
            floats(field(case, "delta")),
            floats(field(case, "view")),
            number(field(case, "thickness")),
            number(field(case, "side")),
            number(field(case, "gamma")),
        );
        assert_eq!(got, word(field(case, "mask")), "sample_bits {case:?}");
    }
}

#[test]
fn estimated_back_face_is_bounded_for_a_near_sample() {
    // A sample 0.1 m away can extend only 0.05 m behind its visible face.
    // With gamma = pi/4, its angular interval runs from u = 0.85355 to 0.97434.
    // This spans four sectors from 27, rather than extending to the final sector.
    assert_eq!(
        sample_bits(
            [0.1, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            0.5,
            1.0,
            std::f32::consts::FRAC_PI_4,
        ),
        0x7800_0000
    );
}

/// Checks `accumulate` against its doc comment: `new = bits & !mask` lights only when `facing`,
/// as `radiance x popcount(new) / 32`, and the mask becomes `mask | bits`.
#[test]
fn accumulate_matches_the_fixture() {
    let root = fixture();
    for case in items(field(&root, "accumulate")) {
        let mut irradiance = floats(field(case, "irradiance"));
        let mask = accumulate(
            word(field(case, "mask")),
            word(field(case, "bits")),
            floats(field(case, "radiance")),
            boolean(field(case, "facing")),
            &mut irradiance,
        );
        assert_eq!(mask, word(field(case, "want_mask")), "accumulate {case:?}");
        assert_close(
            irradiance,
            floats(field(case, "want_irradiance")),
            "accumulate",
        );
    }
}

/// Checks `resolve` against its doc comment: `E_ss = PI x mean(sum)`, the visibility is the mean
/// of `1 - popcount(mask) / 32`, and an empty slice list gives `([0, 0, 0], 1)`.
#[test]
fn resolve_matches_the_fixture() {
    let root = fixture();
    for case in items(field(&root, "resolve")) {
        let slices: Vec<([f32; 3], u32)> = items(field(case, "slices"))
            .iter()
            .map(|slice| {
                (
                    floats(field(slice, "irradiance")),
                    word(field(slice, "mask")),
                )
            })
            .collect();
        let (irradiance, visibility) = resolve(&slices);
        assert_close(
            irradiance,
            floats(field(case, "want_irradiance")),
            "resolve E_ss",
        );
        assert_close(
            [visibility],
            [number(field(case, "want_visibility"))],
            "resolve V",
        );
    }
}

/// Checks `compose` against its doc comment, `(E_ss + V x probe.rgb, probe.a)`. The open
/// surface case (E_ss 0, V 1) keeps the probe exactly, so the screen term is never added twice.
#[test]
fn compose_keeps_the_probe_on_an_open_surface_and_matches_the_fixture() {
    let root = fixture();
    for case in items(field(&root, "compose")) {
        let got = compose(floats(field(case, "screen")), floats(field(case, "probe")));
        assert_close(got, floats(field(case, "want")), "compose");
    }
    let probe = [0.3, 0.4, 0.5, 0.9];
    assert_eq!(compose([0.0, 0.0, 0.0, 1.0], probe), probe);
}

/// Checks `temporal_alpha` against its doc comment: 0.1 whatever the motion.
#[test]
fn temporal_alpha_matches_the_fixture() {
    let root = fixture();
    for case in items(field(&root, "temporal_alpha")) {
        let got = temporal_alpha(number(field(case, "moved_px")));
        assert_close([got], [number(field(case, "alpha"))], "temporal_alpha");
    }
}

/// Hand-derived: a back-facing sample over bits 0x0000_00FF of an empty slice occludes them
/// (mask 0xFF) but claims none (lit 0), so the resolve keeps V 1 with E_ss 0 and the composite is
/// the probe exactly: hidden sectors keep the probe irradiance.
#[test]
fn hidden_back_face_keeps_the_probe() {
    let bits = 0x0000_00FF;
    let mut irradiance = [0.0; 3];
    let mask = accumulate(0, bits, [1.0, 0.0, 0.0], false, &mut irradiance);
    let lit = lit_mask(0, 0, bits, false);
    assert_eq!((mask, lit), (bits, 0));
    let (e_ss, visibility) = resolve(&[(irradiance, lit)]);
    assert_eq!((e_ss, visibility), ([0.0; 3], 1.0));
    let probe = [0.2, 0.3, 0.4, 1.0];
    let [r, g, b] = e_ss;
    assert_eq!(compose([r, g, b, visibility], probe), probe);
}

/// Hand-derived: a facing sample with radiance (1, 0, 0) on the same 8 bits claims them
/// (lit 0xFF): V = 1 - 8/32 = 0.75 and E_ss red = PI x 8/32 (0.785398).
#[test]
fn facing_lit_sample_claims_its_sectors() {
    let bits = 0x0000_00FF;
    let mut irradiance = [0.0; 3];
    accumulate(0, bits, [1.0, 0.0, 0.0], true, &mut irradiance);
    let lit = lit_mask(0, 0, bits, true);
    assert_eq!(lit, bits);
    let (e_ss, visibility) = resolve(&[(irradiance, lit)]);
    assert_close(e_ss, [std::f32::consts::PI * 8.0 / 32.0, 0.0, 0.0], "E_ss");
    assert_close([visibility], [0.75], "V");
}

/// Hand-derived: a facing sample behind a hidden one on the same bits finds no fresh sector
/// (`bits & !mask` = 0), so it neither lights nor claims: E_ss 0, lit 0, V 1.
#[test]
fn facing_sample_behind_a_hidden_one_adds_nothing() {
    let bits = 0x0000_00FF;
    let mut irradiance = [0.0; 3];
    let mut mask = accumulate(0, bits, [0.0; 3], false, &mut irradiance);
    let mut lit = lit_mask(0, 0, bits, false);
    lit = lit_mask(lit, mask, bits, true);
    mask = accumulate(mask, bits, [1.0, 0.0, 0.0], true, &mut irradiance);
    assert_eq!((mask, lit), (bits, 0));
    assert_eq!(resolve(&[(irradiance, lit)]), ([0.0; 3], 1.0));
}
