//! Hand-derived fixtures for the CPU twins of the screen probes (S39, contract `gi-screen-cpu`):
//! the SH9 projection and irradiance, the probe grid, pixels and rays, the hemisphere mapping and
//! the temporal alpha. The expected values live in `tests/fixtures/{sh9,probes,temporal}.json`,
//! derived by hand or analytically; nothing is taken from running the code under test.

use raa_gi::parts::screen::cpu::{
    PROBE_SPACING, hemisphere_direction, probe_grid, probe_pixel, ray_uv, sh9_irradiance,
    sh9_project, temporal_alpha,
};
use raa_loader::json::{JsonValue, parse};

/// The probe tile edge in pixels, as the fixture derivations state it.
const TILE: u32 = 16;

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

fn uints<const N: usize>(value: &JsonValue) -> [u32; N] {
    floats::<N>(value).map(|v| v as u32)
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

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[test]
fn constant_radiance_gives_pi_c() {
    let root = fixture("sh9.json");
    let case = field(&root, "constant");
    let tol = number(field(&root, "tolerance"));
    let radiance = floats::<3>(field(case, "radiance"));
    let samples: Vec<_> = rows::<3>(field(case, "directions"))
        .into_iter()
        .map(|d| (d, radiance))
        .collect();
    let sh = sh9_project(&samples);
    let normals = rows::<3>(field(case, "normals"));
    let want = rows::<3>(field(case, "irradiance"));
    assert_eq!(normals.len(), want.len());
    for (normal, want) in normals.into_iter().zip(want) {
        assert_close(
            sh9_irradiance(&sh, normal),
            want,
            tol,
            "constant irradiance",
        );
    }
}

#[test]
fn single_direction_lobe() {
    let root = fixture("sh9.json");
    let case = field(&root, "lobe");
    let tol = number(field(&root, "tolerance"));
    let sample = (
        floats::<3>(field(case, "direction")),
        floats::<3>(field(case, "radiance")),
    );
    let sh = sh9_project(&[sample]);
    assert_close(sh[0], [number(field(case, "sh0")); 3], tol, "lobe sh0");
    assert_close(sh[6], [number(field(case, "sh6")); 3], tol, "lobe sh6");
    for index in items(field(case, "zero")) {
        let i = number(index) as usize;
        assert!(
            sh[i].iter().all(|v| v.abs() <= tol),
            "lobe coefficient {i}: {:?}",
            sh[i]
        );
    }
    let normals = rows::<3>(field(case, "normals"));
    let want: Vec<f32> = items(field(case, "irradiance"))
        .iter()
        .map(number)
        .collect();
    assert_eq!(normals.len(), want.len());
    for (normal, want) in normals.into_iter().zip(want) {
        assert_close(
            sh9_irradiance(&sh, normal),
            [want; 3],
            tol,
            "lobe irradiance",
        );
    }
}

#[test]
fn probe_grid_sizes() {
    assert_eq!(PROBE_SPACING, TILE);
    let root = fixture("probes.json");
    for case in items(field(&root, "grids")) {
        let viewport = uints::<2>(field(case, "viewport"));
        let want = uints::<2>(field(case, "grid"));
        assert_eq!(probe_grid(viewport), want, "viewport {viewport:?}");
    }
}

#[test]
fn probe_pixels_stay_in_tile() {
    let root = fixture("probes.json");
    for case in items(field(&root, "tiles")) {
        let viewport = uints::<2>(field(case, "viewport"));
        let tile = uints::<2>(field(case, "tile"));
        for frame in 0..32 {
            let pixel = probe_pixel(tile, viewport, frame);
            for (axis, &p) in pixel.iter().enumerate() {
                let lo = tile[axis] * TILE;
                let hi = (lo + TILE - 1).min(viewport[axis] - 1);
                assert!(
                    (lo..=hi).contains(&p),
                    "tile {tile:?} frame {frame}: {pixel:?}"
                );
            }
            if frame < 16 {
                let again = probe_pixel(tile, viewport, frame + 16);
                assert_eq!(
                    pixel,
                    again,
                    "tile {tile:?} frame {frame} and {}",
                    frame + 16
                );
            }
        }
    }
    let edge = field(&root, "edge");
    let viewport = uints::<2>(field(edge, "viewport"));
    let tile = uints::<2>(field(edge, "tile"));
    let want = uints::<2>(field(edge, "pixel"));
    for frame in 0..32 {
        assert_eq!(
            probe_pixel(tile, viewport, frame),
            want,
            "edge frame {frame}"
        );
    }
}

#[test]
fn uv_strata_and_hemisphere_directions() {
    let root = fixture("probes.json");
    let rays: Vec<u32> = items(field(&root, "rays"))
        .iter()
        .map(|v| number(v) as u32)
        .collect();
    let tiles: Vec<[u32; 2]> = items(field(&root, "uv_tiles"))
        .iter()
        .map(uints::<2>)
        .collect();
    for &p in &rays {
        for tile in &tiles {
            for x in 0..p {
                for y in 0..p {
                    for frame in 0..4 {
                        let uv = ray_uv(*tile, [x, y], frame, p);
                        let (lo, hi) = (x as f32 / p as f32, (x + 1) as f32 / p as f32);
                        assert!((lo..hi).contains(&uv[0]), "rays {p} tile {tile:?}: {uv:?}");
                        assert!(
                            (0.0..1.0).contains(&uv[1]),
                            "rays {p} tile {tile:?}: {uv:?}"
                        );
                    }
                }
            }
        }
    }

    let hemi = field(&root, "hemisphere");
    let tol = number(field(hemi, "tolerance"));
    let cases = items(field(hemi, "cases"));
    for case in cases {
        let uv = floats::<2>(field(case, "uv"));
        let normal = floats::<3>(field(case, "normal"));
        let want = floats::<3>(field(case, "direction"));
        assert_close(
            hemisphere_direction(uv, normal),
            want,
            tol,
            "hemisphere case",
        );
    }
    for normal in rows::<3>(field(hemi, "unit_normals")) {
        for case in cases {
            let dir = hemisphere_direction(floats::<2>(field(case, "uv")), normal);
            assert!(
                (dot(dir, dir).sqrt() - 1.0).abs() <= tol,
                "unit length: {dir:?}"
            );
            assert!(dot(dir, normal) >= -1e-6, "{dir:?} below {normal:?}");
        }
    }
}

#[test]
fn temporal_alpha_cases() {
    let root = fixture("temporal.json");
    let tol = number(field(&root, "tolerance"));
    for case in items(field(&root, "cases")) {
        let moved = number(field(case, "moved"));
        let valid = matches!(field(case, "valid"), JsonValue::Bool(true));
        let want = number(field(case, "alpha"));
        let got = temporal_alpha(moved, valid);
        assert!(
            (got - want).abs() <= tol,
            "moved {moved} valid {valid}: got {got}, want {want}"
        );
    }
}
