//! Q11: the dof compute module passes naga validation, defines the four compute entries
//! named by `shaders.rs` with their workgroup sizes, binds only group 0 (no PRELUDE) with
//! core write-only storage formats, and `DofParams` has the field order, size (128) and
//! `lens2` offset of `shaders.rs`. The CPU twin of the autofocus CoC matches `coc_px`.

// The shader module is crate-private; the test compiles its file.
#[path = "../src/shaders.rs"]
mod shaders;

// `shaders.rs` names the descriptor as `super::DofDesc`.
use raa_dof::{DofDesc, coc_px};
use raa_shader::Composer;

/// Lines of `wgsl` without `//` comments.
fn code_lines(wgsl: &str) -> impl Iterator<Item = &str> {
    wgsl.lines()
        .map(|l| l.split("//").next().unwrap_or("").trim())
        .filter(|l| !l.is_empty())
}

/// `(stage attribute, fn name)` of every entry point: an `@vertex`, `@fragment` or
/// `@compute` line followed by the `fn` it applies to.
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

#[test]
fn dof_module_validates() {
    if let Err(e) = Composer::validate(shaders::DOF_COC_WGSL) {
        panic!("dof_coc: {e}");
    }
}

#[test]
fn four_compute_entries() {
    let entries = entry_points(shaders::DOF_COC_WGSL);
    let wg = shaders::COC_WORKGROUP;
    let tile = shaders::TILE_SIZE;
    let coc = format!("@compute @workgroup_size({wg}, {wg})");
    let tiles = format!("@compute @workgroup_size({tile}, {tile})");
    let expected = [
        ("@compute @workgroup_size(1)", shaders::CS_AUTOFOCUS),
        (coc.as_str(), shaders::CS_COC),
        (tiles.as_str(), shaders::CS_TILES),
        (coc.as_str(), shaders::CS_DILATE),
    ];
    for e in expected {
        assert!(entries.contains(&e), "missing {e:?} in {entries:?}");
    }
    assert_eq!(entries.len(), expected.len(), "{entries:?}");
    let wgsl = shaders::DOF_COC_WGSL;
    assert_eq!(wgsl.matches("@compute").count(), 4);
    assert_eq!(wgsl.matches("@vertex").count(), 0);
    assert_eq!(wgsl.matches("@fragment").count(), 0);
}

#[test]
fn only_group_zero_core_storage_formats_and_no_prelude() {
    let mut bindings = Vec::new();
    for l in code_lines(shaders::DOF_COC_WGSL) {
        assert!(!l.contains("raa_"), "PRELUDE identifier: {l}");
        if let Some(i) = l.find("texture_storage_2d<") {
            let args = &l[i + "texture_storage_2d<".len()..];
            assert!(
                args.starts_with("r32float, write>") || args.starts_with("rg32float, write>"),
                "not a core write-only storage format: {l}"
            );
        }
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
    assert_eq!(bindings, [0, 1, 2, 3, 4, 5, 6, 7]);
    let wgsl = shaders::DOF_COC_WGSL;
    assert!(wgsl.contains("var depth_tex: texture_depth_2d;"));
    assert!(wgsl.contains("var<storage, read_write> focus: f32;"));
}

#[test]
fn dof_params_field_order_matches_size() {
    let got: Vec<(String, String)> = code_lines(shaders::DOF_COC_WGSL)
        .skip_while(|l| !l.starts_with("struct DofParams"))
        .skip(1)
        .take_while(|l| !l.starts_with('}'))
        .filter_map(|l| l.split_once(':'))
        .map(|(n, t)| {
            (
                n.trim().to_string(),
                t.trim().trim_end_matches(',').to_string(),
            )
        })
        .collect();
    let expected = [
        ("inv_proj", "mat4x4<f32>", 64u64),
        ("lens", "vec4<f32>", 16),
        ("lens2", "vec4<f32>", 16),
        ("viewport", "vec4<f32>", 16),
        ("autofocus", "vec4<f32>", 16),
    ];
    let got: Vec<(&str, &str)> = got.iter().map(|(n, t)| (n.as_str(), t.as_str())).collect();
    let want: Vec<(&str, &str)> = expected.iter().map(|(n, t, _)| (*n, *t)).collect();
    assert_eq!(got, want);
    let size: u64 = expected.iter().map(|(_, _, s)| s).sum();
    assert_eq!(size, shaders::PARAMS_SIZE);
    assert_eq!(shaders::PARAMS_SIZE, 128);
    assert_eq!(shaders::PARAMS_SIZE % 16, 0);
    let lens2: u64 = expected.iter().take(2).map(|(_, _, s)| s).sum();
    assert_eq!(lens2, shaders::PARAMS_LENS2_OFFSET);
    assert_eq!(shaders::PARAMS_LENS2_OFFSET, 80);
    // the autofocus aperture term uses the focal length, never K * s / s'
    let wgsl = shaders::DOF_COC_WGSL;
    assert!(wgsl.contains("let sp = max(af, f + 0.001);"));
    assert!(wgsl.contains("k * (s - f) / (sp - f)"));
}

fn assert_twin(focal_length_mm: f32, s: f32, s_prime: f32, f_stop: f32, viewport_h: f32) {
    let d = DofDesc {
        focus_distance_m: s,
        f_stop,
        focal_length_mm,
        // no clamp in the way of the comparison
        max_coc_px: 64.0,
        ..DofDesc::default()
    };
    let at = DofDesc {
        focus_distance_m: s_prime,
        ..d
    };
    for z in [
        0.3, 0.7, 0.95, 1.0, 1.05, 1.5, 3.0, 9.0, 10.0, 11.0, 40.0, 1000.0,
    ] {
        let z = z * s_prime;
        let want = coc_px(z, &at, viewport_h);
        let got = shaders::coc_px_at_focus(z, &d, viewport_h, s_prime);
        let tol = 1e-3 * want.abs().max(1e-3);
        assert!(
            (got - want).abs() <= tol,
            "f {focal_length_mm} mm, s {s} -> s' {s_prime}, z {z}: twin {got}, coc_px {want}"
        );
    }
}

#[test]
fn autofocus_twin_matches_coc_px() {
    // 600 mm at 1 m: the case the K * s / s' approximation gets wrong
    assert_twin(600.0, 10.0, 1.0, 64.0, 720.0);
    assert_twin(600.0, 1.0, 1.0, 64.0, 1440.0);
    assert_twin(600.0, 2.0, 1.0, 32.0, 720.0);
    // 50 mm at 10 m (the defaults), from a nearer and a farther manual focus
    assert_twin(50.0, 10.0, 10.0, 2.8, 720.0);
    assert_twin(50.0, 3.0, 10.0, 2.8, 720.0);
    assert_twin(50.0, 30.0, 10.0, 2.8, 1080.0);
    // without autofocus s' = s and the term is the host's lens.y
    let d = DofDesc::default();
    let k = shaders::aperture_term_px(&d, 720.0);
    assert_eq!(shaders::coc_px_at_focus(20.0, &d, 720.0, 10.0), k * 0.5);
}

#[test]
fn autofocus_twin_clamps_and_rejects_bad_depth() {
    let d = DofDesc::default();
    let limit = d.max_coc_px * 1440.0 / 720.0;
    assert_eq!(shaders::coc_px_at_focus(0.06, &d, 1440.0, 10.0), -limit);
    // an autofocus distance at or inside the focal point is kept beyond it
    let inside = shaders::coc_px_at_focus(5.0, &d, 720.0, 0.01);
    assert_eq!(inside, shaders::coc_px_at_focus(5.0, &d, 720.0, 0.051));
    assert!(inside.is_finite() && inside > 0.0);
    for z in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert_eq!(shaders::coc_px_at_focus(z, &d, 720.0, 10.0), 0.0);
    }
}
