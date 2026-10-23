//! Q11: the bloom render module passes naga validation, defines the fullscreen vertex entry
//! and the four fragment entries named by `shaders.rs`, binds only group 0 (no PRELUDE), and
//! `BloomParams` has the field order and size of `PARAMS_SIZE`.

// The shader module is crate-private; the test compiles its file.
#[path = "../src/shaders.rs"]
mod shaders;

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
fn bloom_module_validates() {
    if let Err(e) = Composer::validate(shaders::BLOOM_WGSL) {
        panic!("bloom: {e}");
    }
}

#[test]
fn one_vertex_and_four_fragment_entries() {
    let entries = entry_points(shaders::BLOOM_WGSL);
    let expected = [
        ("@vertex", shaders::VS_MAIN),
        ("@fragment", shaders::FS_DOWN_FIRST),
        ("@fragment", shaders::FS_DOWN),
        ("@fragment", shaders::FS_UP),
        ("@fragment", shaders::FS_COMPOSITE),
    ];
    for e in expected {
        assert!(entries.contains(&e), "missing {e:?} in {entries:?}");
    }
    assert_eq!(entries.len(), expected.len(), "{entries:?}");
    let wgsl = shaders::BLOOM_WGSL;
    assert_eq!(wgsl.matches("@vertex").count(), 1);
    assert_eq!(wgsl.matches("@fragment").count(), 4);
    assert_eq!(wgsl.matches("@compute").count(), 0);
}

#[test]
fn only_group_zero_and_no_prelude() {
    let mut bindings = Vec::new();
    for l in code_lines(shaders::BLOOM_WGSL) {
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
fn bloom_params_field_order_matches_size() {
    let got: Vec<(String, String)> = code_lines(shaders::BLOOM_WGSL)
        .skip_while(|l| !l.starts_with("struct BloomParams"))
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
        ("texel_size", "vec2<f32>", 8u64),
        ("radius", "f32", 4),
        ("strength", "f32", 4),
    ];
    let got: Vec<(&str, &str)> = got.iter().map(|(n, t)| (n.as_str(), t.as_str())).collect();
    let want: Vec<(&str, &str)> = expected.iter().map(|(n, t, _)| (*n, *t)).collect();
    assert_eq!(got, want);
    let size: u64 = expected.iter().map(|(_, _, s)| s).sum();
    assert_eq!(size, shaders::PARAMS_SIZE);
}
