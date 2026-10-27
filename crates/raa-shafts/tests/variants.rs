//! Q11: the sun-shafts render module passes naga validation, defines the fullscreen vertex
//! entry and the two fragment entries named by `shaders.rs`, declares only private group-0
//! bindings plus the group-1 lighting bindings 0, 1, 2 and 11 copied verbatim from
//! `raa_shader::PRELUDE_WGSL` (with its `ShadowGpu` and `EnvGpu` members), and
//! `ShaftsParams` has the field order and size of `PARAMS_SIZE`.

// The shader module is crate-private; the test compiles its file.
#[path = "../src/shaders.rs"]
mod shaders;

use raa_shader::{Composer, PRELUDE_WGSL};

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

/// Member lines of `struct <name> { ... }` in `wgsl`.
fn struct_members<'a>(wgsl: &'a str, name: &str) -> Vec<&'a str> {
    let head = format!("struct {name} {{");
    let members: Vec<&str> = code_lines(wgsl)
        .skip_while(|l| *l != head)
        .skip(1)
        .take_while(|l| !l.starts_with('}'))
        .collect();
    assert!(!members.is_empty(), "struct {name} not found");
    members
}

/// Binding indices of the `@group(<group>)` declarations of `wgsl`.
fn bindings(wgsl: &str, group: u32) -> Vec<u32> {
    let prefix = format!("@group({group}) @binding(");
    code_lines(wgsl)
        .filter_map(|l| l.strip_prefix(prefix.as_str()))
        .map(|rest| {
            rest.split(')')
                .next()
                .and_then(|b| b.parse::<u32>().ok())
                .unwrap_or_else(|| panic!("binding index: {rest}"))
        })
        .collect()
}

#[test]
fn shafts_module_validates() {
    if let Err(e) = Composer::validate(shaders::SHAFTS_WGSL) {
        panic!("shafts: {e}");
    }
}

#[test]
fn one_vertex_and_two_fragment_entries() {
    let entries = entry_points(shaders::SHAFTS_WGSL);
    let expected = [
        ("@vertex", shaders::VS_MAIN),
        ("@fragment", shaders::FS_MARCH),
        ("@fragment", shaders::FS_COMPOSITE),
    ];
    for e in expected {
        assert!(entries.contains(&e), "missing {e:?} in {entries:?}");
    }
    assert_eq!(entries.len(), expected.len(), "{entries:?}");
    let wgsl = shaders::SHAFTS_WGSL;
    assert_eq!(wgsl.matches("@vertex").count(), 1);
    assert_eq!(wgsl.matches("@fragment").count(), 2);
    assert_eq!(wgsl.matches("@compute").count(), 0);
}

#[test]
fn groups_are_private_zero_and_lighting_one_only() {
    let wgsl = shaders::SHAFTS_WGSL;
    assert_eq!(bindings(wgsl, 0), [0, 1, 2, 3, 4]);
    assert_eq!(bindings(wgsl, 1), [0, 1, 2, 11]);
    let groups = code_lines(wgsl).filter(|l| l.contains("@group(")).count();
    assert_eq!(groups, 9, "only groups 0 and 1 are declared");
    // No PRELUDE prepended: the group-0 frame bindings are absent.
    assert!(!wgsl.contains("raa_frame"), "PRELUDE prepended");
}

#[test]
fn group_one_lines_occur_verbatim_in_prelude() {
    let prelude: Vec<&str> = code_lines(PRELUDE_WGSL).collect();
    for l in code_lines(shaders::SHAFTS_WGSL).filter(|l| l.starts_with("@group(1)")) {
        assert!(prelude.contains(&l), "not in PRELUDE_WGSL: {l}");
    }
}

#[test]
fn lighting_structs_match_prelude() {
    for name in ["ShadowGpu", "EnvGpu"] {
        let ours = struct_members(shaders::SHAFTS_WGSL, name);
        let prelude = struct_members(PRELUDE_WGSL, name);
        assert_eq!(ours, prelude, "struct {name} differs from PRELUDE_WGSL");
    }
}

#[test]
fn shafts_params_field_order_matches_size() {
    let got: Vec<(String, String)> = struct_members(shaders::SHAFTS_WGSL, "ShaftsParams")
        .into_iter()
        .filter_map(|l| l.split_once(':'))
        .map(|(n, t)| {
            (
                n.trim().to_string(),
                t.trim().trim_end_matches(',').to_string(),
            )
        })
        .collect();
    let expected = [
        ("inv_view_proj", "mat4x4<f32>", 64u64),
        ("view", "mat4x4<f32>", 64),
        ("camera_pos", "vec4<f32>", 16),
        ("sun_dir", "vec4<f32>", 16),
        ("sun_color", "vec4<f32>", 16),
        ("viewport", "vec4<f32>", 16),
        ("params", "vec4<f32>", 16),
    ];
    let got: Vec<(&str, &str)> = got.iter().map(|(n, t)| (n.as_str(), t.as_str())).collect();
    let want: Vec<(&str, &str)> = expected.iter().map(|(n, t, _)| (*n, *t)).collect();
    assert_eq!(got, want);
    let size: u64 = expected.iter().map(|(_, _, s)| s).sum();
    assert_eq!(size, shaders::PARAMS_SIZE);
}

#[test]
fn march_depth_is_finite_in_binary16_and_bilateral_weights_remain_finite() {
    let source = shaders::SHAFTS_WGSL;
    let sky: f32 = source
        .split("const SKY_DEPTH: f32 = ")
        .nth(1)
        .unwrap()
        .split(';')
        .next()
        .unwrap()
        .parse()
        .unwrap();
    // Binary16's maximum finite value: an old 1e6 sentinel fails here.
    assert!(sky > 1000.0 && sky <= 65504.0);
    assert!(source.contains("clamp(view_depth_of(world), 0.0, SKY_DEPTH)"));
    assert!(source.contains("clamp(view_depth_of(unproject(uv, d)), 0.0, SKY_DEPTH)"));
    for depth in [3.0_f32, sky] {
        let mut sum = 0.0;
        let mut weight = 0.0;
        for tap in [sky, sky, sky, 3.0] {
            let rel = (tap - depth).abs() / depth.max(1e-3);
            let w = 0.25 / (1e-2 + rel);
            sum += 0.5 * w;
            weight += w;
        }
        assert!(weight.is_finite() && weight > 0.0);
        assert!((sum / weight - 0.5).abs() < 1e-5);
    }
}
