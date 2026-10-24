//! Q11: the SSR Hi-Z / HDR mip module passes naga validation, defines the two compute
//! entries (8x8 workgroups), the fullscreen vertex entry and `fs_downsample` named by
//! `shaders.rs`, binds only group 0 (no PRELUDE), and its storage texture is r32float with
//! write access. The trace / composite module validates, defines the fullscreen vertex entry
//! plus `fs_trace` and `fs_composite`, declares the private group-0 bindings plus the
//! group-1 lighting bindings 3-6 copied verbatim from `raa_shader::PRELUDE_WGSL` (with its
//! `EnvSh` members), and `SsrParams` has the field order and size of `SSR_PARAMS_SIZE`.

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
fn ssr_hiz_module_validates() {
    if let Err(e) = Composer::validate(shaders::SSR_HIZ_WGSL) {
        panic!("ssr_hiz: {e}");
    }
}

#[test]
fn compute_vertex_and_fragment_entries() {
    let compute = format!(
        "@compute @workgroup_size({0}, {0}, 1)",
        shaders::HIZ_WORKGROUP
    );
    let entries = entry_points(shaders::SSR_HIZ_WGSL);
    let expected = [
        (compute.as_str(), shaders::CS_HIZ_COPY),
        (compute.as_str(), shaders::CS_HIZ_REDUCE),
        ("@vertex", shaders::VS_FULLSCREEN),
        ("@fragment", shaders::FS_DOWNSAMPLE),
    ];
    for e in expected {
        assert!(entries.contains(&e), "missing {e:?} in {entries:?}");
    }
    assert_eq!(entries.len(), expected.len(), "{entries:?}");
    assert_eq!(shaders::HIZ_WORKGROUP, 8);
}

#[test]
fn storage_is_r32float_write_and_inputs_use_texture_load() {
    let wgsl = shaders::SSR_HIZ_WGSL;
    let storage: Vec<&str> = code_lines(wgsl)
        .filter(|l| l.contains("texture_storage_"))
        .collect();
    assert_eq!(storage.len(), 1, "{storage:?}");
    assert!(
        storage[0].contains("texture_storage_2d<r32float, write>"),
        "{}",
        storage[0]
    );
    assert!(code_lines(wgsl).any(|l| l.contains("var depth: texture_depth_2d;")));
    assert!(code_lines(wgsl).any(|l| l.contains("var hiz_src: texture_2d<f32>;")));
    assert!(code_lines(wgsl).any(|l| l.contains("textureLoad(depth,")));
    assert!(code_lines(wgsl).any(|l| l.contains("textureLoad(hiz_src,")));
    // The reduce cites the CPU rule it mirrors.
    assert!(wgsl.contains("src/hiz.rs"));
}

#[test]
fn only_group_zero_and_no_prelude() {
    let mut bindings = Vec::new();
    for l in code_lines(shaders::SSR_HIZ_WGSL) {
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
    assert_eq!(bindings, [0, 1, 2, 3, 4, 5]);
}

#[test]
fn hiz_params_is_one_uvec4() {
    let fields: Vec<&str> = code_lines(shaders::SSR_HIZ_WGSL)
        .skip_while(|l| !l.starts_with("struct HizParams"))
        .skip(1)
        .take_while(|l| !l.starts_with('}'))
        .collect();
    assert_eq!(fields, ["size: vec4<u32>,"]);
    assert_eq!(shaders::HIZ_PARAMS_SIZE, 16);
}

#[test]
fn ssr_trace_module_validates() {
    if let Err(e) = Composer::validate(shaders::SSR_TRACE_WGSL) {
        panic!("ssr_trace: {e}");
    }
}

#[test]
fn trace_vertex_and_two_fragment_entries() {
    let entries = entry_points(shaders::SSR_TRACE_WGSL);
    let expected = [
        ("@vertex", shaders::VS_FULLSCREEN),
        ("@fragment", shaders::FS_TRACE),
        ("@fragment", shaders::FS_COMPOSITE),
    ];
    for e in expected {
        assert!(entries.contains(&e), "missing {e:?} in {entries:?}");
    }
    assert_eq!(entries.len(), expected.len(), "{entries:?}");
    assert_eq!(shaders::SSR_TRACE_WGSL.matches("@compute").count(), 0);
}

#[test]
fn trace_groups_are_private_zero_and_lighting_one_only() {
    let wgsl = shaders::SSR_TRACE_WGSL;
    assert_eq!(bindings(wgsl, 0), [0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(bindings(wgsl, 1), [3, 4, 5, 6]);
    let groups = code_lines(wgsl).filter(|l| l.contains("@group(")).count();
    assert_eq!(groups, 12, "only groups 0 and 1 are declared");
    // No PRELUDE prepended: its group-0 frame bindings and helpers are absent.
    assert!(!wgsl.contains("raa_frame"), "PRELUDE prepended");
    assert!(!wgsl.contains("fn raa_"), "PRELUDE prepended");
}

#[test]
fn trace_group_one_lines_occur_verbatim_in_prelude() {
    let prelude: Vec<&str> = code_lines(PRELUDE_WGSL).collect();
    let ours: Vec<&str> = code_lines(shaders::SSR_TRACE_WGSL)
        .filter(|l| l.starts_with("@group(1)"))
        .collect();
    assert_eq!(ours.len(), 4, "{ours:?}");
    for l in ours {
        assert!(prelude.contains(&l), "not in PRELUDE_WGSL: {l}");
    }
}

#[test]
fn env_sh_struct_matches_prelude() {
    let ours = struct_members(shaders::SSR_TRACE_WGSL, "EnvSh");
    let prelude = struct_members(PRELUDE_WGSL, "EnvSh");
    assert_eq!(ours, prelude, "struct EnvSh differs from PRELUDE_WGSL");
}

#[test]
fn ssr_params_field_order_matches_size() {
    let got: Vec<(String, String)> = struct_members(shaders::SSR_TRACE_WGSL, "SsrParams")
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
        ("proj", "mat4x4<f32>", 64u64),
        ("inv_proj", "mat4x4<f32>", 64),
        ("view", "mat4x4<f32>", 64),
        ("inv_view", "mat4x4<f32>", 64),
        ("viewport", "vec4<f32>", 16),
        ("params", "vec4<f32>", 16),
        ("params2", "vec4<f32>", 16),
    ];
    let got: Vec<(&str, &str)> = got.iter().map(|(n, t)| (n.as_str(), t.as_str())).collect();
    let want: Vec<(&str, &str)> = expected.iter().map(|(n, t, _)| (*n, *t)).collect();
    assert_eq!(got, want);
    let size: u64 = expected.iter().map(|(_, _, s)| s).sum();
    assert_eq!(size, shaders::SSR_PARAMS_SIZE);
}

#[test]
fn trace_marches_64_steps_then_8_refinements() {
    let lines: Vec<&str> = code_lines(shaders::SSR_TRACE_WGSL).collect();
    assert!(lines.contains(&"const MAX_STEPS: u32 = 64u;"));
    assert!(lines.contains(&"const REFINE_STEPS: u32 = 8u;"));
    assert!(lines.contains(&"const EDGE_FADE: f32 = 0.1;"));
}
