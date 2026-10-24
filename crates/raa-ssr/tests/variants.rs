//! Q11: the SSR Hi-Z / HDR mip module passes naga validation, defines the two compute
//! entries (8x8 workgroups), the fullscreen vertex entry and `fs_downsample` named by
//! `shaders.rs`, binds only group 0 (no PRELUDE), and its storage texture is r32float with
//! write access.

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
