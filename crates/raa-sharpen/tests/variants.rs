//! P5-Q1: the sharpen module validates with naga, defines exactly `vs_main` and `fs_sharpen`,
//! binds group 0 bindings 0..1 only, and its params struct matches `PARAMS_SIZE`.

use raa_shader::Composer;
use raa_sharpen::{FS_SHARPEN, PARAMS_SIZE, SHARPEN_WGSL, VS_MAIN};

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

/// `(binding, declaration)` of every group-0 binding; panics on another group.
fn bindings(wgsl: &str) -> Vec<(u32, &str)> {
    code_lines(wgsl)
        .filter_map(|l| l.strip_prefix("@group("))
        .map(|rest| {
            let rest = rest
                .strip_prefix("0) @binding(")
                .unwrap_or_else(|| panic!("group other than 0: {rest}"));
            let (b, decl) = rest.split_once(')').unwrap_or_else(|| panic!("{rest}"));
            (b.parse().unwrap_or_else(|_| panic!("{rest}")), decl.trim())
        })
        .collect()
}

#[test]
fn sharpen_module_validates() {
    if let Err(e) = Composer::validate(SHARPEN_WGSL) {
        panic!("sharpen: {e}");
    }
}

#[test]
fn entries_vertex_and_fragment() {
    assert_eq!(
        entry_points(SHARPEN_WGSL),
        [("@vertex", VS_MAIN), ("@fragment", FS_SHARPEN)]
    );
    assert_eq!(SHARPEN_WGSL.matches("@compute").count(), 0);
}

#[test]
fn bindings_and_no_prelude() {
    for l in code_lines(SHARPEN_WGSL) {
        assert!(!l.contains("raa_"), "PRELUDE identifier: {l}");
    }
    let b = bindings(SHARPEN_WGSL);
    let indices: Vec<u32> = b.iter().map(|(i, _)| *i).collect();
    assert_eq!(indices, [0, 1]);
    assert!(
        b[0].1.starts_with("var<uniform> params: SharpenParams"),
        "{}",
        b[0].1
    );
    assert!(b[1].1.contains("texture_2d<f32>"), "{}", b[1].1);
}

#[test]
fn params_fields_match_size() {
    let params = [
        ("sharpness", "f32"),
        ("_pad0", "u32"),
        ("_pad1", "u32"),
        ("_pad2", "u32"),
    ];
    assert_eq!(fields(SHARPEN_WGSL, "SharpenParams"), params);
    assert_eq!(params.len() * 4, PARAMS_SIZE);
}

#[test]
fn attribution_line_present() {
    assert_eq!(
        SHARPEN_WGSL.lines().next().unwrap_or(""),
        "// Adapted from solstice@7e465b5 crates/solstice-render/shaders/post/tonemap.wgsl, \
         MIT licence, Copyright (c) 2026 SamG-Coder"
    );
}

#[test]
fn fs_sharpen_uses_group_zero_bindings() {
    use raa_core::wgpu::naga;
    let module = match naga::front::wgsl::parse_str(SHARPEN_WGSL) {
        Ok(m) => m,
        Err(e) => panic!("parse: {e}"),
    };
    let info = match naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    {
        Ok(i) => i,
        Err(e) => panic!("validate: {e}"),
    };
    let Some(index) = module
        .entry_points
        .iter()
        .position(|e| e.name == FS_SHARPEN)
    else {
        panic!("no {FS_SHARPEN} entry");
    };
    let fs = info.get_entry_point(index);
    let mut used: Vec<u32> = module
        .global_variables
        .iter()
        .filter(|(h, _)| !fs[*h].is_empty())
        .filter_map(|(_, g)| g.binding.as_ref())
        .filter(|b| b.group == 0)
        .map(|b| b.binding)
        .collect();
    used.sort_unstable();
    assert_eq!(used, [0, 1], "fs_sharpen bindings: {used:?}");
}
