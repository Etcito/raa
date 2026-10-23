//! Q11: the `shadow.contact` compute module passes naga validation, binds only group 0, has
//! exactly one compute entry `contact`, does not depend on the S10 frame uniform and declares
//! the `ContactParams` fields in the contract order.

// The shader source module is crate-private; the test compiles its file.
#[path = "../src/shaders.rs"]
mod shaders;

use raa_shader::Composer;

/// Lines of `wgsl` without `//` comments.
fn code_lines(wgsl: &str) -> impl Iterator<Item = &str> {
    wgsl.lines()
        .map(|l| l.split("//").next().unwrap_or("").trim())
        .filter(|l| !l.is_empty())
}

#[test]
fn contact_module_validates() {
    if let Err(e) = Composer::validate(shaders::CONTACT_WGSL) {
        panic!("contact shader refused: {e}");
    }
}

#[test]
fn contact_uses_only_group_0() {
    let mut groups = 0;
    for l in code_lines(shaders::CONTACT_WGSL) {
        for (i, _) in l.match_indices("@group(") {
            assert!(l[i..].starts_with("@group(0)"), "{l}");
            groups += 1;
        }
    }
    assert_eq!(groups, 3);
}

#[test]
fn contact_has_one_compute_entry_named_contact() {
    let lines: Vec<&str> = code_lines(shaders::CONTACT_WGSL).collect();
    let computes: Vec<usize> = (0..lines.len())
        .filter(|i| lines[*i].contains("@compute"))
        .collect();
    assert_eq!(computes.len(), 1, "{computes:?}");
    let i = computes[0];
    let decl = format!("{} {}", lines[i], lines.get(i + 1).unwrap_or(&""));
    assert!(decl.contains("@workgroup_size(8, 8)"), "{decl}");
    assert!(decl.contains("fn contact("), "{decl}");
    assert!(!shaders::CONTACT_WGSL.contains("@vertex"));
    assert!(!shaders::CONTACT_WGSL.contains("@fragment"));
}

#[test]
fn contact_does_not_reference_raa_frame() {
    assert!(!shaders::CONTACT_WGSL.contains("raa_frame"));
}

#[test]
fn contact_params_field_order_matches_contract() {
    let body: Vec<(String, String)> = code_lines(shaders::CONTACT_WGSL)
        .skip_while(|l| !l.starts_with("struct ContactParams"))
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
        ("proj", "mat4x4<f32>", 64),
        ("inv_proj", "mat4x4<f32>", 64),
        ("viewport", "vec4<f32>", 16),
        ("sun_view", "vec4<f32>", 16),
        ("params", "vec4<f32>", 16),
    ];
    let got: Vec<(&str, &str)> = body.iter().map(|(n, t)| (n.as_str(), t.as_str())).collect();
    let want: Vec<(&str, &str)> = expected.iter().map(|(n, t, _)| (*n, *t)).collect();
    assert_eq!(got, want);
    let size: u64 = expected.iter().map(|(_, _, s)| s).sum();
    assert_eq!(size, shaders::PARAMS_SIZE);
}
