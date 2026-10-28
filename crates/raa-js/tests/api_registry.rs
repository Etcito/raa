//! build.rs output: `api_files.txt` lists every `src/api/*.rs` stem in file-name order and
//! `api_registry.rs` calls each file's `install` and `frame` in that order.

use std::fs;
use std::path::Path;

fn source_stems() -> Vec<String> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("api");
    let mut stems = Vec::new();
    if let Ok(entries) = fs::read_dir(&dir) {
        for e in entries {
            let p = e.expect("readable src/api entry").path();
            if p.is_file() && p.extension().and_then(|x| x.to_str()) == Some("rs") {
                let stem = p.file_stem().and_then(|s| s.to_str()).expect("utf-8 stem");
                stems.push(stem.to_string());
            }
        }
    }
    stems.sort();
    stems
}

fn out_file(name: &str) -> String {
    let path = Path::new(env!("OUT_DIR")).join(name);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Byte offsets of `needles` in `body`, asserting each is present.
fn positions(body: &str, needles: &[String]) -> Vec<usize> {
    needles
        .iter()
        .map(|n| {
            body.find(n.as_str())
                .unwrap_or_else(|| panic!("missing `{n}`"))
        })
        .collect()
}

#[test]
fn api_files_lists_every_api_file_sorted() {
    let listed: Vec<String> = out_file("api_files.txt")
        .lines()
        .map(str::to_string)
        .collect();
    assert_eq!(listed, source_stems());
}

#[test]
fn registry_declares_a_path_module_per_file() {
    let reg = out_file("api_registry.rs");
    for stem in source_stems() {
        assert!(reg.contains(&format!("mod {stem};")), "no mod {stem}");
        assert!(
            reg.contains(&format!("/{stem}.rs\"]")),
            "no #[path] for {stem}"
        );
    }
}

fn feature_rows() -> Vec<(String, u32, u32)> {
    include_str!("../src/feature_map.rs")
        .split("(\"")
        .skip(1)
        .filter_map(|row| {
            let (name, rest) = row.split_once("\", ")?;
            let (order, bit) = rest.split_once(", ")?;
            let bit = bit.split(|c| c == ')' || c == ',').next()?;
            Some((name.to_string(), order.parse().ok()?, bit.parse().ok()?))
        })
        .collect()
}

#[test]
fn feature_bits_keep_prior_rows_and_register_skin_at_20() {
    #[rustfmt::skip]
    let expected = [
        ("sky.env", 10, 0), ("shadow.csm", 20, 1), ("prepass", 30, 2), ("ao.gtao", 40, 3),
        ("shadow.contact", 45, 4), ("particles.sim", 50, 5), ("forward", 60, 6),
        ("fur.shells", 500, 7), ("particles.opaque", 600, 8), ("sky.background", 900, 9),
        ("ssr", 65, 10), ("fog", 68, 11), ("sunshafts", 69, 12), ("opaque.copy", 70, 13),
        ("transparent.oit", 80, 14), ("particles.blend", 85, 15), ("dof", 110, 16),
        ("bloom", 120, 17), ("grading", 130, 18), ("tonemap", 200, 19)];
    let rows = feature_rows();
    assert_eq!(rows.len(), expected.len() + 1, "one appended skin row");
    let actual: Vec<(&str, u32, u32)> = rows[..expected.len()]
        .iter()
        .map(|(n, o, b)| (n.as_str(), *o, *b))
        .collect();
    assert_eq!(actual, expected);
    assert_eq!(rows.last(), Some(&("skin".to_string(), 15, 20)));
}

#[test]
fn registry_calls_install_and_frame_in_file_order() {
    let reg = out_file("api_registry.rs");
    let install_at = reg.find("pub(crate) fn install_all(").expect("install_all");
    let frame_at = reg.find("pub(crate) fn frame_all(").expect("frame_all");
    assert!(install_at < frame_at);
    let install_body = &reg[install_at..frame_at];
    let frame_body = &reg[frame_at..];
    let stems = source_stems();

    let calls: Vec<String> = stems
        .iter()
        .map(|s| format!("{s}::install(ctx)?;"))
        .collect();
    let at = positions(install_body, &calls);
    assert!(at.windows(2).all(|w| w[0] < w[1]), "install order {at:?}");

    let calls: Vec<String> = stems
        .iter()
        .map(|s| format!("{s}::frame(st, gpu, input)?;"))
        .collect();
    let at = positions(frame_body, &calls);
    assert!(at.windows(2).all(|w| w[0] < w[1]), "frame order {at:?}");

    assert!(install_body.contains("Ok(())") && frame_body.contains("Ok(())"));
}
