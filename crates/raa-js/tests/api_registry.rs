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
