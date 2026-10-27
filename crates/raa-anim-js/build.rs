//! Discover API modules in stable filename order, as in raa-js.
use std::{env, fmt::Write as _, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=src/api");
    println!("cargo:rerun-if-changed=build.rs");
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let mut files: Vec<_> = fs::read_dir(root.join("src/api"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|e| e == "rs"))
        .collect();
    files.sort();
    let mut source = String::new();
    let mut names = String::new();
    let mut calls = String::new();
    for path in files {
        let stem = path.file_stem().unwrap().to_str().unwrap();
        assert!(
            stem.starts_with(|c: char| c.is_ascii_lowercase() || c == '_')
                && stem
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'),
            "API filenames must be snake_case"
        );
        writeln!(names, "{stem}").unwrap();
        writeln!(source, "#[path = {:?}]\npub mod {stem};", path).unwrap();
        writeln!(calls, "    {stem}::register(&mut cases);").unwrap();
    }
    source.push_str("pub fn registered_cases() -> Vec<Case> {\n    let mut cases = Vec::new();\n");
    source.push_str(&calls);
    source.push_str("    cases\n}\n");
    fs::write(out.join("api_registry.rs"), source).unwrap();
    fs::write(out.join("api_files.txt"), names).unwrap();
}
