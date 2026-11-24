//! Discovery of the render graph extension files: lists `src/nodes/*.rs`,
//! `src/exec_ext/*.rs`, `src/frozen/*.rs`, `graphs/*.json` and `presets/*.json`, each
//! sorted by file stem in C-locale (byte) order, and writes `$OUT_DIR/registry.rs`.
//!
//! A missing directory registers nothing. The JSON files are read at build time for
//! their string `name`; a file that does not parse or has no such name fails the build.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::{env, fs, io};

use raa_loader::json::{JsonValue, parse};

/// The extension directories, by crate-relative path.
const RUST_DIRS: [&str; 3] = ["src/nodes", "src/exec_ext", "src/frozen"];
const JSON_DIRS: [&str; 2] = ["graphs", "presets"];

fn main() {
    for path in [
        "src/exec/area_model.wgsl",
        "src/exec_ext/compute/learned/pilot.wgsl",
    ] {
        println!("cargo:rerun-if-changed={path}");
        let source = fs::read_to_string(path).unwrap_or_else(|error| panic!("{path}: {error}"));
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|error| panic!("{path}: {}", error.emit_to_string(&source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{path}: {error:?}"));
    }
    println!("cargo:rerun-if-changed=build.rs");
    for dir in RUST_DIRS.iter().chain(JSON_DIRS.iter()) {
        println!("cargo:rerun-if-changed={dir}");
    }
    println!("cargo:rerun-if-changed=src/exec.rs");
    println!("cargo::rustc-check-cfg=cfg(raa_graph_exec)");

    let manifest =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
    let out = PathBuf::from(env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
    let has_exec = manifest.join("src").join("exec.rs").is_file();
    if has_exec {
        println!("cargo:rustc-cfg=raa_graph_exec");
    }

    let nodes = rust_files(&manifest, RUST_DIRS[0]);
    let exec_ext = rust_files(&manifest, RUST_DIRS[1]);
    let frozen = rust_files(&manifest, RUST_DIRS[2]);
    let starters = json_files(&manifest, JSON_DIRS[0]);
    let presets = json_files(&manifest, JSON_DIRS[1]);
    // The directory stamps catch added and removed files; these catch a changed `name`.
    for file in starters.iter().chain(presets.iter()) {
        println!("cargo:rerun-if-changed={}", file.path.display());
    }

    let source = registry_source(&nodes, &exec_ext, &frozen, has_exec, &starters, &presets);
    let target = out.join("registry.rs");
    if let Err(e) = fs::write(&target, source) {
        panic!("raa-graph build.rs: cannot write {}: {e}", target.display());
    }
}

/// A Rust extension file: its module stem and absolute path.
struct RustFile {
    stem: String,
    path: PathBuf,
}

/// A JSON file: its stem, absolute path and the string `name` field.
struct JsonFile {
    stem: String,
    path: PathBuf,
    name: String,
}

/// `(stem, absolute path)` of every `*.rs` file of `rel`, sorted by stem in byte order.
fn rust_files(manifest: &Path, rel: &str) -> Vec<RustFile> {
    let dir = manifest.join(rel);
    let entries = match fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Vec::new(),
        Err(e) => panic!("raa-graph build.rs: cannot list {rel}: {e}"),
    };
    let mut files = Vec::new();
    for entry in entries {
        let path = entry
            .unwrap_or_else(|e| panic!("raa-graph build.rs: cannot read an entry of {rel}: {e}"))
            .path();
        if !path.is_file() || path.extension().and_then(|x| x.to_str()) != Some("rs") {
            continue;
        }
        let stem = match path.file_stem().and_then(|s| s.to_str()) {
            Some(stem) => stem.to_string(),
            None => panic!("raa-graph build.rs: non UTF-8 file name in {rel}"),
        };
        let absolute = absolute(&path, &format!("{rel}/{stem}.rs"));
        files.push(RustFile {
            stem,
            path: absolute,
        });
    }
    files.sort_by(|a, b| a.stem.cmp(&b.stem));
    files
}

/// The JSON files of `rel` with their `name` field, sorted by stem in byte order.
fn json_files(manifest: &Path, rel: &str) -> Vec<JsonFile> {
    let dir = manifest.join(rel);
    let entries = match fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Vec::new(),
        Err(e) => panic!("raa-graph build.rs: cannot list {rel}: {e}"),
    };
    let mut files = Vec::new();
    for entry in entries {
        let path = entry
            .unwrap_or_else(|e| panic!("raa-graph build.rs: cannot read an entry of {rel}: {e}"))
            .path();
        if !path.is_file() || path.extension().and_then(|x| x.to_str()) != Some("json") {
            continue;
        }
        let stem = match path.file_stem().and_then(|s| s.to_str()) {
            Some(stem) => stem.to_string(),
            None => panic!("raa-graph build.rs: non UTF-8 file name in {rel}"),
        };
        let name = format!("{rel}/{stem}.json");
        let text = match fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) => panic!("raa-graph build.rs: cannot read {name}: {e}"),
        };
        let value = match parse(&text) {
            Ok(value) => value,
            Err(e) => panic!("raa-graph build.rs: {name} is not JSON: {e}"),
        };
        let JsonValue::Object(fields) = value else {
            panic!("raa-graph build.rs: {name} is not a JSON object");
        };
        let Some(JsonValue::String(label)) = fields
            .iter()
            .find(|(key, _)| key == "name")
            .map(|(_, value)| value)
        else {
            panic!("raa-graph build.rs: {name} has no string `name`");
        };
        files.push(JsonFile {
            stem,
            path: absolute(&path, &name),
            name: label.clone(),
        });
    }
    files.sort_by(|a, b| a.stem.cmp(&b.stem));
    files
}

/// The absolute path of `path`, named by `rel` when canonicalisation fails.
fn absolute(path: &Path, rel: &str) -> PathBuf {
    match fs::canonicalize(path) {
        Ok(absolute) => absolute,
        Err(e) => panic!("raa-graph build.rs: cannot resolve {rel}: {e}"),
    }
}

/// The Rust source of the registry: modules plus the discovery functions and tables.
fn registry_source(
    nodes: &[RustFile],
    exec_ext: &[RustFile],
    frozen: &[RustFile],
    has_exec: bool,
    starters: &[JsonFile],
    presets: &[JsonFile],
) -> String {
    let mut s = String::from(
        "// @generated by crates/raa-graph/build.rs from the extension directories; do not edit.\n\n",
    );

    for file in nodes.iter().chain(frozen.iter()) {
        let _ = writeln!(s, "#[path = {:?}]\nmod {};\n", path_str(file), file.stem);
    }
    for file in exec_ext {
        let _ = writeln!(
            s,
            "#[cfg(raa_graph_exec)]\n#[path = {:?}]\nmod exec_{};\n",
            path_str(file),
            file.stem
        );
    }

    // Handoff: `families_all()` returns `<stem>::family()` of every `src/nodes/*.rs`, in
    // C-locale stem order; card 'raa-graph core 4/5: catalog functions' calls it in
    // `Catalog::build` together with `STARTERS`, `PRESETS` and `frozen_names()` (src/lib.rs),
    // and card 'raa-graph validate 2/2: inputs, cycles, wiring, availability and budget'
    // calls it to run the `validate_node` hooks in registry order.
    s.push_str("/// One family per discovered node file, in file-stem order.\n");
    s.push_str("pub fn families_all() -> Vec<Box<dyn crate::catalog::NodeFamily>> {\n");
    if nodes.is_empty() {
        s.push_str("    Vec::new()\n");
    } else {
        s.push_str("    vec![\n");
        for file in nodes {
            let _ = writeln!(s, "        {}::family(),", file.stem);
        }
        s.push_str("    ]\n");
    }
    s.push_str("}\n\n");

    s.push_str("/// One (name, plan, source) tuple per frozen pipeline, in file-stem order.\n");
    s.push_str("#[allow(clippy::type_complexity)]\n");
    s.push_str(
        "pub fn frozen_all() -> Vec<(&'static str, fn() -> crate::plan::GraphPlan, &'static str)> {\n",
    );
    if frozen.is_empty() {
        s.push_str("    Vec::new()\n");
    } else {
        s.push_str("    vec![\n");
        for file in frozen {
            let _ = writeln!(
                s,
                "        ({:?}, {}::plan, {}::SOURCE_JSON),",
                file.stem, file.stem, file.stem
            );
        }
        s.push_str("    ]\n");
    }
    s.push_str("}\n\n");

    json_table(&mut s, "STARTERS", "starter graphs", starters);
    json_table(&mut s, "PRESETS", "compute presets", presets);

    if has_exec {
        s.push_str(
            "/// One extension per discovered executor extension file, in file-stem order.\n",
        );
        s.push_str("pub fn extensions_all() -> Vec<Box<dyn crate::exec::ExecExtension>> {\n");
        if exec_ext.is_empty() {
            s.push_str("    Vec::new()\n");
        } else {
            s.push_str("    vec![\n");
            for file in exec_ext {
                let _ = writeln!(s, "        exec_{}::extension(),", file.stem);
            }
            s.push_str("    ]\n");
        }
        s.push_str("}\n");
    }

    s
}

/// The absolute path of a Rust file as a UTF-8 string.
fn path_str(file: &RustFile) -> &str {
    match file.path.to_str() {
        Some(path) => path,
        None => panic!("raa-graph build.rs: non UTF-8 path {}", file.path.display()),
    }
}

/// Appends `pub const <name>: &[(&str, &str, &str)]` with one `(name, stem, json)` tuple.
fn json_table(s: &mut String, name: &str, what: &str, files: &[JsonFile]) {
    let _ = writeln!(
        s,
        "/// The {what}: (JSON `name`, file stem, file text), in file-stem order."
    );
    if files.is_empty() {
        let _ = writeln!(s, "pub const {name}: &[(&str, &str, &str)] = &[];\n");
        return;
    }
    let _ = writeln!(s, "pub const {name}: &[(&str, &str, &str)] = &[");
    for file in files {
        let path = match file.path.to_str() {
            Some(path) => path,
            None => panic!("raa-graph build.rs: non UTF-8 path {}", file.path.display()),
        };
        let _ = writeln!(
            s,
            "    ({:?}, {:?}, include_str!({path:?})),",
            file.name, file.stem
        );
    }
    s.push_str("];\n\n");
}
