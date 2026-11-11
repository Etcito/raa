//! Discover the declared M2 cloud backend source (`src/clouds/mod.rs`).
//!
//! The M1 registry always supplies the completed neutral/legacy backend. The M2
//! producer file is compiled and registered only when it exists: an absent M2
//! source is never referenced by a compiled module. `src/clouds/**` stays M2's
//! directory; this build script only observes whether its declared entry point is
//! present and never writes it.
//!
//! The generated `cloud_backend_registry.rs` declares the optional module only when present, so the crate root includes it without naming a
//! module that may not exist.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::{env, fs};

/// The declared M2 entry point, relative to the crate manifest.
const M2_SOURCE: &str = "src/clouds/mod.rs";

/// Generate both module discovery and registration from the same candidate set.
pub fn generate_registry(source: &std::path::Path, candidates: &[&str]) -> Result<String, String> {
    let mut seen = std::collections::BTreeSet::new();
    let mut registry = String::new();
    let mut registrations = String::new();
    for &name in candidates {
        if !seen.insert(name) {
            return Err(format!("duplicate cloud backend `{name}`"));
        }
        match name {
            "neutral" => registrations
                .push_str("registry.register(NEUTRAL_BACKEND, NeutralCloudRenderer::create)?;\n"),
            "m2" => {
                writeln!(registry, "#[path = {source:?}] pub(crate) mod clouds;").unwrap();
                registrations.push_str("registry.register(M2_BACKEND, clouds::create)?;\n");
            }
            _ => return Err(format!("unknown cloud backend `{name}`")),
        }
    }
    if !seen.contains("neutral") {
        return Err("neutral cloud backend is required".into());
    }
    writeln!(registry, "fn register_builtin(registry: &mut CloudBackendRegistry) -> Result<(), RaaError> {{\n{registrations}Ok(())\n}}").unwrap();
    Ok(registry)
}

pub fn registry_for_presence(source: &std::path::Path, present: bool) -> String {
    let candidates = if present {
        &["neutral", "m2"][..]
    } else {
        &["neutral"][..]
    };
    generate_registry(source, candidates).expect("unique declared cloud backends")
}

pub fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={M2_SOURCE}");
    println!("cargo::rustc-check-cfg=cfg(raa_m2_clouds)");
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("cargo manifest directory"));
    let source = manifest.join(M2_SOURCE);
    let present = source.is_file();
    if present {
        println!("cargo:rustc-cfg=raa_m2_clouds");
    }
    let out = PathBuf::from(env::var("OUT_DIR").expect("cargo output directory"));
    fs::write(
        out.join("cloud_backend_registry.rs"),
        registry_for_presence(&source, present),
    )
    .expect("write cloud backend registry");
}
