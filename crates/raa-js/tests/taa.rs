//! Text pins and `fn`-pointer type pins for the m31 TAA wiring. The api functions stay
//! crate-private; this test never calls them, it pins the source text and the two public
//! signatures of the upstream crates.

use std::fs;
use std::path::Path;

fn out_file(name: &str) -> String {
    let path = Path::new(env!("OUT_DIR")).join(name);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

#[test]
fn install_file_pins_the_taa_install() {
    let src = include_str!("../src/api/m31_taa.rs");
    for needle in [
        "raa_taa::TaaFeature::install",
        "frame_buffer",
        "ctx.features.insert",
    ] {
        assert!(src.contains(needle), "m31_taa.rs lacks `{needle}`");
    }
}

#[test]
fn pipeline_pins_the_per_view_history() {
    let src = include_str!("../src/api/m07_pipeline.rs");
    for needle in [
        "taa: raa_taa::TaaHandle",
        "raa_taa::TaaFeature::install",
        "post.taa.invalidate",
    ] {
        assert!(src.contains(needle), "m07_pipeline.rs lacks `{needle}`");
    }
}

#[test]
fn registry_pins_the_taa_module() {
    let reg = out_file("api_registry.rs");
    for needle in ["mod m31_taa;", "m31_taa::install", "m31_taa::frame"] {
        assert!(reg.contains(needle), "api_registry.rs lacks `{needle}`");
    }
}

#[test]
#[allow(clippy::type_complexity)]
fn upstream_signatures_match() {
    let _: fn(
        &raa_core::GpuContext,
        &std::rc::Rc<std::cell::RefCell<raa_pipeline::Pipeline>>,
        &wgpu::Buffer,
    ) -> Result<raa_taa::TaaHandle, raa_core::RaaError> = raa_taa::TaaFeature::install;
    let _: fn(&raa_lit::LitHandle) -> wgpu::Buffer = raa_lit::LitHandle::frame_buffer;
}
