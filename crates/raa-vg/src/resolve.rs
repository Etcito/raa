//! The S32 resolve module of RAA's virtualized geometry: the WGSL module builder for
//! `vg.resolve`. The `VgSurface` outputs, the `VgFrameGpu` writer and the `Resolve` pipeline
//! land with their own cards; this file holds only the builder. Adapted from solstice (MIT),
//! Copyright (c) 2026 SamG-Coder: `crates/solstice-render/shaders/vg/resolve.wgsl`, revision
//! 7e465b5. The wetness include, the waterline bindings, the material texture tiers and the
//! emissive output are removed.

use crate::culling::PASS_VIEW_WGSL;

const RESOLVE_WGSL: &str = include_str!("wgsl/resolve.wgsl");

/// The resolve module: `raa_shader::DEPTH_WGSL` + [`PASS_VIEW_WGSL`] + `resolve.wgsl`, with
/// entry point `main` at `@workgroup_size(8, 8)` and group-2 bindings 0-12 (a 320 B `VgFrame`
/// uniform, the eight read-only scene and visibility buffers, then the four write storage
/// textures). Handoff to the `VgSurface`/`Resolve` card: build the compute pipeline from
/// `main` and bind that list.
pub fn resolve_module_wgsl() -> String {
    [raa_shader::DEPTH_WGSL, PASS_VIEW_WGSL, RESOLVE_WGSL].concat()
}
