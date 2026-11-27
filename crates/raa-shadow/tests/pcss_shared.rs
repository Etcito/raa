//! Contract `shadow-pcss-kernel`: the `csm` Shadow provider builds on the prelude's shared
//! PCSS kernel instead of defining its own copy of the Poisson table and rotation helpers.

// The provider is crate-private; the test compiles its source file.
#[path = "../src/provider.rs"]
mod provider;

#[test]
fn csm_provider_calls_the_shared_kernel() {
    let wgsl = provider::csm_provider().wgsl;
    assert!(
        wgsl.contains("raa_shadow_blockers(i32(cascade_idx), uv, clip.z, search_texels, res)"),
        "{wgsl}"
    );
    assert!(
        wgsl.contains(
            "raa_shadow_pcf(i32(cascade_idx), uv, clip.z, penumbra_texels, res, raa_csm_frame_rotation(frag.xy))"
        ),
        "{wgsl}"
    );
    // The jitter-turned rotation builds on the prelude's per-pixel rotation.
    assert!(wgsl.contains("return raa_csm_rotation(p) + turn"), "{wgsl}");
    assert!(!wgsl.contains("const RAA_CSM_POISSON"), "{wgsl}");
    assert!(!wgsl.contains("fn raa_csm_rotation("), "{wgsl}");
}
