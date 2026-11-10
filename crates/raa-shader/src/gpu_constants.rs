//! Shared GPU sizes and logical feature addressing.
//!
//! Handoff: `pub const fn feature_address(id: u32) -> Option<(usize, u32)>` in
//! `crates/raa-shader/src/gpu_constants.rs` is re-exported by `raa-shader`; the frame
//! writer uses it to attach logical feature ids to `FrameGpu.flags` words and bits.
//! The generated `raa_feature_enabled(id: u32) -> bool` in `PRELUDE_WGSL` reads the
//! same `raa_frame.flags` registry; id 24 is tiling and 25..=38 are reservations.
//!
//! Handoff: `pub const MAX_LOCAL_LIGHTS: usize` and `pub const LIGHT_GPU_SIZE: usize`
//! in `crates/raa-shader/src/gpu_constants.rs` are re-exported by `raa-shader`; the
//! light serializer attaches these to its allocation and record stride. Generated WGSL
//! exposes the same constants for the light uniform array and its indexed records.

include!(concat!(env!("OUT_DIR"), "/gpu_constants.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_constants_values_and_feature_addresses() {
        assert_eq!(
            (FRAME_GPU_SIZE, MAX_LOCAL_LIGHTS, LIGHT_GPU_SIZE),
            (768, 256, 64)
        );
        assert_eq!((TILE_SIZE, TILE_MAX_LIGHTS, TILE_WORDS), (16, 127, 128));
        assert_eq!(WEATHER_LIGHTING_GPU_SIZE, 64);
        for (id, expected) in [
            (0, (2, 0)),
            (24, (2, 24)),
            (31, (2, 31)),
            (32, (3, 0)),
            (38, (3, 6)),
            (63, (3, 31)),
        ] {
            assert_eq!(feature_address(id), Some(expected));
        }
        assert_eq!(feature_address(64), None);
        assert_eq!(feature_address(u32::MAX), None);
    }

    #[test]
    fn gpu_constants_generated_wgsl_is_standalone() {
        let source = crate::PRELUDE_WGSL;
        for declaration in [
            "const FRAME_GPU_SIZE: u32 = 768u;",
            "const MAX_LOCAL_LIGHTS: u32 = 256u;",
            "const LIGHT_GPU_SIZE: u32 = 64u;",
            "const WEATHER_LIGHTING_GPU_SIZE: u32 = 64u;",
            "const TILE_SIZE: u32 = 16u;",
            "const TILE_MAX_LIGHTS: u32 = 127u;",
            "const TILE_WORDS: u32 = 128u;",
        ] {
            assert!(source.contains(declaration));
        }
        let module = naga::front::wgsl::parse_str(source).expect("parse standalone prelude");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("validate standalone prelude");
    }
}
