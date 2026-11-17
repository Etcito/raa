//! Sharpening settings (S47 `SharpenDesc`) and their validation (`post.invalid`).

use raa_core::RaaError;

/// Sharpening settings. The JS keys are `enabled` and `sharpness`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SharpenDesc {
    /// Whether the `sharpen` pass runs; off by default.
    pub enabled: bool,
    /// Sharpening amount, finite in [0, 1]; the default 0.9 is the solstice value.
    pub sharpness: f32,
}

impl Default for SharpenDesc {
    fn default() -> Self {
        SharpenDesc {
            enabled: false,
            sharpness: 0.9,
        }
    }
}

impl SharpenDesc {
    /// Refuses with code post.invalid when `sharpness` is not finite or lies outside 0..=1;
    /// returns Ok and changes nothing otherwise.
    ///
    /// Handoff: raa-sharpen 3/3: sharpen pass, install and SharpenHandle (its
    /// `SharpenHandle::set` calls this before it stores the desc) and api-m34: setSharpening and
    /// sharpening on the Engine (its `merge_sharpening` calls this on the merged desc). Today
    /// `SharpenDesc::validate(&self) -> Result<(), RaaError>` is a pure check of
    /// `SharpenDesc.sharpness`, with no state; a refusal is
    /// `RaaError::FrameGraph { code: "post.invalid", message }`.
    pub fn validate(&self) -> Result<(), RaaError> {
        let v = self.sharpness;
        if v.is_finite() && (0.0..=1.0).contains(&v) {
            return Ok(());
        }
        Err(RaaError::FrameGraph {
            code: "post.invalid",
            message: format!("`sharpness` must be in [0, 1], got {v}"),
        })
    }
}
