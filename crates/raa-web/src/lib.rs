//! Shared web runtime types and frame statistics.
//!
//! On every target: [`FrameInput`], [`FrameCounters`], [`FrameStats`], the [`WebApp`] trait
//! and the [`stats`] helpers. On wasm32 only: [`WebRuntime`], [`RuntimeOptions`] and the
//! JS conversions [`stats_to_js`] and [`error_to_js`].

use raa_core::{FrameGraph, GpuContext, RaaError};

#[cfg(target_arch = "wasm32")]
mod js;
#[cfg(target_arch = "wasm32")]
mod runtime;
pub mod stats;

/// Information about the frame being updated.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameInput {
    pub frame: u64,
    pub now_ms: f64,
    pub dt_ms: f64,
    pub size: (u32, u32),
}

/// Work counters reported by an application for a frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameCounters {
    pub instances: u32,
    pub draw_calls: u32,
    pub triangles: u64,
}

/// An application driven by the web runtime. It lives here (not in the wasm-only runtime) so
/// that applications also compile, and can be unit-tested, on the host.
pub trait WebApp {
    /// Adds the application's targets and passes to `graph`, which is compiled afterwards.
    fn init(
        &mut self,
        gpu: &GpuContext,
        graph: &mut FrameGraph,
        size: (u32, u32),
        surface_format: wgpu::TextureFormat,
        surface_is_srgb: bool,
    ) -> Result<(), RaaError>;

    /// CPU work and uploads for one frame, before the graph executes.
    fn update(&mut self, gpu: &GpuContext, input: &FrameInput) -> Result<FrameCounters, RaaError>;

    /// Called after the surface has been resized. Default: nothing.
    fn resize(&mut self, _gpu: &GpuContext, _size: (u32, u32)) {}
}

#[cfg(target_arch = "wasm32")]
pub use js::{error_to_js, stats_to_js};
#[cfg(target_arch = "wasm32")]
pub use runtime::{RuntimeOptions, WebRuntime};
pub use stats::FrameStats;

/// Quantises a linear colour to the 8-bit RGBA a canvas stores: sRGB-encoded when the
/// canvas format is `*Srgb`, as is (linear) otherwise. Alpha is never encoded.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn quantise_color(color: wgpu::Color, srgb: bool) -> [u8; 4] {
    let encode = |linear: f64| {
        let c = linear.clamp(0.0, 1.0);
        if !srgb {
            c
        } else if c <= 0.003_130_8 {
            c * 12.92
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        }
    };
    // The values are clamped to [0, 1], so the rounded products fit in u8.
    let unorm8 = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    [
        unorm8(encode(color.r)),
        unorm8(encode(color.g)),
        unorm8(encode(color.b)),
        unorm8(color.a),
    ]
}

#[cfg(test)]
mod tests {
    use super::quantise_color;

    #[test]
    fn clear_color_quantises_as_stored_on_unorm_canvases() {
        let clear = raa_core::CLEAR_COLOR;
        assert_eq!(quantise_color(clear, false), [13, 15, 20, 255]);
    }

    #[test]
    fn clear_color_is_srgb_encoded_on_srgb_canvases() {
        let clear = raa_core::CLEAR_COLOR;
        assert_eq!(quantise_color(clear, true), [63, 69, 80, 255]);
    }

    #[test]
    fn quantise_clamps_out_of_range_channels() {
        let c = wgpu::Color {
            r: -1.0,
            g: 2.0,
            b: 0.0,
            a: 1.0,
        };
        assert_eq!(quantise_color(c, true), [0, 255, 0, 255]);
    }
}
