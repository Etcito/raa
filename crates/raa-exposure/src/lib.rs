//! Automatic exposure from a luminance histogram (pass `exposure.auto`, S46). This share holds
//! the S46 constants and the [`AutoExposureDesc`] settings with their validation (`post.invalid`).
//! Later shares of this crate add the maths, the shaders, the readback and the feature.

mod desc;
mod maths;

pub use desc::AutoExposureDesc;
pub use maths::{adapt, band_average, bin_index, exposure_from_average, luminance, settled};

/// Name of the feature pass.
pub const AUTO_EXPOSURE_PASS_NAME: &str = "exposure.auto";
/// Order of the pass in the frame graph: after bloom (120), before grading (130).
pub const AUTO_EXPOSURE_ORDER: u32 = 125;
/// Histogram bins over the log2 luminance range.
pub const HISTOGRAM_BINS: u32 = 128;
/// Log2 luminance of the first bin.
pub const MIN_LOG_LUMINANCE: f32 = -8.0;
/// Log2 span covered by the bins, from `MIN_LOG_LUMINANCE` upwards.
pub const LOG_LUMINANCE_RANGE: f32 = 18.0;
/// The histogram samples every 4th pixel in x and y.
pub const SAMPLE_STRIDE: u32 = 4;
/// Histogram compute workgroup size, 16 x 16.
pub const HISTOGRAM_WORKGROUP: u32 = 16;
/// Luminance below this is not counted in the histogram.
pub const LUMINANCE_FLOOR: f32 = 1e-5;
/// Middle grey that the average luminance is mapped to.
pub const MIDDLE_GREY: f32 = 0.18;
/// Settled threshold: two exposure results closer than this many EV count as settled.
pub const SETTLED_EV: f32 = 0.01;
/// Bytes of the `ExposureParams` uniform.
pub const PARAMS_SIZE: usize = 48;
/// Bytes of the `ExposureState` record, which is also the readback record.
pub const RESULT_SIZE: usize = 16;
/// Readback slots in the pool.
pub const READBACK_SLOTS: usize = 3;
