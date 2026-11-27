//! Configurable pass pipeline of the lit renderer (S12): surface-sized targets and the
//! HDR ping-pong chain the feature passes read and write, and the group-1 lighting
//! resources (S10) the lit passes bind. `Pipeline` runs the enabled feature passes in order
//! inside one raa-core frame graph pass, `raa.pipeline`.

mod brdf_atlas;
mod env;
mod graph_frame;
mod interlace;
mod lighting;
mod output;
mod pass;
mod pass_timings;
mod pipeline;
mod resource_table;
mod targets;
mod tunable;

pub use brdf_atlas::{
    BRDF_ATLAS_FORMAT, BRDF_ATLAS_HEIGHT, BRDF_ATLAS_WIDTH, BRDF_LUT_SIZE, LTC_TABLE_SIZE,
    LTC1_ORIGIN, LTC2_ORIGIN, create_brdf_atlas, ltc_table_halfs,
};
pub use env::EnvGpu;
pub use graph_frame::GraphFrame;
pub use interlace::{
    INTERLACE_PHASES, InterlaceSignals, IrradianceRate, LightingInterlace, ShadowRate,
};
pub use lighting::{
    AREA_SHADOW_MASK_LAYERS, AREA_STOCHASTIC_GPU_SIZE, LightingResources, LocalShadowSlot,
    MAX_LOCAL_SHADOWS, MAX_STOCHASTIC_LIGHTS,
    WeatherLightingGpu,
};
pub use pass::{FeaturePass, PassInfo, PassRole, PrepareCtx, RecordCtx};
pub use pass_timings::{PassGpuTimingSample, PassGpuTimings};
pub use pipeline::{Pipeline, PipelineConfig, SubmissionState};
pub use resource_table::{
    ImageDesc, ImageDim, PublishedImage, ResourceTable, SampleKind, sample_kind_for,
};
pub use targets::{ActiveRect, HdrChain, Targets};
pub use tunable::{TunableSpec, find_tunable, validate_tunable};

/// Format of the HDR chain, the opaque copy, bloom and the env cube (HDR_FORMAT).
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
