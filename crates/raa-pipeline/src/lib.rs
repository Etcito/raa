//! Configurable pass pipeline of the lit renderer (S12): surface-sized targets and the
//! HDR ping-pong chain the feature passes read and write, and the group-1 lighting
//! resources (S10) the lit passes bind. `Pipeline` runs the enabled feature passes in order
//! inside one raa-core frame graph pass, `raa.pipeline`.

mod env;
mod lighting;
mod pass;
mod pipeline;
mod targets;

pub use env::EnvGpu;
pub use lighting::LightingResources;
pub use pass::{FeaturePass, PassInfo, PassRole, PrepareCtx, RecordCtx};
pub use pipeline::{Pipeline, PipelineConfig};
pub use targets::{HdrChain, Targets};

/// Format of the HDR chain, the opaque copy, bloom and the env cube (HDR_FORMAT).
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
