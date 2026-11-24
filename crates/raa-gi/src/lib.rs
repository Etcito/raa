//! Global illumination of RAA (module m30, S36-S39): the crate root with the S36 settings
//! ([`GiSettings`], [`GiSettingsPatch`], [`GiQuality`], [`GiQualityPatch`], [`DebugView`]), the
//! `gi.*` refusal codes ([`error`]) and the finest-voxel constant. Later cards add the clipmap
//! volume, the CPU twins, membership, stats, the part seam and the feature handle.
//! Adapted solstice parts name their source in their module docs; see README.md.

pub mod clipmap;
pub mod cpu;
mod debug;
pub mod emissive;
pub mod error;
pub mod feature;
pub mod forward_driver;
pub mod layouts;
pub mod membership;
pub mod parts;
pub mod settings;
pub mod shaders;
pub mod stages;
pub mod stats;
mod tunables;
pub mod vertex_colors;
mod volume;

pub use raa_sdf_trace::mesh::spherical;
pub use stats::GiStats;

pub use emissive::EmissiveSources;

pub use tunables::{GiTunables, tunables};

pub use parts::ssgi::ssgi_counts;

pub use vertex_colors::GiVertexColors;

pub use settings::{
    DebugView, FINEST_VOXEL_METRES, GiPreset, GiQuality, GiQualityPatch, GiSettings,
    GiSettingsPatch, IndirectMode, SsgiQuality,
};

pub use membership::{Driver, InstanceGi};

pub use feature::{GiFeature, GiHandle, inactive_reason};
