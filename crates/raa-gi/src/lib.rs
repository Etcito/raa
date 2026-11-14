//! Global illumination of RAA (module m30, S36-S39): the crate root with the S36 settings
//! ([`GiSettings`], [`GiSettingsPatch`], [`GiQuality`], [`GiQualityPatch`], [`DebugView`]), the
//! `gi.*` refusal codes ([`error`]) and the finest-voxel constant. Later cards add the clipmap
//! volume, the CPU twins, membership, stats, the part seam and the feature handle.
//! Adapted solstice parts name their source in their module docs; see README.md.

pub mod clipmap;
pub mod error;
pub mod settings;

pub use settings::{
    DebugView, FINEST_VOXEL_METRES, GiQuality, GiQualityPatch, GiSettings, GiSettingsPatch,
};
