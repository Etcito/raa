//! Virtualized geometry of RAA (S31, S32, module m29): the crate root with the settings
//! ([`VgSettings`], [`VgSettingsPatch`], [`DebugView`]), the `vg.*` refusal codes ([`error`])
//! and the path eligibility of an instance ([`eligibility`]). Virtualized instances are drawn
//! through clusters from raa-cluster; skinned, deformed, transparent and fur content stays on
//! the forward path. The disabled state records no pass and allocates no resource.
//! Adapted solstice parts name their source in their module docs; see README.md.

mod frame_passes;

pub mod culling;
pub mod eligibility;
pub mod error;
pub mod feature;
pub mod gbuffer;
pub mod hzb;
pub mod raster;
pub mod resolve;
pub mod scene;
pub mod settings;
pub mod shade;
pub mod shadow;
pub mod stats;

pub use culling::{
    CASCADE_BASE, GI_BASE, GI_SLOTS, InstanceFilter, MAIN_FIRST, MAIN_SECOND, PASS_VIEW_SIZE,
    PassView, SLOT_STRIDE, VG_VIEW_SLOTS, ViewSlots,
};
pub use eligibility::{InstancePath, refusal_message};
pub use feature::{VgFeature, VgHandle, VgStats, inactive_reason};
pub use hzb::{Hzb, hzb_down_module_wgsl, hzb_seed_module_wgsl, hzb_size};
pub use raster::{RASTER_COMMON_WGSL, VIS_WRITE_WGSL, raster_hw_module_wgsl};
pub use resolve::resolve_module_wgsl;
pub use scene::{
    INSTANCE_CASTS_SHADOW, INSTANCE_DOUBLE_SIDED, INSTANCE_MOVABLE, INSTANCE_PREVIOUS_VALID,
    InstanceRequests, MeshKeys, SectionRange, VG_INSTANCE_SIZE, VgInstanceRecord, VgScene,
    hierarchy_word, write_instance,
};
pub use settings::{DebugView, VgSettings, VgSettingsPatch};
pub use shadow::{CascadeWork, cascade_views, record_cascades};
