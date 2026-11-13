//! Virtualized geometry of RAA (S31, S32, module m29): the crate root with the settings
//! ([`VgSettings`], [`VgSettingsPatch`], [`DebugView`]), the `vg.*` refusal codes ([`error`])
//! and the path eligibility of an instance ([`eligibility`]). Virtualized instances are drawn
//! through clusters from raa-cluster; skinned, deformed, transparent and fur content stays on
//! the forward path. The disabled state records no pass and allocates no resource.
//! Adapted solstice parts name their source in their module docs; see README.md.

pub mod eligibility;
pub mod error;
pub mod settings;

pub use eligibility::{InstancePath, refusal_message};
pub use settings::{DebugView, VgSettings, VgSettingsPatch};
