//! Executor extension `wiring` (S57, module raa-graph-wiring). This share holds the device-free
//! pooling and history functions of `wiring/plan_fns.rs` and the extension skeleton, which applies
//! nothing yet: the graph's behaviour is unchanged by this file.

use crate::catalog::Catalog;
use crate::error::Issue;
use crate::exec::{ExecExtension, FeatureHandles};
use crate::plan::{GraphPlan, PlanHistory, PlanNode, PlanPortRef, PlanResource};

#[allow(dead_code)] // the wiring cards call these; the tests compile the same file for themselves
#[path = "wiring/plan_fns.rs"]
mod plan_fns;

// Re-exported for the wiring cards; nothing in this share calls them yet.
#[allow(unused_imports)]
pub use plan_fns::{history_resets, pool_resources};

/// The extension, discovered by build.rs (stem `wiring`, registered after `views`).
pub fn extension() -> Box<dyn ExecExtension> {
    Box::new(WiringExtension::new())
}

/// Handoff: `WiringExtension` is taken over by card 'Wiring 3/5: colour-chain graph frame, resource
/// pool and order from links', which extends `WiringExtension::apply` and `WiringExtension::clear`.
/// Today `apply` returns `Ok(())` and `clear` does nothing, so the extension changes no graph; the
/// struct holds only the `Catalog::build()` result of `WiringExtension::new` in its field `catalog`.
pub(crate) struct WiringExtension {
    #[allow(dead_code)] // read by the colour-chain frame of the next card
    catalog: Catalog,
}

impl WiringExtension {
    /// The extension with its own catalog.
    fn new() -> WiringExtension {
        WiringExtension {
            catalog: Catalog::build(),
        }
    }
}

impl ExecExtension for WiringExtension {
    fn name(&self) -> &'static str {
        "wiring"
    }

    fn apply(
        &mut self,
        _plan: &GraphPlan,
        _previous: Option<&GraphPlan>,
        _h: &mut FeatureHandles<'_>,
    ) -> Result<(), Vec<Issue>> {
        Ok(())
    }

    fn clear(&mut self, _h: &mut FeatureHandles<'_>) {}
}
