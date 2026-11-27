//! Executor extension `local_shadow`: keeps the `shadow.local` pass (cached cube shadows of the
//! local lights with `shadows`, raa-sdf-trace mesh/local_shadow.rs) enabled under every plan. It
//! has no node: it is a property of the lights, not of the graph, and it traces nothing (and binds
//! the dummy) while no light casts shadows or no BVH-consuming GI node provides trace geometry.
//! `apply_pipeline` disables entries a plan does not name and orders them by their order value
//! (36, after `light.tiles` and before the forward), so this extension only re-enables it.

use raa_sdf_trace::mesh::LOCAL_SHADOW_NAME;

use crate::error::{Issue, Severity, Stage};
use crate::exec::{ExecExtension, FeatureHandles};
use crate::plan::GraphPlan;

pub fn extension() -> Box<dyn ExecExtension> {
    Box::new(LocalShadowExtension)
}

struct LocalShadowExtension;

impl ExecExtension for LocalShadowExtension {
    fn name(&self) -> &'static str {
        "local_shadow"
    }

    fn apply(
        &mut self,
        _plan: &GraphPlan,
        _previous: Option<&GraphPlan>,
        h: &mut FeatureHandles<'_>,
    ) -> Result<(), Vec<Issue>> {
        let mut pipeline = h.pipeline.borrow_mut();
        if pipeline.entry_enabled(LOCAL_SHADOW_NAME).is_none() {
            return Ok(());
        }
        pipeline
            .set_enabled(LOCAL_SHADOW_NAME, true)
            .map_err(|error| {
                vec![Issue {
                    code: "graph.apply",
                    severity: Severity::Error,
                    stage: Stage::Apply,
                    node: None,
                    port: None,
                    param: None,
                    link: None,
                    message: error.message().to_owned(),
                    fix: "Reinstall the engine's local shadow pass.".into(),
                }]
            })
    }

    fn clear(&mut self, _h: &mut FeatureHandles<'_>) {}
}
