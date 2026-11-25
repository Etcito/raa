//! Executor extension `nrc`: configures the neural radiance cache from the plan's
//! `neural.radiance-cache` node and
//! sets the forward frame feature that selects its lookup. The node's pipeline entry is enabled by
//! the plan like any pass; `clear` and a plan without the node turn the feature off.

use raa_sdf_trace::mesh::{NRC_FRAME_FEATURE, NrcSettings};

use crate::error::{Issue, Severity, Stage};
use crate::exec::{ExecExtension, FeatureHandles};
use crate::plan::GraphPlan;
use crate::registry::nrc::{NRC_TYPE, settings_from_params};

pub fn extension() -> Box<dyn ExecExtension> {
    Box::new(NrcExtension { active: false })
}

struct NrcExtension {
    active: bool,
}

impl NrcExtension {
    fn set(&mut self, h: &mut FeatureHandles<'_>, settings: NrcSettings) -> Result<(), String> {
        if let Some(nrc) = h.handles.nrc.as_ref() {
            nrc.borrow_mut()
                .configure(settings)
                .map_err(|error| error.message().to_owned())?;
        }
        if let Some(lit) = h.lit.as_deref_mut() {
            lit.set_frame_feature(NRC_FRAME_FEATURE, settings.enabled);
        }
        self.active = settings.enabled;
        Ok(())
    }
}

impl ExecExtension for NrcExtension {
    fn name(&self) -> &'static str {
        "nrc"
    }

    fn apply(
        &mut self,
        plan: &GraphPlan,
        _previous: Option<&GraphPlan>,
        h: &mut FeatureHandles<'_>,
    ) -> Result<(), Vec<Issue>> {
        let Some(node) = plan.nodes.iter().find(|node| node.type_id == NRC_TYPE) else {
            if self.active {
                self.clear(h);
            }
            return Ok(());
        };
        if h.handles.nrc.is_none() {
            return Err(vec![Issue {
                code: "graph.unavailable",
                severity: Severity::Error,
                stage: Stage::Apply,
                node: Some(node.id.clone()),
                port: None,
                param: None,
                link: None,
                message: "The neural radiance cache is unavailable in this engine.".into(),
                fix: "Install the neural radiance cache before applying this graph.".into(),
            }]);
        }
        self.set(h, settings_from_params(&node.params))
            .map_err(|message| {
                vec![Issue {
                    code: "graph.param_range",
                    severity: Severity::Error,
                    stage: Stage::Apply,
                    node: Some(node.id.clone()),
                    port: None,
                    param: None,
                    link: None,
                    message,
                    fix: "Use 1 to 5 hidden layers and the listed sample counts.".into(),
                }]
            })
    }

    fn clear(&mut self, h: &mut FeatureHandles<'_>) {
        let settings = h
            .handles
            .nrc
            .as_ref()
            .map(|nrc| nrc.borrow().settings)
            .unwrap_or_default();
        let _ = self.set(
            h,
            NrcSettings {
                enabled: false,
                ..settings
            },
        );
    }

    /// Keeps the frame feature in step with the cache while a graph is active (other setters
    /// rewrite the frame flags between frames).
    fn frame(&mut self, h: &mut FeatureHandles<'_>) -> Result<(), raa_core::RaaError> {
        if let Some(lit) = h.lit.as_deref_mut()
            && lit.frame_feature(NRC_FRAME_FEATURE) != self.active
        {
            lit.set_frame_feature(NRC_FRAME_FEATURE, self.active);
        }
        Ok(())
    }
}
