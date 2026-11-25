//! Executor extension `ddgi`: configures the DDGI volume from the plan's `pass.gi.ddgi` node and
//! sets the forward frame feature that selects its lookup. The node's pipeline entry is enabled by
//! the plan like any pass; `clear` and a plan without the node turn the feature off.

use raa_sdf_trace::mesh::{DDGI_FRAME_FEATURE, DdgiSettings};

use crate::error::{Issue, Severity, Stage};
use crate::exec::{ExecExtension, FeatureHandles};
use crate::plan::GraphPlan;
use crate::registry::ddgi::{DDGI_TYPE, settings_from_params};

pub fn extension() -> Box<dyn ExecExtension> {
    Box::new(DdgiExtension { active: false })
}

struct DdgiExtension {
    active: bool,
}

impl DdgiExtension {
    fn set(&mut self, h: &mut FeatureHandles<'_>, settings: DdgiSettings) -> Result<(), String> {
        if let Some(ddgi) = h.handles.ddgi.as_ref() {
            ddgi.borrow_mut()
                .configure(settings)
                .map_err(|error| error.message().to_owned())?;
        }
        if let Some(lit) = h.lit.as_deref_mut() {
            lit.set_frame_feature(DDGI_FRAME_FEATURE, settings.enabled);
        }
        self.active = settings.enabled;
        Ok(())
    }
}

impl ExecExtension for DdgiExtension {
    fn name(&self) -> &'static str {
        "ddgi"
    }

    fn apply(
        &mut self,
        plan: &GraphPlan,
        _previous: Option<&GraphPlan>,
        h: &mut FeatureHandles<'_>,
    ) -> Result<(), Vec<Issue>> {
        let Some(node) = plan.nodes.iter().find(|node| node.type_id == DDGI_TYPE) else {
            if self.active {
                self.clear(h);
            }
            return Ok(());
        };
        if h.handles.ddgi.is_none() {
            return Err(vec![Issue {
                code: "graph.unavailable",
                severity: Severity::Error,
                stage: Stage::Apply,
                node: Some(node.id.clone()),
                port: None,
                param: None,
                link: None,
                message: "The DDGI feature is unavailable in this engine.".into(),
                fix: "Install the DDGI feature before applying this graph.".into(),
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
                    fix: "Use at most 32768 probes, 2 to 128 per axis.".into(),
                }]
            })
    }

    fn clear(&mut self, h: &mut FeatureHandles<'_>) {
        let settings = h
            .handles
            .ddgi
            .as_ref()
            .map(|ddgi| ddgi.borrow().settings)
            .unwrap_or_default();
        let _ = self.set(
            h,
            DdgiSettings {
                enabled: false,
                ..settings
            },
        );
    }

    /// Keeps the frame feature in step with the volume while a graph is active (other setters
    /// rewrite the frame flags between frames).
    fn frame(&mut self, h: &mut FeatureHandles<'_>) -> Result<(), raa_core::RaaError> {
        if let Some(lit) = h.lit.as_deref_mut()
            && lit.frame_feature(DDGI_FRAME_FEATURE) != self.active
        {
            lit.set_frame_feature(DDGI_FRAME_FEATURE, self.active);
        }
        Ok(())
    }
}
