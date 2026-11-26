//! Executor extension `exposure`: writes the parameters of the plan's `pass.exposure.auto` node to
//! every auto-exposure handle (the main pipeline's and each viewport's). The node's pipeline entry
//! `exposure.auto` is enabled by the plan like any pass; `clear` and a plan without the node put
//! back the descriptions found before the first apply. While a graph with the node is active, `frame`
//! writes the node's description again over any external `setAutoExposure` (a host applying its own
//! auto-exposure preference would otherwise turn the pass-through into an adapting exposure).

use raa_exposure::AutoExposureDesc;

use crate::error::{Issue, Severity, Stage};
use crate::exec::{ExecExtension, FeatureHandles};
use crate::plan::GraphPlan;
use crate::registry::exposure::{EXPOSURE_TYPE, desc_from_params};

pub fn extension() -> Box<dyn ExecExtension> {
    Box::new(ExposureExtension {
        saved: None,
        desired: None,
    })
}

struct ExposureExtension {
    /// The handles' descriptions before the first apply, put back by `clear`.
    saved: Option<Vec<AutoExposureDesc>>,
    /// The description of the active node, kept on every handle by `frame`.
    desired: Option<AutoExposureDesc>,
}

impl ExecExtension for ExposureExtension {
    fn name(&self) -> &'static str {
        "exposure"
    }

    fn apply(
        &mut self,
        plan: &GraphPlan,
        _previous: Option<&GraphPlan>,
        h: &mut FeatureHandles<'_>,
    ) -> Result<(), Vec<Issue>> {
        let Some(node) = plan.nodes.iter().find(|node| node.type_id == EXPOSURE_TYPE) else {
            if self.saved.is_some() {
                self.clear(h);
            }
            return Ok(());
        };
        let issue = |code: &'static str, message: String, fix: &str| Issue {
            code,
            severity: Severity::Error,
            stage: Stage::Apply,
            node: Some(node.id.clone()),
            port: None,
            param: None,
            link: None,
            message,
            fix: fix.into(),
        };
        if h.handles.auto_exposure.is_empty() {
            return Err(vec![issue(
                "graph.unavailable",
                "The automatic exposure feature is unavailable in this engine.".into(),
                "Install the auto-exposure feature before applying this graph.",
            )]);
        }
        let desc = desc_from_params(&node.params);
        desc.validate().map_err(|error| {
            vec![issue(
                "graph.param_range",
                error.message().to_owned(),
                "Keep lowPercent below highPercent and minEv at most maxEv.",
            )]
        })?;
        if self.saved.is_none() {
            self.saved = Some(h.handles.auto_exposure.iter().map(|a| a.get()).collect());
        }
        for handle in &h.handles.auto_exposure {
            handle.set(desc).map_err(|error| {
                vec![issue(
                    "graph.param_range",
                    error.message().to_owned(),
                    "Use finite values in the parameter ranges.",
                )]
            })?;
        }
        self.desired = Some(desc);
        Ok(())
    }

    fn frame(&mut self, h: &mut FeatureHandles<'_>) -> Result<(), raa_core::RaaError> {
        if let Some(desc) = self.desired {
            for handle in &h.handles.auto_exposure {
                if handle.get() != desc {
                    handle.set(desc)?;
                }
            }
        }
        Ok(())
    }

    fn clear(&mut self, h: &mut FeatureHandles<'_>) {
        self.desired = None;
        let Some(saved) = self.saved.take() else {
            return;
        };
        for (index, handle) in h.handles.auto_exposure.iter().enumerate() {
            let _ = handle.set(saved.get(index).copied().unwrap_or_default());
        }
    }
}
