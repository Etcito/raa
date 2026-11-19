//! m33 frozen render-graph API (S60): `setFrozenRenderGraph` applies one of the committed frozen
//! pipelines through the m32 executor state. The frozen plans and their saved source JSON come
//! from the `raa-graph` registry, so the chosen plan is applied as a literal: it is never replanned
//! and a stale literal is refused instead of being upgraded.
//!
//! The selection (`FrozenSelection`, `select_frozen`) is plain Rust: `crates/raa-graph/tests/frozen.rs`
//! includes this file natively and exercises the unknown and stale refusals without a device. The
//! wasm32 part sits in `web` and reuses the m32 `RenderGraphState`, `with_handles`, `result_to_json`
//! and `json_to_js`.

use raa_graph::catalog::Catalog;
use raa_graph::error::{Issue, Severity, Stage};
use raa_graph::format::{GraphDoc, parse_graph};
use raa_graph::plan::GraphPlan;
use raa_graph::registry;
use raa_graph::validate::canonical_doc;

/// The frozen registry entry chosen for a name: its literal plan and the parsed, canonical saved
/// source document that becomes the active document (`m32 renderGraphActive` tests `active_doc`).
#[derive(Clone, Debug, PartialEq)]
pub struct FrozenSelection {
    pub plan: GraphPlan,
    pub doc: GraphDoc,
}

/// Selects the frozen pipeline `name` against `catalog`. An absent registry name is
/// `graph.frozen_unknown`; a plan node whose type is missing from the catalog or was generated for
/// another version is `graph.frozen_stale`. Nothing is selected (or applied) in either case.
pub fn select_frozen(name: &str, catalog: &Catalog) -> Result<FrozenSelection, Vec<Issue>> {
    let frozen = registry::frozen_all();
    let Some((_, plan_fn, source_json)) =
        frozen.iter().find(|(candidate, _, _)| *candidate == name)
    else {
        let names: Vec<&str> = frozen.iter().map(|(candidate, _, _)| *candidate).collect();
        let fix = if names.is_empty() {
            "no frozen pipeline is registered in this build".to_string()
        } else {
            format!("use one of the frozen pipelines: {}", names.join(", "))
        };
        return Err(vec![frozen_issue(
            "graph.frozen_unknown",
            None,
            format!("unknown frozen pipeline `{name}`"),
            fix,
        )]);
    };

    let plan = (*plan_fn)();
    let stale: Vec<Issue> = plan
        .nodes
        .iter()
        .filter_map(|node| match catalog.node(&node.type_id) {
            None => Some(frozen_issue(
                "graph.frozen_stale",
                Some(node.id.clone()),
                format!(
                    "frozen node `{}` uses type `{}`, which the catalog does not have",
                    node.id, node.type_id
                ),
                "regenerate the frozen pipeline with raa-graph-freeze for this catalog".to_string(),
            )),
            Some(spec) if node.type_version != spec.version => Some(frozen_issue(
                "graph.frozen_stale",
                Some(node.id.clone()),
                format!(
                    "frozen node `{}` was generated for type `{}` version {} but the catalog has version {}",
                    node.id, node.type_id, node.type_version, spec.version
                ),
                "regenerate the frozen pipeline with raa-graph-freeze instead of upgrading it"
                    .to_string(),
            )),
            Some(_) => None,
        })
        .collect();
    if !stale.is_empty() {
        return Err(stale);
    }

    let doc = parse_graph(source_json)?;
    Ok(FrozenSelection {
        plan,
        doc: canonical_doc(&doc, catalog),
    })
}

/// A `graph.frozen_*` plan-stage issue, with `node` set only when it names a plan node.
fn frozen_issue(code: &'static str, node: Option<String>, message: String, fix: String) -> Issue {
    Issue {
        code,
        severity: Severity::Error,
        stage: Stage::Plan,
        node,
        port: None,
        param: None,
        link: None,
        message,
        fix,
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) use web::{frame, install};

#[cfg(target_arch = "wasm32")]
pub(crate) mod web {
    use super::*;
    use crate::api::m32_render_graph::RenderGraphState;
    use crate::api::m32_render_graph::result_to_json;
    use crate::api::m32_render_graph::web::{json_to_js, with_handles};
    use crate::convert::{INVALID_ARGUMENT, api_error};
    use crate::engine::{Engine, InstallCtx};
    use crate::state::EngineState;
    use raa_core::{GpuContext, RaaError};
    use raa_graph::exec::SetResult;
    use raa_web::FrameInput;
    use wasm_bindgen::JsValue;
    use wasm_bindgen::prelude::wasm_bindgen;

    /// The m32 install already put `RenderGraphState` in the FeatureMap; this file only adds the
    /// Engine method that applies a frozen plan through it.
    pub(crate) fn install(_ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
        Ok(())
    }

    /// The executor's own `frame` hook drives an active frozen plan; this file adds nothing.
    pub(crate) fn frame(
        _st: &mut EngineState,
        _gpu: &GpuContext,
        _input: &FrameInput,
    ) -> Result<(), RaaError> {
        Ok(())
    }

    /// Selects and applies `name`; a refused selection leaves the previous graph and the executor's
    /// last issues unchanged until the returned `RenderGraphResult` replaces `last_set_issues`.
    fn apply_frozen(
        name: &str,
        gpu: &GpuContext,
        st: &mut EngineState,
    ) -> Result<JsValue, RaaError> {
        let result = with_handles(
            st,
            gpu,
            |executor, catalog, handles, _| match select_frozen(name, catalog) {
                Err(issues) => SetResult {
                    ok: false,
                    issues,
                    rebuilt: 0,
                },
                Ok(selection) => executor.set_plan(selection.plan, Some(selection.doc), handles),
            },
        )?;
        let json = result_to_json(&result);
        st.features
            .require_mut::<RenderGraphState>("render graph")?
            .last_set_issues = result.issues;
        Ok(json_to_js(&json))
    }

    #[wasm_bindgen(js_class = Engine)]
    impl Engine {
        /// Handoff: 'Add the Frozen pipeline selector' calls `set_frozen_render_graph`, which applies
        /// the named frozen plan through `GraphExecutor::set_plan` today and returns the S54
        /// `RenderGraphResult`. An absent name is `graph.frozen_unknown`; a stale literal is
        /// `graph.frozen_stale` and nothing is applied.
        #[wasm_bindgen(js_name = setFrozenRenderGraph)]
        pub fn set_frozen_render_graph(&mut self, name: JsValue) -> Result<JsValue, JsValue> {
            self.with_state(|gpu, st| {
                let name = name.as_string().ok_or_else(|| {
                    api_error(
                        INVALID_ARGUMENT,
                        "setFrozenRenderGraph takes a frozen pipeline name",
                    )
                })?;
                apply_frozen(&name, gpu, st)
            })
        }
    }
}
