//! m32 render graph, share 1/2 (S54): the pure JSON helpers of the render-graph API, the
//! owned-setter list, the api state `RenderGraphState`, and the wasm32 runtime that installs the
//! executor and drives its frame (`install`, `frame`, `with_handles`).
//!
//! The helpers are plain Rust, so `tests/render_graph.rs` includes this file and exercises them
//! natively: the issue, result and statistics JSON shapes and the owned-setter order. The wasm32
//! part builds the `HandleSet` from the FeatureMap and runs `GraphExecutor::frame`; the Engine
//! methods of card 'api m32 2/2' attach to the same file and reuse `with_handles` and the JSON
//! boundary helpers.

// Some items are reached only by the host tests or only by a later card's Engine methods, so they
// are dead in one of the two builds.
#![allow(dead_code)]

use raa_graph::catalog::Catalog;
use raa_graph::error::Issue;
use raa_graph::exec::{GraphExecutor, NodeStats, ResourceStat, SetResult};
use raa_loader::json::JsonValue;

/// The render graph state installed into the FeatureMap: the executor that owns the active graph
/// and the catalog its plan was validated against.
pub(crate) struct RenderGraphState {
    pub(crate) executor: GraphExecutor,
    pub(crate) catalog: Catalog,
}

/// The S52 issues as a JSON array of `Issue::to_json` values.
pub fn json_to_issue_entries(issues: &[Issue]) -> JsonValue {
    JsonValue::Array(issues.iter().map(Issue::to_json).collect())
}

/// A `SetResult` as `{ ok, issues, rebuilt }`.
pub fn result_to_json(r: &SetResult) -> JsonValue {
    JsonValue::Object(vec![
        ("ok".to_string(), JsonValue::Bool(r.ok)),
        ("issues".to_string(), json_to_issue_entries(&r.issues)),
        (
            "rebuilt".to_string(),
            JsonValue::Number(f64::from(r.rebuilt)),
        ),
    ])
}

/// The latest frame's node statistics as
/// `{ frameIndex: number|null, nodes: [{ id, type, ran, resources: [...] }] }`.
pub fn stats_to_json(frame_index: Option<u64>, nodes: &[NodeStats]) -> JsonValue {
    let frame_index = frame_index.map_or(JsonValue::Null, |i| JsonValue::Number(i as f64));
    let nodes = nodes.iter().map(node_stats_json).collect();
    JsonValue::Object(vec![
        ("frameIndex".to_string(), frame_index),
        ("nodes".to_string(), JsonValue::Array(nodes)),
    ])
}

/// One `NodeStats` as `{ id, type, ran, resources }`.
fn node_stats_json(node: &NodeStats) -> JsonValue {
    let resources = node.resources.iter().map(resource_stats_json).collect();
    JsonValue::Object(vec![
        ("id".to_string(), JsonValue::String(node.node.clone())),
        ("type".to_string(), JsonValue::String(node.type_id.clone())),
        ("ran".to_string(), JsonValue::Bool(node.ran)),
        ("resources".to_string(), JsonValue::Array(resources)),
    ])
}

/// One `ResourceStat` as `{ port, width, height, layers, mips, format, bytes }`.
fn resource_stats_json(r: &ResourceStat) -> JsonValue {
    JsonValue::Object(vec![
        ("port".to_string(), JsonValue::String(r.port.clone())),
        ("width".to_string(), JsonValue::Number(f64::from(r.width))),
        ("height".to_string(), JsonValue::Number(f64::from(r.height))),
        ("layers".to_string(), JsonValue::Number(f64::from(r.layers))),
        ("mips".to_string(), JsonValue::Number(f64::from(r.mips))),
        ("format".to_string(), JsonValue::String(r.format.clone())),
        ("bytes".to_string(), JsonValue::Number(r.bytes as f64)),
    ])
}

/// The S54 owned setter names, in the executor's order.
pub fn owned_setters() -> Vec<&'static str> {
    GraphExecutor::owned_setters().to_vec()
}

#[cfg(target_arch = "wasm32")]
pub(crate) use web::{frame, install};

#[cfg(target_arch = "wasm32")]
pub(crate) mod web {
    use super::*;
    use crate::api::ViewsState;
    use crate::convert::{INVALID_ARGUMENT, api_error};
    use crate::engine::InstallCtx;
    use crate::state::EngineState;
    use raa_bloom::BloomHandle;
    use raa_cel::CelHandle;
    use raa_contact::ContactHandle;
    use raa_core::{GpuContext, RaaError};
    use raa_dof::DofHandle;
    use raa_fog::FogHandle;
    use raa_gi::GiHandle;
    use raa_grading::GradingHandle;
    use raa_graph::exec::{FeatureHandles, HandleSet};
    use raa_gtao::GtaoHandle;
    use raa_oit::OitHandle;
    use raa_shadow::ShadowHandle;
    use raa_shafts::ShaftsHandle;
    use raa_ssr::SsrHandle;
    use raa_taa::TaaHandle;
    use raa_vg::VgHandle;
    use raa_web::FrameInput;
    use wasm_bindgen::JsValue;

    /// Installs the render graph state. It runs last in `install_all`, so every feature handle
    /// the executor wants already exists.
    pub(crate) fn install(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
        ctx.features.insert(RenderGraphState {
            executor: GraphExecutor::new(),
            catalog: Catalog::build(),
        });
        Ok(())
    }

    /// Runs after every other api file's frame hook and before `lit.prepare`: the executor
    /// restores the graph's values after external setters and advances its frame.
    pub(crate) fn frame(
        st: &mut EngineState,
        gpu: &GpuContext,
        _input: &FrameInput,
    ) -> Result<(), RaaError> {
        with_handles(st, gpu, |executor, _catalog, handles, _viewports| {
            executor.frame(handles)
        })?
    }

    /// Handoff: `with_handles`, `json_to_issue_entries`, `result_to_json`, `stats_to_json`,
    /// `owned_setters`, `json_to_js` and `js_to_json` in crates/raa-js/src/api/m32_render_graph.rs
    /// are called by card 'api m32 2/2: render graph Engine methods' from its `impl Engine` block
    /// in this file. `with_handles` takes `RenderGraphState` out of the FeatureMap through
    /// `FeatureMap::require_mut<RenderGraphState>`, so a missing `install` refuses with
    /// `api.feature_missing`, and otherwise passes the state's `executor` and `catalog`, the
    /// `FeatureHandles` built from the FeatureMap today and the viewport count as the last
    /// argument. State/keys: `RenderGraphState.executor`, `RenderGraphState.catalog`,
    /// `EngineState.tonemap`, `EngineState.pipeline`, `EngineState.lit`, the cloned feature
    /// handles and `ViewsState`. The msaa is the lit renderer's.
    pub(crate) fn with_handles<R>(
        st: &mut EngineState,
        gpu: &GpuContext,
        f: impl FnOnce(&mut GraphExecutor, &Catalog, &mut FeatureHandles<'_>, u32) -> R,
    ) -> Result<R, RaaError> {
        let viewports = viewport_count(st);
        let msaa = st.lit.handle().msaa();
        let mut set = HandleSet {
            tonemap: Some(st.tonemap.clone()),
            shadow: st.features.get::<ShadowHandle>().cloned(),
            gtao: st.features.get::<GtaoHandle>().cloned(),
            contact: st.features.get::<ContactHandle>().cloned(),
            bloom: st.features.get::<BloomHandle>().cloned(),
            oit: st.features.get::<OitHandle>().cloned(),
            ssr: st.features.get::<SsrHandle>().cloned(),
            fog: st.features.get::<FogHandle>().cloned(),
            shafts: st.features.get::<ShaftsHandle>().cloned(),
            dof: st.features.get::<DofHandle>().cloned(),
            grading: st.features.get::<GradingHandle>().cloned(),
            cel: st.features.get::<CelHandle>().cloned(),
            vg: st.features.get::<VgHandle>().cloned(),
            gi: st.features.get::<GiHandle>().cloned(),
            taa: st.features.get::<TaaHandle>().cloned(),
        };
        let pipeline = st.pipeline.clone();
        let lit = &mut st.lit;
        let state = st
            .features
            .require_mut::<RenderGraphState>("render graph")?;
        let RenderGraphState { executor, catalog } = state;
        let mut handles = FeatureHandles {
            gpu,
            pipeline: &pipeline,
            handles: &mut set,
            viewports,
            msaa,
            lit: Some(lit),
        };
        Ok(f(executor, catalog, &mut handles, viewports))
    }

    /// The viewport count of `ViewsState`, or 1 without it.
    fn viewport_count(st: &EngineState) -> u32 {
        st.features.get::<ViewsState>().map_or(1, |views| {
            u32::try_from(views.viewports().len()).unwrap_or(u32::MAX)
        })
    }

    /// The JSON value as a JS value: the canonical writer's text parsed by the JS engine.
    pub(crate) fn json_to_js(v: &JsonValue) -> JsValue {
        let mut text = String::new();
        raa_loader::json::write_pretty(v, &mut text);
        js_sys::JSON::parse(&text).unwrap_or(JsValue::NULL)
    }

    /// A JS value as a JSON value through `JSON.stringify` and the one JSON parser; a value
    /// the stringifier refuses is `api.invalid_argument`.
    pub(crate) fn js_to_json(v: &JsValue) -> Result<JsonValue, RaaError> {
        let text = js_sys::JSON::stringify(v)
            .ok()
            .and_then(|s| s.as_string())
            .ok_or_else(|| api_error(INVALID_ARGUMENT, "value is not JSON-serialisable"))?;
        raa_loader::json::parse(&text)
            .map_err(|e| api_error(INVALID_ARGUMENT, format!("invalid JSON value: {e}")))
    }
}
