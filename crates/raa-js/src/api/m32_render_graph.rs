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
use raa_graph::error::{Issue, Severity, Stage};
use raa_graph::exec::{GraphExecutor, NodeStats, ResourceStat, SetResult};
use raa_loader::json::JsonValue;

/// The render graph state installed into the FeatureMap: the executor that owns the active graph
/// and the catalog its plan was validated against.
pub(crate) struct RenderGraphState {
    pub(crate) executor: GraphExecutor,
    pub(crate) catalog: Catalog,
    /// The issues of the latest `setRenderGraph` that did not throw: a parse refusal, the executor's
    /// `SetResult.issues`, or empty after `null`.
    pub(crate) last_set_issues: Vec<Issue>,
}

/// The `graph.multi_view` warning `renderGraphIssues` appends while a graph is active and more than
/// one viewport is on: the graph is single view, so the extra views keep the engine's own post passes.
pub fn multi_view_warning(viewports: u32) -> Issue {
    Issue {
        code: "graph.multi_view",
        severity: Severity::Warning,
        stage: Stage::Structure,
        node: None,
        port: None,
        param: None,
        link: None,
        message: format!(
            "{viewports} viewports are active while a render graph is set: the graph is single view, so each view keeps the engine's own post passes"
        ),
        fix: "turn the extra viewports off with setViewports, or clear the graph with setRenderGraph(null)"
            .to_string(),
    }
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
    use crate::engine::{Engine, InstallCtx};
    use crate::feature_map::FEATURE_MISSING;
    use crate::state::EngineState;
    use raa_bloom::BloomHandle;
    use raa_cel::CelHandle;
    use raa_contact::ContactHandle;
    use raa_core::{GpuContext, RaaError};
    use raa_dof::DofHandle;
    use raa_fog::FogHandle;
    use raa_gi::GiHandle;
    use raa_grading::GradingHandle;
    use raa_graph::catalog::{NodeTypeSpec, ParamValue};
    use raa_graph::exec::{FeatureHandles, HandleSet};
    use raa_graph::format::{GraphDoc, graph_from_json_value, graph_to_json_value, parse_graph};
    use raa_graph::validate::{GraphLimits, ValidateCtx};
    use raa_gtao::GtaoHandle;
    use raa_oit::OitHandle;
    use raa_shadow::ShadowHandle;
    use raa_shafts::ShaftsHandle;
    use raa_ssr::SsrHandle;
    use raa_taa::TaaHandle;
    use raa_vg::VgHandle;
    use raa_web::FrameInput;
    use wasm_bindgen::JsValue;
    use wasm_bindgen::prelude::wasm_bindgen;

    /// Installs the render graph state. It runs last in `install_all`, so every feature handle
    /// the executor wants already exists.
    pub(crate) fn install(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
        ctx.features.insert(RenderGraphState {
            executor: GraphExecutor::new(),
            catalog: Catalog::build(),
            last_set_issues: Vec::new(),
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
        let RenderGraphState {
            executor, catalog, ..
        } = state;
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

    /// The render graph state of the FeatureMap, read-only; `api.feature_missing` when absent.
    fn graph_state(st: &EngineState) -> Result<&RenderGraphState, RaaError> {
        st.features.get::<RenderGraphState>().ok_or_else(|| {
            api_error(
                FEATURE_MISSING,
                "feature `render graph` is not installed in this engine",
            )
        })
    }

    /// The pipeline's entry names in execution order: the names a pass node must find installed.
    fn entry_names(st: &EngineState) -> Result<Vec<String>, RaaError> {
        let pipeline = st.pipeline.try_borrow().map_err(|_| {
            api_error(
                "pipeline.busy",
                "the pipeline is borrowed (called while a frame executes)",
            )
        })?;
        Ok(pipeline
            .passes()
            .into_iter()
            .map(|pass| pass.name)
            .collect())
    }

    /// The device limits that bound each stage's bindings of a graph (RG_LIMITS).
    fn graph_limits(gpu: &GpuContext) -> GraphLimits {
        let limits = gpu.device().limits();
        GraphLimits {
            sampled_textures: limits.max_sampled_textures_per_shader_stage,
            samplers: limits.max_samplers_per_shader_stage,
            storage_textures: limits.max_storage_textures_per_shader_stage,
            storage_buffers: limits.max_storage_buffers_per_shader_stage,
            uniform_buffers: limits.max_uniform_buffers_per_shader_stage,
        }
    }

    /// What a `setRenderGraph` argument holds once read.
    enum GraphArgument {
        /// Null or undefined: the graph is cleared.
        Clear,
        /// A document the parser accepted.
        Document(GraphDoc),
        /// Text or an object the parser refused, with its S52 issues.
        Refused(Vec<Issue>),
    }

    /// Reads a `setRenderGraph` argument: a string is JSON text, an object goes through `js_to_json`
    /// and `graph_from_json_value`; any other value throws `api.invalid_argument`.
    fn graph_argument(graph: &JsValue) -> Result<GraphArgument, RaaError> {
        if graph.is_null() || graph.is_undefined() {
            return Ok(GraphArgument::Clear);
        }
        let parsed = if let Some(text) = graph.as_string() {
            parse_graph(&text)
        } else if graph.is_object() {
            graph_from_json_value(&js_to_json(graph)?)
        } else {
            return Err(api_error(
                INVALID_ARGUMENT,
                "setRenderGraph takes a render graph object, its JSON text, or null",
            ));
        };
        Ok(match parsed {
            Ok(doc) => GraphArgument::Document(doc),
            Err(issues) => GraphArgument::Refused(issues),
        })
    }

    /// `setRenderGraph`: clears, refuses or applies the argument and returns its `RenderGraphResult`.
    /// The call replaces `last_set_issues` once it returns; a throw leaves it unchanged.
    fn apply_render_graph(
        gpu: &GpuContext,
        st: &mut EngineState,
        graph: &JsValue,
    ) -> Result<JsValue, RaaError> {
        let result = match graph_argument(graph)? {
            GraphArgument::Clear => {
                with_handles(st, gpu, |executor, _, h, _| executor.clear_graph(h))??;
                SetResult {
                    ok: true,
                    issues: Vec::new(),
                    rebuilt: 0,
                }
            }
            GraphArgument::Refused(issues) => SetResult {
                ok: false,
                issues,
                rebuilt: 0,
            },
            GraphArgument::Document(doc) => {
                let names = entry_names(st)?;
                let limits = graph_limits(gpu);
                with_handles(st, gpu, |executor, catalog, h, viewports| {
                    let installed = |name: &str| names.iter().any(|entry| entry == name);
                    let engine_options = [("msaa", ParamValue::Int(i64::from(h.msaa)))];
                    let ctx = ValidateCtx {
                        catalog,
                        limits,
                        installed: &installed,
                        viewports,
                        engine_options: &engine_options,
                    };
                    executor.set_graph(doc, &ctx, h)
                })?
            }
        };
        let json = result_to_json(&result);
        st.features
            .require_mut::<RenderGraphState>("render graph")?
            .last_set_issues = result.issues;
        Ok(json_to_js(&json))
    }

    #[wasm_bindgen(js_class = Engine)]
    impl Engine {
        /// The render graph catalog (S51) as a fresh object. A pass node whose pipeline entry is not
        /// installed in this engine reports `available: false`.
        #[wasm_bindgen(js_name = renderGraphCatalog)]
        pub fn render_graph_catalog(&self) -> Result<JsValue, JsValue> {
            self.with_state(|_, st| {
                let installed = entry_names(st)?;
                let state = graph_state(st)?;
                let available = |spec: &NodeTypeSpec| {
                    spec.pass
                        .is_none_or(|entry| installed.iter().any(|name| name == entry))
                };
                Ok(json_to_js(&state.catalog.to_json(&available)))
            })
        }

        /// Validates and applies a render graph (a document object, its JSON text, or null for the
        /// no-graph configuration). An invalid graph returns `{ ok: false, issues }` and changes
        /// nothing; any other value throws `api.invalid_argument`.
        #[wasm_bindgen(js_name = setRenderGraph)]
        pub fn set_render_graph(&mut self, graph: JsValue) -> Result<JsValue, JsValue> {
            self.with_state(|gpu, st| apply_render_graph(gpu, st, &graph))
        }

        /// The active graph's canonical document, or with no graph the export of today's
        /// configuration.
        #[wasm_bindgen(js_name = renderGraph)]
        pub fn render_graph(&self) -> Result<JsValue, JsValue> {
            self.with_state(|gpu, st| {
                with_handles(st, gpu, |executor, catalog, h, _| {
                    let json = match executor.active_doc() {
                        Some(doc) => graph_to_json_value(doc),
                        None => graph_to_json_value(&GraphExecutor::export_current(catalog, h)),
                    };
                    json_to_js(&json)
                })
            })
        }

        /// Whether a graph is active.
        #[wasm_bindgen(js_name = renderGraphActive)]
        pub fn render_graph_active(&self) -> Result<bool, JsValue> {
            self.with_state(|_, st| Ok(graph_state(st)?.executor.active_doc().is_some()))
        }

        /// The issues of the latest `setRenderGraph`, plus `graph.multi_view` while a graph is active
        /// and more than one viewport is on.
        #[wasm_bindgen(js_name = renderGraphIssues)]
        pub fn render_graph_issues(&self) -> Result<JsValue, JsValue> {
            self.with_state(|_, st| {
                let viewports = viewport_count(st);
                let state = graph_state(st)?;
                let mut issues = state.last_set_issues.clone();
                if state.executor.active_doc().is_some() && viewports > 1 {
                    issues.push(multi_view_warning(viewports));
                }
                Ok(json_to_js(&json_to_issue_entries(&issues)))
            })
        }

        /// The statistics of the latest frame of the active graph (`frameIndex` null before one).
        #[wasm_bindgen(js_name = renderGraphStats)]
        pub fn render_graph_stats(&self) -> Result<JsValue, JsValue> {
            self.with_state(|_, st| {
                let state = graph_state(st)?;
                let stats = stats_to_json(state.executor.frame_index(), &state.executor.stats());
                Ok(json_to_js(&stats))
            })
        }

        /// The setters the graph owns while it is active, in S54 order, as a string array.
        #[wasm_bindgen(js_name = renderGraphOwnedSetters)]
        pub fn render_graph_owned_setters(&self) -> Result<JsValue, JsValue> {
            self.with_state(|_, _| {
                let names = owned_setters()
                    .into_iter()
                    .map(|name| JsonValue::String(name.to_string()))
                    .collect();
                Ok(json_to_js(&JsonValue::Array(names)))
            })
        }
    }
}
