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
    pub(crate) explicit_scene_size: Option<(u32, u32)>,
    pub(crate) frozen_name: Option<String>,
    pub(crate) area_model: Option<String>,
    models: std::collections::BTreeMap<String, raa_graph::learned::AreaPilotPackage>,
    real_models: std::collections::BTreeMap<String, raa_graph::exec::RealModelPackage>,
    selected_real_model: Option<(String, raa_graph::exec::RealModelPackage)>,
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
    use crate::api::m21_views::ViewRect;
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
    use raa_graph::exec::{AreaGraphModel, FeatureHandles, HandleSet};
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
        crate::mesh_trace::install_for_graph(ctx);
        crate::spherical_gi::install_for_graph(ctx)?;
        ctx.features.insert(RenderGraphState {
            executor: GraphExecutor::new(),
            catalog: Catalog::build(),
            explicit_scene_size: None,
            frozen_name: None,
            area_model: None,
            models: Default::default(),
            real_models: Default::default(),
            selected_real_model: None,
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
    /// handles (among them `raa_sky::SkyHandle` as `HandleSet.sky`) and `ViewsState`. The msaa is
    /// the lit renderer's.
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
            mesh_trace: st.features.get::<raa_sdf_trace::mesh::MeshTraceHandle>().cloned(),
            fog: st.features.get::<FogHandle>().cloned(),
            shafts: st.features.get::<ShaftsHandle>().cloned(),
            dof: st.features.get::<DofHandle>().cloned(),
            grading: st.features.get::<GradingHandle>().cloned(),
            cel: st.features.get::<CelHandle>().cloned(),
            vg: st.features.get::<VgHandle>().cloned(),
            gi: st.features.get::<GiHandle>().cloned(),
            spherical: st.features.get::<raa_sdf_trace::mesh::SphericalGiHandle>().cloned(),
            taa: st.features.get::<TaaHandle>().cloned(),
            sky: st.features.get::<raa_sky::SkyHandle>().cloned(),
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

    #[wasm_bindgen]
    impl Engine {
        /// Validate an explicit, versioned experimental model without changing the live graph.
        #[wasm_bindgen(js_name = registerNeuralModel)]
        pub fn register_neural_model(&self, id: String, manifest: String, weights: Vec<u8>) -> Result<(), JsValue> {
            self.with_state(|_, st| {
                if id.is_empty() || id.len() > 64 {
                    return Err(api_error(INVALID_ARGUMENT, "expected a model ID of 1 to 64 bytes"));
                }
                let model = raa_graph::exec::RealModelPackage::validate(&manifest, weights)?;
                let state = st.features.require_mut::<RenderGraphState>("render graph")?;
                if state.real_models.contains_key(&id) {
                    return Err(api_error(INVALID_ARGUMENT, "neural model ID is already registered"));
                }
                state.real_models.insert(id, model);
                Ok(())
            })
        }

        /// Switch independent real-frame heads atomically; null restores the graph's defaults.
        #[wasm_bindgen(js_name = setNeuralReconstruction)]
        pub fn set_neural_reconstruction(&self, options: JsValue) -> Result<(), JsValue> {
            use raa_graph::exec::{RealModelHeads, RealModelSelection};
            self.with_state(|gpu, st| {
                check_pipeline_available(st)?;
                let selection = if options.is_null() {
                    None
                } else {
                    if !options.is_object() || js_sys::Array::is_array(&options) {
                        return Err(api_error(INVALID_ARGUMENT, "expected neural reconstruction options or null"));
                    }
                    let field = |key: &str| js_sys::Reflect::get(&options, &JsValue::from_str(key))
                        .map_err(|_| api_error(INVALID_ARGUMENT, format!("cannot read neural option {key}")));
                    let boolean = |key: &str| {
                        let value = field(key)?;
                        if value.is_undefined() { return Ok(false); }
                        value.as_bool().ok_or_else(|| api_error(INVALID_ARGUMENT, format!("{key} must be boolean")))
                    };
                    let heads = RealModelHeads {
                        area: boolean("area")?, reflection: boolean("reflection")?, sr: boolean("sr")?,
                    };
                    if !heads.any() {
                        return Err(api_error(INVALID_ARGUMENT, "select one or more neural heads, or use null"));
                    }
                    let id = field("model")?.as_string().ok_or_else(|| api_error(INVALID_ARGUMENT, "expected a registered model ID"))?;
                    if st.features.get::<ViewsState>().is_some_and(|v| v.viewports() != [ViewRect::FULL]) {
                        return Err(api_error("graph.multi_view", "real-frame reconstruction requires a full-canvas view"));
                    }
                    let graph = st.features.require_mut::<RenderGraphState>("render graph")?;
                    let model = graph.real_models.get(&id).cloned()
                        .ok_or_else(|| api_error("graph.neural_package", "unknown real-frame model"))?;
                    let size = graph.explicit_scene_size.unwrap_or(self.viewport);
                    Some((id, model, heads, size))
                };
                let candidate = selection.as_ref().map(|(_, package, heads, size)| RealModelSelection {
                    package: package.clone(), heads: *heads, scene_size: *size,
                });
                with_handles(st, gpu, |executor, _, h, _| executor.select_real_model(h, candidate))??;
                st.features.require_mut::<RenderGraphState>("render graph")?.selected_real_model =
                    selection.map(|(id, model, _, _)| (id, model));
                Ok(())
            })
        }

        /// Diagnostics for the last submitted real frame, with explicit selected and executed heads.
        #[wasm_bindgen(js_name = neuralReconstructionStats)]
        pub fn neural_reconstruction_stats(&self) -> Result<JsValue, JsValue> {
            self.with_state(|_, st| {
                let graph = st.features.require_mut::<RenderGraphState>("render graph")?;
                let Some((selected, stats)) = graph.executor.real_model_stats() else { return Ok(JsValue::NULL); };
                let (id, package) = graph.selected_real_model.as_ref().expect("active model registration");
                let heads = |h: raa_graph::exec::RealModelHeads| JsonValue::Object(vec![
                    ("area".into(), JsonValue::Bool(h.area)),
                    ("reflection".into(), JsonValue::Bool(h.reflection)),
                    ("sr".into(), JsonValue::Bool(h.sr)),
                ]);
                let size = |(x, y)| JsonValue::Array(vec![JsonValue::Number(f64::from(x)), JsonValue::Number(f64::from(y))]);
                Ok(json_to_js(&JsonValue::Object(vec![
                    ("model".into(), JsonValue::String(id.clone())),
                    ("weightsSha256".into(), JsonValue::String(package.digest().to_owned())),
                    ("kind".into(), JsonValue::String(package.kind().to_owned())),
                    ("trainedHeads".into(), JsonValue::Array(package.trained_heads().iter().cloned().map(JsonValue::String).collect())),
                    ("selected".into(), heads(selected)),
                    ("ran".into(), heads(stats.ran)),
                    ("realFrame".into(), stats.submitted_frame.map_or(JsonValue::Null, |i| JsonValue::Number(i as f64))),
                    ("sceneSize".into(), size(stats.scene_size)),
                    ("displaySize".into(), size(stats.display_size)),
                    ("gpuBytes".into(), JsonValue::Number(stats.bytes as f64)),
                    ("dispatches".into(), JsonValue::Number(f64::from(stats.dispatches))),
                    ("historyFrames".into(), JsonValue::Number(f64::from(stats.history_frames))),
                ])))
            })
        }
    }

    impl Engine {
        pub(crate) fn camera_graph_selected(st: &EngineState) -> bool {
            graph_state(st).ok().is_some_and(|graph| {
                graph.explicit_scene_size.is_none() && graph.executor.active_plan().is_some_and(|plan| {
                    plan.nodes.iter().any(|node| node.type_id == "neural.frame-generation")
                })
            })
        }
    }

    /// The viewport count of `ViewsState`, or 1 without it.
    pub(crate) fn viewport_count(st: &EngineState) -> u32 {
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
    pub(crate) fn graph_state(st: &EngineState) -> Result<&RenderGraphState, RaaError> {
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

    pub(crate) type SelectedRealModel = (String, raa_graph::exec::RealModelPackage);

    pub(crate) fn graph_model(
        st: &EngineState, doc: &GraphDoc, size: (u32, u32),
    ) -> Result<Option<(SelectedRealModel, raa_graph::exec::RealModelSelection)>, SetResult> {
        use raa_graph::exec::{RealModelHeads, RealModelSelection};
        let mut name: Option<&str> = None;
        let mut heads = RealModelHeads::default();
        let issue = |message| refusal("graph.neural_package", message, "Register one compatible real-frame package for the selected heads, or clear the model fields for native processing.");
        for node in &doc.nodes {
            let head = match node.type_id.as_str() {
                "neural.area-reconstruction" => &mut heads.area,
                "neural.reflection-reconstruction" => &mut heads.reflection,
                "neural.super-resolution" => &mut heads.sr,
                _ => continue,
            };
            let Some((_, JsonValue::String(id))) = node.params.iter().find(|param| param.0 == "model") else { continue; };
            if id.is_empty() { continue; }
            if name.is_some_and(|old| old != id) {
                return Err(issue("the joint real-frame nodes require the same registered model ID"));
            }
            name = Some(id);
            *head = true;
        }
        let Some(name) = name else { return Ok(None); };
        let graph = graph_state(st).map_err(|_| issue("render graph state is missing"))?;
        let package = graph.real_models.get(name).cloned().ok_or_else(|| issue("a named neural package is not registered"))?;
        if st.features.get::<ViewsState>().is_some_and(|v| v.viewports() != [ViewRect::FULL]) {
            return Err(issue("real-frame model selection requires a full-canvas viewport"));
        }
        Ok(Some(((name.to_owned(), package.clone()), RealModelSelection { package, heads, scene_size: size })))
    }

    pub(crate) fn remember_graph_model(st: &mut EngineState, selected: Option<SelectedRealModel>) -> Result<(), RaaError> {
        st.features.require_mut::<RenderGraphState>("render graph")?.selected_real_model = selected;
        Ok(())
    }

    /// `setRenderGraph`: clears, refuses or applies the argument and returns its `RenderGraphResult`.
    /// The call replaces `last_set_issues` once it returns; a throw leaves it unchanged.
    fn apply_render_graph(
        gpu: &GpuContext,
        st: &mut EngineState,
        graph: &JsValue,
        area_model: Option<AreaGraphModel>,
        scene_size: (u32, u32),
    ) -> Result<SetResult, RaaError> {
        Ok(match graph_argument(graph)? {
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
                let (selected, real_model) = match graph_model(st, &doc, scene_size) {
                    Ok(Some((selected, model))) => (Some(selected), Some(model)),
                    Ok(None) => (None, None),
                    Err(result) => return Ok(result),
                };
                let names = entry_names(st)?;
                let limits = graph_limits(gpu);
                let result = with_handles(st, gpu, |executor, catalog, h, viewports| {
                    let installed = |name: &str| names.iter().any(|entry| entry == name);
                    let engine_options = [("msaa", ParamValue::Int(i64::from(h.msaa)))];
                    let ctx = ValidateCtx {
                        catalog,
                        limits,
                        installed: &installed,
                        viewports,
                        engine_options: &engine_options,
                    };
                    executor.set_graph_with_models(doc, &ctx, h, area_model, real_model)
                })?;
                if result.ok { remember_graph_model(st, selected)?; }
                result
            }
        })
    }

    pub(crate) fn finish_selection(
        st: &mut EngineState,
        result: SetResult,
        scene_size: Option<(u32, u32)>,
        frozen_name: Option<String>,
        area_model: Option<String>,
    ) -> Result<JsValue, RaaError> {
        let json = result_to_json(&result);
        let state = st
            .features
            .require_mut::<RenderGraphState>("render graph")?;
        if result.ok {
            if state.explicit_scene_size != scene_size {
                st.pipeline.borrow_mut().set_scene_size(scene_size);
                st.scene_invocation.discontinuity_all();
                state.explicit_scene_size = scene_size;
            }
            state.frozen_name = frozen_name;
            state.area_model = area_model;
            if state.executor.real_model_stats().is_none() { state.selected_real_model = None; }
        }
        state.last_set_issues = result.issues;
        Ok(json_to_js(&json))
    }

    pub(crate) fn check_pipeline_available(st: &EngineState) -> Result<(), RaaError> {
        st.pipeline.try_borrow_mut().map(|_| ()).map_err(|_| {
            api_error(
                "pipeline.busy",
                "the pipeline is borrowed while a frame executes",
            )
        })
    }

    fn refusal(code: &'static str, message: &str, fix: &str) -> SetResult {
        SetResult {
            ok: false,
            rebuilt: 0,
            issues: vec![Issue {
                code,
                severity: Severity::Error,
                stage: Stage::Structure,
                node: None,
                port: None,
                param: None,
                link: None,
                message: message.into(),
                fix: fix.into(),
            }],
        }
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
            let viewport = self.viewport;
            self.with_state(|gpu, st| {
                check_pipeline_available(st)?;
                let result = apply_render_graph(gpu, st, &graph, None, viewport)?;
                finish_selection(st, result, None, None, None)
            })
        }

        /// Register immutable FP32 area-pilot bytes after validating the package ABI and hashes.
        #[wasm_bindgen(js_name = registerAreaRenderModel)]
        pub fn register_area_render_model(
            &mut self,
            name: String,
            manifest: String,
            weights: Vec<u8>,
        ) -> Result<(), JsValue> {
            self.with_state(|_, st| {
                let state = st
                    .features
                    .require_mut::<RenderGraphState>("render graph")?;
                if name.trim().is_empty() || state.models.contains_key(&name) {
                    return Err(api_error(
                        INVALID_ARGUMENT,
                        "render model name is empty or already registered",
                    ));
                }
                let package = raa_graph::learned::AreaPilotPackage::validate(&manifest, weights)?;
                state.models.insert(name, package);
                Ok(())
            })
        }

        /// Select an editable or frozen graph and its optional fixed scene extent in one call.
        /// The canvas and legacy resize continue to determine the display extent.
        #[wasm_bindgen(js_name = setRenderGraphSelection)]
        pub fn set_render_graph_selection(
            &mut self,
            graph: JsValue,
            frozen: Option<String>,
            scene_width: Option<f64>,
            scene_height: Option<f64>,
            area_model: Option<String>,
        ) -> Result<JsValue, JsValue> {
            let viewport = self.viewport;
            self.with_state(|gpu, st| {
                let limit = gpu.device().limits().max_texture_dimension_2d;
                let dimension = |n: f64| n.fract() == 0.0 && n >= 1.0 && n <= f64::from(limit);
                let size = match (scene_width, scene_height) {
                    (None, None) => None,
                    (Some(w), Some(h)) if dimension(w) && dimension(h) => Some((w as u32, h as u32)),
                    _ => return Err(api_error(INVALID_ARGUMENT, "scene extent needs two positive integer dimensions within device limits")),
                };
                let clear = graph.is_null() || graph.is_undefined();
                if frozen.is_some() && !clear || frozen.is_none() && clear && (size.is_some() || area_model.is_some()) {
                    return Err(api_error(INVALID_ARGUMENT, "specify one graph when selecting a frozen graph, a model, or an explicit scene extent"));
                }
                let package = match &area_model {
                    Some(name) => match graph_state(st)?.models.get(name) {
                        Some(package) => Some(package.clone()),
                        None => return finish_selection(st,
                            refusal("graph.model_unknown", "the selected area model is not registered",
                                "Register the model package on this engine before selecting it."),
                            size, frozen, area_model),
                    },
                    None => None,
                };
                let single_full_view = size.is_some() || package.is_some();
                let result = if (frozen.is_some() || !clear) && viewport_count(st) > 1 {
                    refusal("graph.multi_view", "render graph selection drives one view",
                        "Turn off the extra viewports before selecting a graph.")
                } else if single_full_view && st.features.get::<ViewsState>().is_some_and(|v| v.viewports() != [ViewRect::FULL]) {
                    refusal("graph.multi_view", "scene extent and area model selection require a full-canvas viewport",
                        "Restore the full-canvas viewport before selecting this graph.")
                } else if size.is_some() && st.features.get::<ViewsState>().is_some_and(|v| v.render_scale() != 1.0) {
                    refusal("graph.scale_conflict", "explicit scene extent requires legacy render scale 1",
                        "Set legacy render scale to 1 before selecting an explicit scene extent.")
                } else {
                    check_pipeline_available(st)?;
                    let model = package.map(|package| AreaGraphModel {
                        package, scene_size: size.unwrap_or(viewport),
                    });
                    match &frozen {
                        Some(name) => super::super::m33_frozen_render_graph::web::apply_frozen(name, gpu, st, model, size.unwrap_or(viewport))?,
                        None => apply_render_graph(gpu, st, &graph, model, size.unwrap_or(viewport))?,
                    }
                };
                finish_selection(st, result, size, frozen, area_model)
            })
        }

        /// Metadata for the accepted selection. Refused requests leave it unchanged.
        #[wasm_bindgen(js_name = renderGraphSelection)]
        pub fn render_graph_selection(&self) -> Result<JsValue, JsValue> {
            self.with_state(|_, st| {
                let state = graph_state(st)?;
                let frozen = state
                    .frozen_name
                    .clone()
                    .map_or(JsonValue::Null, JsonValue::String);
                let size = state.explicit_scene_size.map_or(JsonValue::Null, |(w, h)| {
                    JsonValue::Array(vec![
                        JsonValue::Number(f64::from(w)),
                        JsonValue::Number(f64::from(h)),
                    ])
                });
                Ok(json_to_js(&JsonValue::Object(vec![
                    ("frozen".into(), frozen),
                    ("sceneSize".into(), size),
                    (
                        "areaModel".into(),
                        state
                            .area_model
                            .clone()
                            .map_or(JsonValue::Null, JsonValue::String),
                    ),
                ])))
            })
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
