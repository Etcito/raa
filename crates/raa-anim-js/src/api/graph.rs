//! S19 graph loading and state creation over the existing S16/S17 handles.
//! The graph borrows the rig and clips the caller already loaded; ticking,
//! snapshots and the graph-run selftest belong to their own cards.
use crate::clips::{AnimationClip, AnimationRig};
use raa_anim_graph::{GraphError, GraphRuntime, GraphState};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// A compiled S19 graph retaining the rig and clips it was loaded with.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct AnimationGraph {
    runtime: GraphRuntime,
}

/// One independent S19 execution state created from an [`AnimationGraph`].
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct AnimationGraphState {
    #[allow(dead_code)]
    state: GraphState,
}

/// Load S19 graph JSON against the rig handle and the supplied clip handles.
/// The clip handles are borrowed and stay usable after this call.
pub fn load_graph(
    json: &str,
    rig: &AnimationRig,
    clips: &[&AnimationClip],
) -> Result<AnimationGraph, GraphError> {
    let mut owned = Vec::with_capacity(clips.len());
    for clip in clips {
        let (_, clip) = clip.parts();
        owned.push(clip.clone());
    }
    GraphRuntime::load(json, rig.shared_rig(), &owned).map(|runtime| AnimationGraph { runtime })
}

/// The `graph-run` selftest case is contributed by its own card.
pub fn register(_cases: &mut Vec<crate::Case>) {}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl AnimationGraph {
    /// `GraphState::new` for this graph; every returned state is independent.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = newState))]
    pub fn new_state(&self) -> AnimationGraphState {
        AnimationGraphState {
            state: self.runtime.new_state(),
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use wasm_bindgen::__rt::{RcRef, WasmPtr};
    use wasm_bindgen::convert::RefFromWasmAbi;

    fn error(e: GraphError) -> JsValue {
        let value = js_sys::Error::new(&e.message);
        js_sys::Reflect::set(&value, &"code".into(), &e.code.into()).expect("fresh error");
        js_sys::Reflect::set(&value, &"path".into(), &e.path.into()).expect("fresh error");
        value.into()
    }

    fn invalid(message: &str) -> JsValue {
        let value = js_sys::TypeError::new(message);
        js_sys::Reflect::set(&value, &"code".into(), &"api.invalid_input".into())
            .expect("fresh error");
        js_sys::Reflect::set(&value, &"path".into(), &"".into()).expect("fresh error");
        value.into()
    }

    // wasm-bindgen arrays of exported types take ownership of their elements.
    // Borrow the class instead so the caller's clip handles stay usable.
    fn borrow_clip(value: &JsValue) -> Result<RcRef<AnimationClip>, JsValue> {
        let pointer = js_sys::Reflect::get(value, &JsValue::from_str("__wbg_ptr"))
            .ok()
            .and_then(|pointer| pointer.as_f64())
            .filter(|pointer| {
                pointer.is_finite() && *pointer > 0.0 && *pointer <= f64::from(u32::MAX)
            })
            .ok_or_else(|| invalid("clips must contain AnimationClip handles"))?;
        let abi = WasmPtr::from_usize(pointer as usize);
        Ok(unsafe { <AnimationClip as RefFromWasmAbi>::ref_from_abi(abi) })
    }

    #[wasm_bindgen(js_name = loadGraph)]
    pub fn graph(
        json: &str,
        rig: &AnimationRig,
        clips: &js_sys::Array,
    ) -> Result<AnimationGraph, JsValue> {
        if !js_sys::Array::is_array(clips) {
            return Err(invalid("clips must be an array of AnimationClip handles"));
        }
        let mut anchors = Vec::with_capacity(clips.length() as usize);
        for value in clips.iter() {
            anchors.push(borrow_clip(&value)?);
        }
        let clips: Vec<&AnimationClip> = anchors.iter().map(|anchor| &**anchor).collect();
        super::load_graph(json, rig, &clips).map_err(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clips::{load_clip, load_rig};
    use raa_anim_graph::scenario_data::{PENGUIN_CLIPS, PENGUIN_GRAPH, PENGUIN_RIG, clip_text};

    fn penguin() -> (AnimationRig, Vec<AnimationClip>) {
        let rig = load_rig(PENGUIN_RIG).expect("penguin rig");
        let clips = PENGUIN_CLIPS
            .iter()
            .map(|spec| load_clip(&rig, &clip_text(spec)).expect("penguin clip"))
            .collect();
        (rig, clips)
    }

    #[test]
    fn loads_graph_from_handles_and_creates_independent_states() {
        let (rig, clips) = penguin();
        let refs: Vec<&AnimationClip> = clips.iter().collect();
        let graph = load_graph(PENGUIN_GRAPH, &rig, &refs).expect("penguin graph");
        assert_eq!(graph.runtime.graph().name, "penguin");
        assert_eq!(graph.runtime.clips().len(), clips.len());
        let first = graph.new_state();
        let second = graph.new_state();
        assert_eq!(first.state, graph.runtime.new_state());
        assert_eq!(first.state, second.state);
        assert_eq!(first.state.tick, 0);
        assert_eq!(first.state.layers.len(), graph.runtime.graph().layers.len());
    }

    #[test]
    fn clip_handles_stay_usable_after_loading() {
        let (rig, clips) = penguin();
        let refs: Vec<&AnimationClip> = clips.iter().collect();
        for _ in 0..2 {
            load_graph(PENGUIN_GRAPH, &rig, &refs).expect("repeat load");
        }
        let (pose, _) = crate::clips::sample_clip(&clips[0], 0).expect("sample after load");
        assert!(!pose.skin.is_empty());
    }

    #[test]
    fn refuses_invalid_documents_without_creating_a_graph() {
        let (rig, _) = penguin();
        let error = load_graph("{}", &rig, &[])
            .err()
            .expect("empty graph refused");
        assert!(error.code.starts_with("graph."));
        let missing = load_graph(PENGUIN_GRAPH, &rig, &[])
            .err()
            .expect("missing clips refused");
        assert_eq!(missing.code, "graph.clip_unknown");
    }

    #[test]
    fn refuses_a_graph_for_another_rig() {
        let other = load_rig(include_str!(
            "../../../raa-anim/tests/fixtures/rig-basic.json"
        ))
        .expect("basic rig");
        let error = load_graph(PENGUIN_GRAPH, &other, &[])
            .err()
            .expect("rig mismatch refused");
        assert_eq!(error.code, "graph.schema");
    }
}
