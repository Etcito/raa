//! m37 asynchronous billboard baking (M6): the bake store behind `bakeBillboard` and
//! `removeBillboardBake`, the caller-owned proxy mesh of `createBillboardProxy` and the
//! install/frame hooks of the auto-discovered api module.
//!
//! A bake starts as a pending store entry. [`LitRenderer::bake_billboard`] renders the
//! eight directions; only a completed bake becomes ready, so a failed or removed bake can
//! never be selected by a billboard group. Removal detaches the dependent groups to LOD0
//! before the published atlas images are freed, and a completion that arrives after its
//! entry was removed hands its atlas ids back for disposal.
//!
//! The store and its conversions compile on the host: `tests/billboard_bake_api.rs`
//! includes this file with `#[path]` and drives them without a GPU.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use raa_core::RaaError;
use raa_lit::BillboardBakeResult;
use raa_math::Mat4;

/// Stable code of every bake-store refusal.
pub const BILLBOARD_INVALID: &str = "billboard.invalid";
/// Stable code of a bake that failed while rendering or publishing.
pub const BILLBOARD_BAKE_FAILED: &str = "billboard.bake_failed";
/// The four square tile resolutions the bake API accepts.
pub const BILLBOARD_RESOLUTIONS: [u32; 4] = [64, 128, 256, 512];

fn bake_error(code: &'static str, message: impl Into<String>) -> RaaError {
    RaaError::FrameGraph {
        code,
        message: message.into(),
    }
}

/// Refuses with [`BILLBOARD_INVALID`].
pub fn invalid(message: impl Into<String>) -> RaaError {
    bake_error(BILLBOARD_INVALID, message)
}

/// Refuses with [`BILLBOARD_BAKE_FAILED`].
#[allow(dead_code)]
pub fn bake_failed(message: impl Into<String>) -> RaaError {
    bake_error(BILLBOARD_BAKE_FAILED, message)
}

/// The store's copy of a completed bake: the public `BillboardBake` result, with the id
/// allocated by `bakeBillboard`. `tiles` and `extent` are texels, `center`/`size` metres.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BillboardBake {
    pub id: u32,
    pub diffuse: u32,
    pub normal: u32,
    pub opacity: u32,
    pub center: [f32; 3],
    pub size: [f32; 2],
    pub directions: u8,
    pub tiles: [[u32; 4]; 8],
    pub extent: [u32; 2],
}

impl BillboardBake {
    /// Converts the renderer's [`BillboardBakeResult`] into the caller-owned bake, copying
    /// the atlas handles, framing, tiles and extent unchanged.
    pub fn from_result(id: u32, result: BillboardBakeResult) -> Self {
        Self {
            id,
            diffuse: result.diffuse,
            normal: result.normal,
            opacity: result.opacity,
            center: result.center,
            size: result.size,
            directions: raa_lit::BILLBOARD_DIRECTIONS as u8,
            tiles: result.tiles,
            extent: result.extent,
        }
    }

    /// The three atlas image handles, in publication order.
    pub fn atlas(&self) -> [u32; 3] {
        [self.diffuse, self.normal, self.opacity]
    }
}

/// Validates the bake resolution: exactly 64, 128, 256 or 512.
pub fn validate_resolution(resolution: u32) -> Result<u32, RaaError> {
    if BILLBOARD_RESOLUTIONS.contains(&resolution) {
        Ok(resolution)
    } else {
        Err(invalid(format!(
            "resolution must be 64, 128, 256 or 512, got {resolution}"
        )))
    }
}

/// Expands the complete set of LOD0 instance handles to their submitted draw tags.
/// `tag_of` maps a live instance handle to its group tag. The list must be nonempty, hold
/// no duplicate handle and name no stale/unknown handle; tags are unique and keep input
/// order. Nothing is mutated.
pub fn expand_tags(
    instances: &[u32],
    mut tag_of: impl FnMut(u32) -> Option<u32>,
) -> Result<Vec<u32>, RaaError> {
    if instances.is_empty() {
        return Err(invalid("instances must hold at least one LOD0 draw"));
    }
    let mut handles = BTreeSet::new();
    let mut tags = Vec::with_capacity(instances.len());
    let mut seen_tags = BTreeSet::new();
    for &handle in instances {
        if !handles.insert(handle) {
            return Err(invalid(format!("instance {handle} is listed twice")));
        }
        let tag =
            tag_of(handle).ok_or_else(|| invalid(format!("instance {handle} is not live")))?;
        if seen_tags.insert(tag) {
            tags.push(tag);
        }
    }
    Ok(tags)
}

/// Resolves the bake's object frame: the root must be a live scene node with an
/// invertible world transform, returned with the node handle it was resolved through.
pub fn root_frame(
    root: u32,
    mut world_of: impl FnMut(u32) -> Option<Mat4>,
) -> Result<Mat4, RaaError> {
    let world =
        world_of(root).ok_or_else(|| invalid(format!("root {root} is not a live scene node")))?;
    world
        .inverse()
        .ok_or_else(|| invalid(format!("root {root} has no invertible world frame")))?;
    Ok(world)
}

/// The lifecycle state of one bake-store entry.
#[derive(Clone, Debug, PartialEq)]
pub enum BillboardBakeState {
    /// The GPU work is in flight; the bake cannot be selected.
    Pending,
    /// The complete bake was published.
    Ready(BillboardBake),
    /// The bake failed; no atlas image is owned by the store.
    Failed(String),
}

/// One bake-store entry.
#[derive(Clone, Debug, PartialEq)]
pub struct BillboardBakeEntry {
    pub id: u32,
    pub root: u32,
    pub state: BillboardBakeState,
}

impl BillboardBakeEntry {
    pub fn bake(&self) -> Option<&BillboardBake> {
        match &self.state {
            BillboardBakeState::Ready(bake) => Some(bake),
            _ => None,
        }
    }
}

/// What a completion did to the store.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Completion {
    /// The pending entry became ready.
    Published,
    /// The entry was failed or removed meanwhile: the caller must free these atlas ids.
    Stale([u32; 3]),
    /// The entry already failed or completed; no atlas id is owned.
    Discarded,
}

/// One step of a bake removal, in order: the group detach always precedes the atlas free.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RemovalStep {
    Detach(u32),
    Free(u32),
}

/// The registry of asynchronous bakes, keyed by the caller-facing bake id.
#[derive(Debug)]
pub struct BillboardBakeStore {
    next_id: u32,
    entries: BTreeMap<u32, BillboardBakeEntry>,
}

impl Default for BillboardBakeStore {
    fn default() -> Self {
        Self::new()
    }
}

impl BillboardBakeStore {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            entries: BTreeMap::new(),
        }
    }

    /// Allocates a pending entry for `root` and returns its id.
    pub fn begin(&mut self, root: u32) -> u32 {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        self.entries.insert(
            id,
            BillboardBakeEntry {
                id,
                root,
                state: BillboardBakeState::Pending,
            },
        );
        id
    }

    /// Publishes the complete bake of a still-pending entry. A removed/failed entry hands
    /// its freshly published atlas ids back for disposal and cannot be selected.
    pub fn complete(&mut self, id: u32, bake: BillboardBake) -> Completion {
        match self.entries.get_mut(&id) {
            Some(entry) if entry.state == BillboardBakeState::Pending => {
                entry.state = BillboardBakeState::Ready(bake);
                Completion::Published
            }
            Some(_) => Completion::Discarded,
            None => Completion::Stale(bake.atlas()),
        }
    }

    /// Marks a pending entry failed and returns whether it was still pending.
    pub fn fail(&mut self, id: u32, message: impl Into<String>) -> bool {
        match self.entries.get_mut(&id) {
            Some(entry) if entry.state == BillboardBakeState::Pending => {
                entry.state = BillboardBakeState::Failed(message.into());
                true
            }
            _ => false,
        }
    }

    /// The complete bake behind `id`, or None when the entry is pending, failed, removed
    /// or unknown.
    pub fn ready(&self, id: u32) -> Option<BillboardBake> {
        self.entries
            .get(&id)
            .and_then(|entry| entry.bake())
            .copied()
    }

    #[allow(dead_code)]
    pub fn is_ready(&self, id: u32) -> bool {
        self.ready(id).is_some()
    }

    /// The ids of every complete bake a group may resolve.
    pub fn ready_ids(&self) -> BTreeSet<u32> {
        self.entries
            .iter()
            .filter(|(_, entry)| matches!(entry.state, BillboardBakeState::Ready(_)))
            .map(|(&id, _)| id)
            .collect()
    }

    #[allow(dead_code)]
    pub fn state(&self, id: u32) -> Option<&BillboardBakeState> {
        self.entries.get(&id).map(|entry| &entry.state)
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Removes `id`, returning its entry. An unknown id refuses [`BILLBOARD_INVALID`].
    pub fn take(&mut self, id: u32) -> Result<BillboardBakeEntry, RaaError> {
        self.entries
            .remove(&id)
            .ok_or_else(|| invalid(format!("bake {id} is unknown or was removed")))
    }

    /// Device loss: every pending entry is failed and every published handle is
    /// invalidated. The device owns the released atlas images.
    pub fn invalidate(&mut self) {
        for entry in self.entries.values_mut() {
            entry.state = BillboardBakeState::Failed("the GPU device was lost".into());
        }
        self.entries.clear();
    }
}

/// Removes bake `id` in the contract's order: detach the dependent groups to LOD0 first,
/// then free the three atlas images. `step` observes every step once.
pub fn remove_bake(
    store: &mut BillboardBakeStore,
    id: u32,
    step: &mut dyn FnMut(RemovalStep),
) -> Result<BillboardBake, RaaError> {
    let entry = store.take(id)?;
    step(RemovalStep::Detach(id));
    let bake = match entry.state {
        BillboardBakeState::Ready(bake) => bake,
        _ => {
            return Err(invalid(format!("bake {id} is not complete")));
        }
    };
    for &image in &bake.atlas() {
        step(RemovalStep::Free(image));
    }
    Ok(bake)
}

/// The shared bake store kept in the engine's feature map, so the group preparation can
/// resolve a ready bake while the async API owns the pending entry.
pub type SharedBakeStore = Rc<RefCell<BillboardBakeStore>>;

/// The pending/failed gate the group preparation consults before it may select LOD1.
#[allow(dead_code)]
pub fn ready_for_groups(store: &SharedBakeStore, bake: u32) -> bool {
    store.borrow().is_ready(bake)
}

/// Owns a value taken out of a shared cell for the duration of an asynchronous operation
/// and puts it back when dropped, whether the operation succeeded, failed or its future
/// was cancelled. The cell stays empty during the await, so no `RefCell` borrow is ever
/// held across it. Dropping the guard is the only restoration path, so an error return
/// and a cancelled future take the same code as a success.
#[derive(Debug)]
pub struct StateGuard<T> {
    cell: Rc<RefCell<Option<T>>>,
    state: Option<T>,
}

impl<T> StateGuard<T> {
    /// Takes the value out of `cell`; `None` when the cell is already empty.
    pub fn take(cell: Rc<RefCell<Option<T>>>) -> Option<Self> {
        let state = cell.borrow_mut().take()?;
        Some(Self {
            cell,
            state: Some(state),
        })
    }

    #[allow(dead_code)]
    pub fn get(&self) -> Option<&T> {
        self.state.as_ref()
    }

    pub fn get_mut(&mut self) -> Option<&mut T> {
        self.state.as_mut()
    }
}

impl<T> Drop for StateGuard<T> {
    fn drop(&mut self) {
        let Some(state) = self.state.take() else {
            return;
        };
        if let Ok(mut guard) = self.cell.try_borrow_mut()
            && guard.is_none()
        {
            *guard = Some(state);
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) use web::{frame, install};

#[cfg(target_arch = "wasm32")]
mod web {
    use super::*;
    use crate::convert::{DISPOSED, api_error};
    use crate::engine::{Engine, InstallCtx};
    use crate::state::EngineState;
    use js_sys::{Array, Object, Reflect};
    use raa_core::GpuContext;
    use raa_lit::BillboardBakeInput;
    use raa_web::FrameInput;
    use wasm_bindgen::prelude::*;

    /// Installs the bake store in the engine's feature map.
    pub(crate) fn install(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
        ctx.features
            .insert(Rc::new(RefCell::new(BillboardBakeStore::new())));
        Ok(())
    }

    /// Device loss rejects every pending bake and invalidates the published handles; the
    /// device owns the released atlas images.
    pub(crate) fn frame(
        st: &mut EngineState,
        gpu: &GpuContext,
        _input: &FrameInput,
    ) -> Result<(), RaaError> {
        if gpu.ensure_live().is_err()
            && let Some(store) = st.features.get::<SharedBakeStore>()
        {
            store.borrow_mut().invalidate();
        }
        Ok(())
    }

    fn required(obj: &JsValue, key: &str) -> Result<JsValue, RaaError> {
        if !obj.is_object() {
            return Err(invalid("a billboard bake description must be an object"));
        }
        let v = Reflect::get(obj, &JsValue::from_str(key))
            .map_err(|_| invalid(format!("cannot read '{key}'")))?;
        if v.is_undefined() || v.is_null() {
            return Err(invalid(format!("'{key}' is required")));
        }
        Ok(v)
    }

    fn number(obj: &JsValue, key: &str) -> Result<f64, RaaError> {
        required(obj, key)?
            .as_f64()
            .ok_or_else(|| invalid(format!("'{key}' must be a number")))
    }

    fn handle_number(v: f64, name: &str) -> Result<u32, RaaError> {
        if v.fract() == 0.0 && (0.0..=f64::from(u32::MAX)).contains(&v) {
            Ok(v as u32)
        } else {
            Err(invalid(format!(
                "'{name}' must be a handle number, got {v}"
            )))
        }
    }

    fn instances_list(obj: &JsValue) -> Result<Vec<u32>, RaaError> {
        let v = required(obj, "instances")?;
        if !Array::is_array(&v) {
            return Err(invalid("'instances' must be an array of instance handles"));
        }
        let a: &Array = v.unchecked_ref();
        let mut out = Vec::with_capacity(a.length() as usize);
        for i in 0..a.length() {
            let item = a
                .get(i)
                .as_f64()
                .ok_or_else(|| invalid(format!("'instances[{i}]' must be a number")))?;
            out.push(handle_number(item, "instances[]")?);
        }
        Ok(out)
    }

    fn resolution_of(obj: &JsValue) -> Result<u32, RaaError> {
        let v = number(obj, "resolution")?;
        if v.fract() != 0.0 || !(0.0..=f64::from(u32::MAX)).contains(&v) {
            return Err(invalid(format!(
                "'resolution' must be a whole number, got {v}"
            )));
        }
        validate_resolution(v as u32)
    }

    /// The descriptor fields that need no engine state: validated before the state is
    /// taken, so a malformed descriptor never mutates the engine.
    struct BakeSpec {
        instances: Vec<u32>,
        root: u32,
        resolution: u32,
    }

    /// One validated bake request: the complete submitted tags, the root frame and the
    /// tile resolution. Nothing is mutated.
    struct BakeRequest {
        root: u32,
        root_world: Mat4,
        tags: Vec<u32>,
        resolution: u32,
    }

    fn parse_spec(desc: &JsValue) -> Result<BakeSpec, RaaError> {
        Ok(BakeSpec {
            instances: instances_list(desc)?,
            root: handle_number(number(desc, "root")?, "root")?,
            resolution: resolution_of(desc)?,
        })
    }

    /// Resolves the live handles of a validated [`BakeSpec`] against the taken state. A
    /// stale or unknown handle refuses before any GPU work or mutation.
    fn resolve_request(st: &EngineState, spec: BakeSpec) -> Result<BakeRequest, RaaError> {
        let root_world = root_frame(spec.root, |h| {
            st.nodes
                .get(h)
                .ok()
                .and_then(|node| st.scene.world(node).ok())
        })?;
        let tags = expand_tags(&spec.instances, |h| {
            st.instances
                .get_ref(h)
                .ok()
                .filter(|group| !group.stale)
                .map(|group| group.tag)
        })?;
        Ok(BakeRequest {
            root: spec.root,
            root_world,
            tags,
            resolution: spec.resolution,
        })
    }

    /// The contract's order for removing a bake: detach dependent groups to LOD0 first,
    /// then free the three atlas images.
    fn detach_and_free(st: &mut EngineState, gpu: &GpuContext, id: u32) -> Result<(), RaaError> {
        let store = st
            .features
            .get::<SharedBakeStore>()
            .cloned()
            .ok_or_else(|| invalid("the billboard bake feature is not installed"))?;
        let mut store = store.borrow_mut();
        let mut step = |step: RemovalStep| match step {
            RemovalStep::Detach(bake) => {
                crate::api::m36_billboard_groups::detach_bake_groups(st, bake);
            }
            RemovalStep::Free(image) => {
                let _ = st.lit.remove_texture(gpu, image);
            }
        };
        remove_bake(&mut store, id, &mut step)?;
        Ok(())
    }

    fn array(values: &[u32]) -> Array {
        values.iter().map(|&v| JsValue::from(v)).collect()
    }

    fn bake_value(bake: &BillboardBake) -> JsValue {
        let obj = Object::new();
        let set = |key: &str, value: JsValue| {
            let _ = Reflect::set(&obj, &JsValue::from_str(key), &value);
        };
        set("id", JsValue::from(bake.id));
        set("diffuse", JsValue::from(bake.diffuse));
        set("normal", JsValue::from(bake.normal));
        set("opacity", JsValue::from(bake.opacity));
        let center: Array = bake.center.iter().map(|&v| JsValue::from(v)).collect();
        set("center", center.into());
        let size: Array = bake.size.iter().map(|&v| JsValue::from(v)).collect();
        set("size", size.into());
        set("directions", JsValue::from(bake.directions));
        let tiles: Array = bake
            .tiles
            .iter()
            .map(|tile| JsValue::from(array(tile)))
            .collect();
        set("tiles", tiles.into());
        set("extent", array(&bake.extent).into());
        obj.into()
    }

    fn disposed() -> JsValue {
        error_to_js(&api_error(DISPOSED, "the engine was disposed"))
    }

    fn error_to_js(error: &RaaError) -> JsValue {
        raa_web::error_to_js(error)
    }

    #[wasm_bindgen(js_class = Engine)]
    impl Engine {
        /// Bakes the eight directions of `desc` and resolves with the complete
        /// `BillboardBake`; a pending bake is never selectable, and a failure rejects with
        /// `billboard.bake_failed` after releasing the partial atlas images.
        ///
        /// Handoff: "Add the engine billboard presentation bridge" takes over from
        /// `Engine.bakeBillboard` (this comment is anchored on its definition) and
        /// `Engine.setBillboardGroup`
        /// (`crates/raa-js/src/api/m36_billboard_groups.rs:854 set_billboard_group`). The
        /// store below publishes only a complete bake, which
        /// `prepare_billboard_groups` (`crates/raa-js/src/api/m36_billboard_groups.rs:593`)
        /// resolves through `SharedBakeStore`. The bridge builds its display quad with the
        /// Rust provider `raa_lit::billboard_proxy`
        /// (`crates/raa-lit/src/billboard_bake.rs:132`) and uploads it through
        /// `EngineState::upload_scratch` (`crates/raa-js/src/engine.rs:108`). Earlier
        /// handoffs this rests on: `BillboardGroupRegistry::set`
        /// (`crates/raa-js/src/api/m36_billboard_groups.rs:188`, handoff at `:170`) and
        /// `select_billboard` (`crates/raa-js/src/billboard_selection.rs:75`).
        #[wasm_bindgen(js_name = bakeBillboard)]
        pub async fn bake_billboard(&mut self, desc: JsValue) -> Result<JsValue, JsValue> {
            // Descriptor shape first: a malformed descriptor refuses with
            // `billboard.invalid` before the engine state is taken or mutated.
            let spec = parse_spec(&desc).map_err(|e| error_to_js(&e))?;
            let (gpu, rc) = self.bake_context()?;
            // The state is owned by the guard across the await; the cell is empty, so no
            // `RefCell` borrow is held. Dropping the guard (success, error or cancelled
            // future) restores the state.
            let mut restore = StateGuard::take(rc).ok_or_else(disposed)?;
            let st = restore.get_mut().expect("the bake state is present");
            st.scene.update_world();
            let request = resolve_request(st, spec).map_err(|e| error_to_js(&e))?;
            let store = st
                .features
                .get::<SharedBakeStore>()
                .cloned()
                .ok_or_else(|| {
                    error_to_js(&invalid("the billboard bake feature is not installed"))
                })?;
            let id = store.borrow_mut().begin(request.root);
            let input = BillboardBakeInput {
                tags: request.tags,
                root_world: request.root_world,
                resolution: request.resolution,
            };
            let result = st.lit.bake_billboard(gpu, &mut st.scene, input).await;
            match result {
                Ok(result) => {
                    let bake = BillboardBake::from_result(id, result);
                    let completion = store.borrow_mut().complete(id, bake);
                    match completion {
                        Completion::Published => Ok(bake_value(&bake)),
                        Completion::Stale(images) => {
                            for image in images {
                                let _ = st.lit.remove_texture(gpu, image);
                            }
                            Err(error_to_js(&invalid(
                                "the billboard bake was removed before it completed",
                            )))
                        }
                        Completion::Discarded => Err(error_to_js(&invalid(format!(
                            "billboard bake {id} already finished"
                        )))),
                    }
                }
                Err(error) => {
                    if error.code() == "core.gpu" {
                        // Device loss: reject every pending bake and invalidate every
                        // published handle at once; the device owns the stale images.
                        store.borrow_mut().invalidate();
                    } else {
                        store.borrow_mut().fail(id, error.to_string());
                    }
                    let code = if error.code() == BILLBOARD_INVALID {
                        BILLBOARD_INVALID
                    } else {
                        BILLBOARD_BAKE_FAILED
                    };
                    Err(error_to_js(&bake_error(code, error.to_string())))
                }
            }
        }

        /// Releases the complete bake `id`: dependent groups detach to LOD0 before the
        /// atlas images are freed. An unknown or stale id refuses `billboard.invalid`.
        #[wasm_bindgen(js_name = removeBillboardBake)]
        pub fn remove_billboard_bake(&mut self, id: u32) -> Result<(), JsValue> {
            self.with_state(|gpu, st| detach_and_free(st, gpu, id))
        }

        /// Uploads a caller-owned four-vertex proxy quad for the ready bake `bake` and
        /// returns its mesh handle. An invalid or stale bake refuses `billboard.invalid`
        /// before any allocation. `EngineState::upload_scratch` is the one upload path
        /// (mesh-source retention and scratch reuse included).
        ///
        /// Handoff: "Install and invalidate Game billboard presentations" takes over from
        /// `Engine.createBillboardProxy` (this comment is anchored on its definition). It
        /// receives a caller-owned mesh only: the quad comes from the Rust provider
        /// `raa_lit::billboard_proxy` (`crates/raa-lit/src/billboard_bake.rs:132`) and is
        /// uploaded through `EngineState::upload_scratch`
        /// (`crates/raa-js/src/engine.rs:108`); no instance or material is created. The
        /// bake store's `remove_bake`
        /// (`crates/raa-js/src/api/m37_billboard_bake.rs:300`) never removes that mesh,
        /// so the consumer disposes its instance before the mesh. Earlier handoffs this
        /// rests on: `BillboardGroupRegistry::set`
        /// (`crates/raa-js/src/api/m36_billboard_groups.rs:188`, handoff at `:170`) and the
        /// provider `Engine::bake_context` (`crates/raa-js/src/engine.rs:434`).
        #[wasm_bindgen(js_name = createBillboardProxy)]
        pub fn create_billboard_proxy(&mut self, bake: u32) -> Result<u32, JsValue> {
            self.with_state(|gpu, st| {
                let store = st
                    .features
                    .get::<SharedBakeStore>()
                    .cloned()
                    .ok_or_else(|| invalid("the billboard bake feature is not installed"))?;
                let ready = store.borrow().ready(bake).ok_or_else(|| {
                    invalid(format!("bake {bake} is not ready or no longer exists"))
                })?;
                st.scratch = raa_lit::billboard_proxy(ready.center, ready.size)?;
                st.upload_scratch(gpu)
            })
        }
    }
}
