//! m36 whole-tree billboard groups (M6): the shared Rust registry behind
//! `setBillboardGroup` and `billboardGroupState`, its install into `FeatureMap` and the
//! Engine registration/query adapters.
//!
//! Per-view preparation is the second share of `m6-group-registry-api`, owned by
//! "Prepare billboard groups for each camera": it runs for each view after that view's
//! camera/viewport are installed and before `lit.prepare`. Until the bake store is
//! attached every bake is unresolved, so preparation keeps LOD0 visible and reports
//! `ready == false`. This file supplies validated state registration and preparation,
//! and does not render.
//!
//! The registry and preparation core below compile on the host: `tests/billboard_groups.rs`
//! and `tests/billboard_group_prepare.rs` include this file with `#[path]`.

use std::collections::{BTreeMap, BTreeSet};

use raa_core::RaaError;
use raa_math::{Aabb, Mat4, Vec3};

// The landed selection handoff: `crate::billboard_selection` inside the wasm crate, the
// same module through the library crate in the host integration tests that include this
// file with `#[path]`.
#[cfg(target_arch = "wasm32")]
use crate::billboard_selection::select_billboard;
#[cfg(not(target_arch = "wasm32"))]
use raa_js::billboard_selection::select_billboard;

/// Stable code of every refusal from the group registry, the same code
/// `crate::billboard_selection::BILLBOARD_INVALID` carries (this file is compiled
/// standalone by `tests/billboard_groups.rs`, which has no `crate::billboard_selection`).
pub const BILLBOARD_INVALID: &str = "billboard.invalid";

fn invalid(message: impl Into<String>) -> RaaError {
    RaaError::FrameGraph {
        code: BILLBOARD_INVALID,
        message: message.into(),
    }
}

/// The requested level of a group: automatic switching, forced LOD0 or forced proxy.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BillboardLevel {
    Auto,
    Full,
    Proxy,
}

/// The validated settings of one billboard group.
#[derive(Clone, Debug, PartialEq)]
pub struct BillboardGroupConfig {
    /// JS scene-node handle shared by the group's instances.
    pub root: u32,
    /// The complete, unique LOD0 instance handles; never contains `proxy`.
    pub full: Vec<u32>,
    /// The proxy instance handle.
    pub proxy: u32,
    /// Bake-store id; this registry does not resolve it.
    pub bake: u32,
    /// Positive projected-size threshold in pixels.
    pub switch_pixels: f32,
    /// Whole-number switch bias in -3..=3.
    pub bias: i8,
    /// Requested level.
    pub level: BillboardLevel,
}

/// The last prepared status of a group. A fresh group is level 0, direction 0,
/// not ready and has no error.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BillboardGroupStatus {
    pub level: u8,
    pub direction: u8,
    pub ready: bool,
    pub error: Option<String>,
}

/// The hysteresis record of one viewport (0..=3).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BillboardViewState {
    pub level: u8,
    pub direction: u8,
}

/// One registered group: its settings, last status, per-view hysteresis records and the
/// caller visibility saved before any group override.
#[derive(Clone, Debug, PartialEq)]
pub struct BillboardGroupRecord {
    pub config: BillboardGroupConfig,
    pub status: BillboardGroupStatus,
    pub views: BTreeMap<u8, BillboardViewState>,
    pub original_visibility: BTreeMap<u32, bool>,
}

/// The private registry of whole-tree billboard groups, keyed by caller id.
#[derive(Debug, Default)]
pub struct BillboardGroupRegistry {
    groups: BTreeMap<String, BillboardGroupRecord>,
}

/// Validates one group against the live handles and the instances `taken` by the other
/// groups. Nothing is mutated; every refusal carries [`BILLBOARD_INVALID`].
pub(crate) fn validate_group(
    config: &BillboardGroupConfig,
    live_roots: &BTreeSet<u32>,
    live_instances: &BTreeMap<u32, bool>,
    taken: &BTreeSet<u32>,
) -> Result<(), RaaError> {
    if !live_roots.contains(&config.root) {
        return Err(invalid(format!(
            "root {} is not a live scene node",
            config.root
        )));
    }
    if config.full.is_empty() {
        return Err(invalid("full must hold at least one LOD0 instance"));
    }
    let mut seen = BTreeSet::new();
    for &handle in &config.full {
        if !live_instances.contains_key(&handle) {
            return Err(invalid(format!("full instance {handle} is not live")));
        }
        if !seen.insert(handle) {
            return Err(invalid(format!("full instance {handle} is listed twice")));
        }
        if handle == config.proxy {
            return Err(invalid(format!("full instance {handle} is also the proxy")));
        }
        if taken.contains(&handle) {
            return Err(invalid(format!(
                "instance {handle} belongs to another group"
            )));
        }
    }
    if !live_instances.contains_key(&config.proxy) {
        return Err(invalid(format!(
            "proxy instance {} is not live",
            config.proxy
        )));
    }
    if taken.contains(&config.proxy) {
        return Err(invalid(format!(
            "proxy instance {} belongs to another group",
            config.proxy
        )));
    }
    if !config.switch_pixels.is_finite() || config.switch_pixels <= 0.0 {
        return Err(invalid("switch_pixels must be finite and positive"));
    }
    if !(-3..=3).contains(&config.bias) {
        return Err(invalid("bias must be in -3..=3"));
    }
    Ok(())
}

fn validate_id(id: &str) -> Result<(), RaaError> {
    if id.is_empty() {
        Err(invalid("a group id must not be empty"))
    } else {
        Ok(())
    }
}

impl BillboardGroupRegistry {
    pub fn new() -> Self {
        Self {
            groups: BTreeMap::new(),
        }
    }

    /// Handoff: "Prepare billboard groups for each camera" takes over from
    /// [`BillboardGroupRegistry::set`], the symbol this comment is anchored on. It reads
    /// the saved visibility records in [`BillboardGroupRecord::original_visibility`] (one
    /// `handle -> caller visibility` entry per full and proxy instance, captured before
    /// any group override), the unresolved-bake semantics
    /// ([`BillboardGroupConfig::bake`] is a bake-store id this registry does not resolve,
    /// so a group stays `ready == false` until preparation validates it) and the
    /// LIFECYCLE sentence: "install creates BillboardGroupRegistry in FeatureMap; Engine
    /// destruction drops it. set captures caller visibility before overriding it;
    /// removal/replacement restores prior visibility before any new group preparation."
    /// The earlier handoff for the same downstream work is
    /// `crates/raa-js/src/billboard_selection.rs:61 select_billboard`.
    ///
    /// Today this validates `id`, the live root, the nonempty unique `full` list, the
    /// proxy handle, disjoint membership against every other group, `switch_pixels` and
    /// `bias` before any mutation; then it stores the record with an empty per-view map and
    /// a not-ready status and returns the prior handles and their saved visibility in
    /// ascending handle order for the Engine adapter to restore.
    pub fn set(
        &mut self,
        id: &str,
        config: Option<BillboardGroupConfig>,
        live_roots: &BTreeSet<u32>,
        live_instances: &BTreeMap<u32, bool>,
    ) -> Result<Vec<(u32, bool)>, RaaError> {
        validate_id(id)?;
        let Some(config) = config else {
            return Ok(self.remove(id));
        };
        let mut taken = BTreeSet::new();
        for (other_id, record) in &self.groups {
            if other_id == id {
                continue;
            }
            taken.extend(record.config.full.iter().copied());
            taken.insert(record.config.proxy);
        }
        validate_group(&config, live_roots, live_instances, &taken)?;
        let restored = self.remove(id);
        let original_visibility = config
            .full
            .iter()
            .copied()
            .chain(std::iter::once(config.proxy))
            .map(|handle| (handle, live_instances[&handle]))
            .collect();
        self.groups.insert(
            id.to_owned(),
            BillboardGroupRecord {
                config,
                status: BillboardGroupStatus::default(),
                views: BTreeMap::new(),
                original_visibility,
            },
        );
        Ok(restored)
    }

    pub fn get(&self, id: &str) -> Option<&BillboardGroupRecord> {
        self.groups.get(id)
    }

    /// Mutable record access for the host preparation tests; the wasm adapter takes the
    /// records through its own registry borrow.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn get_mut_for_test(&mut self, id: &str) -> &mut BillboardGroupRecord {
        self.groups.get_mut(id).expect("registered group")
    }

    fn remove(&mut self, id: &str) -> Vec<(u32, bool)> {
        self.groups.remove(id).map_or_else(Vec::new, |record| {
            record.original_visibility.into_iter().collect()
        })
    }
}

/// The installed camera, viewport and object frame one per-view preparation reads.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct BillboardPrepareRecipe {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y_rad: f32,
    pub near: f32,
    pub ortho_height: Option<f32>,
    /// Viewport size in backing pixels.
    pub viewport: (u32, u32),
    /// World transform of the group root (the bake's object frame).
    pub root_world: Mat4,
    /// World transform of the proxy instance (the parameters' local frame).
    pub proxy_world: Mat4,
    /// World bounds of the group's real LOD0 instances.
    pub bounds: Aabb,
}

/// The level/direction one view selected and the 16 proxy parameter words.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct BillboardPrepared {
    pub level: u8,
    pub direction: u8,
    /// `InstanceExtra.params`: right, direction, up, N00, then N row-major.
    pub params: [f32; 16],
}

fn check_view(view: u8) -> Result<(), RaaError> {
    if view > 3 {
        return Err(invalid(format!("view {view} must be within 0..=3")));
    }
    Ok(())
}

/// Projects the LOD0 bounds to a diameter in backing pixels for `recipe.viewport`.
fn projected_diameter(recipe: &BillboardPrepareRecipe) -> f32 {
    let height = recipe.viewport.1 as f32;
    let diagonal = recipe.bounds.diagonal();
    if height <= 0.0 || !diagonal.is_finite() || diagonal <= 0.0 {
        return 0.0;
    }
    if let Some(ortho) = recipe.ortho_height
        && ortho > 0.0
    {
        return diagonal * height / ortho;
    }
    let forward = (recipe.target - recipe.eye).normalize_or_zero();
    let distance = (recipe.bounds.center() - recipe.eye)
        .dot(forward)
        .max(recipe.near.max(1.0e-6));
    if !distance.is_finite() || distance <= 0.0 {
        return 0.0;
    }
    let focal = height / (2.0 * (0.5 * recipe.fov_y_rad).tan());
    (diagonal * focal / distance).max(0.0)
}

/// `v` in the rotation frame of `world` (a direction, not a point), normalized.
fn in_local_frame(world: &Mat4, v: Vec3) -> Vec3 {
    let inverse = world.inverse().unwrap_or(Mat4::IDENTITY);
    let local = inverse.transform_vector(v).normalize_or_zero();
    if local.length_squared() > 0.0 {
        local
    } else {
        v
    }
}

/// The camera-facing right/up pair expressed in the proxy's local frame, plus the
/// row-major object-to-world inverse-transpose normal matrix.
fn proxy_params(recipe: &BillboardPrepareRecipe, direction: u8) -> [f32; 16] {
    let forward = (recipe.target - recipe.eye).normalize_or_zero();
    let mut right = forward.cross(recipe.up);
    if right.length_squared() <= 0.0 {
        right = Vec3::X;
    }
    right = right.normalize_or_zero();
    let mut up = right.cross(forward);
    if up.length_squared() <= 0.0 {
        up = recipe.up;
    }
    up = up.normalize_or_zero();
    let right = in_local_frame(&recipe.proxy_world, right);
    let up = in_local_frame(&recipe.proxy_world, up);
    // `normal_matrix` returns the columns of inverse(world).transpose(); the contract
    // stores the normal matrix by rows.
    let n = recipe.proxy_world.normal_matrix();
    let cell = |row: usize, col: usize| n[col][row];
    let mut p = [0.0; 16];
    p[..3].copy_from_slice(&right.to_array());
    p[3] = f32::from(direction);
    p[4..7].copy_from_slice(&up.to_array());
    p[7] = cell(0, 0);
    p[8] = cell(0, 1);
    p[9] = cell(0, 2);
    p[10] = cell(1, 0);
    p[11] = cell(1, 1);
    p[12] = cell(1, 2);
    p[13] = cell(2, 0);
    p[14] = cell(2, 1);
    p[15] = cell(2, 2);
    p
}

/// Prepares one group for one view: selects the direction and (when the bake is
/// resolved) the level through [`select_billboard`], records this view's hysteresis and
/// returns the proxy parameters. `bake_ready == false` forces LOD0 and `ready == false`.
pub(crate) fn prepare_billboard_record(
    record: &mut BillboardGroupRecord,
    view: u8,
    recipe: &BillboardPrepareRecipe,
    bake_ready: bool,
) -> Result<BillboardPrepared, RaaError> {
    check_view(view)?;
    let root_inverse = recipe
        .root_world
        .inverse()
        .ok_or_else(|| invalid("the billboard group root frame is not invertible"))?;
    let camera_local = root_inverse.transform_point(recipe.eye).to_array();
    let projected = projected_diameter(recipe);
    let forced = match record.config.level {
        BillboardLevel::Auto => None,
        BillboardLevel::Full => Some(0),
        BillboardLevel::Proxy => Some(1),
    };
    let previous = record.views.get(&view).map_or(0, |state| state.level);
    let (selected, direction) = select_billboard(
        camera_local,
        projected,
        record.config.switch_pixels,
        record.config.bias,
        previous,
        forced,
    )?;
    let level = if bake_ready { selected } else { 0 };
    record
        .views
        .insert(view, BillboardViewState { level, direction });
    record.status = BillboardGroupStatus {
        level,
        direction,
        ready: bake_ready,
        error: None,
    };
    Ok(BillboardPrepared {
        level,
        direction,
        params: proxy_params(recipe, direction),
    })
}

/// Records a failed preparation for one view: LOD0 stays selected, the status is not
/// ready and carries the diagnostic.
pub(crate) fn fail_billboard_record(
    record: &mut BillboardGroupRecord,
    view: u8,
    message: impl Into<String>,
) -> Result<(), RaaError> {
    check_view(view)?;
    let direction = record
        .views
        .get(&view)
        .map_or(record.status.direction, |state| state.direction);
    record.views.insert(
        view,
        BillboardViewState {
            level: 0,
            direction,
        },
    );
    record.status = BillboardGroupStatus {
        level: 0,
        direction,
        ready: false,
        error: Some(message.into()),
    };
    Ok(())
}

#[cfg(target_arch = "wasm32")]
pub(crate) use web::{frame, install, prepare_billboard_groups};

#[cfg(target_arch = "wasm32")]
mod web {
    use super::*;
    use crate::api::ViewsState;
    use crate::convert::HandleMap;
    use crate::engine::{Engine, InstallCtx};
    use crate::state::{EngineState, InstanceGroup};
    use js_sys::{Array, Object, Reflect};
    use raa_core::GpuContext;
    use raa_lit::LitRenderer;
    use raa_scene::{NodeId, Scene};
    use raa_web::FrameInput;
    use wasm_bindgen::{JsCast, prelude::*};

    /// The canvas backing size of the current invocation, read by the per-view
    /// preparation to turn a projected world size into pixels.
    #[derive(Copy, Clone, Debug, Default)]
    struct BillboardFrameSize((u32, u32));

    pub(crate) fn install(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
        ctx.features.insert(BillboardGroupRegistry::new());
        ctx.features.insert(BillboardFrameSize::default());
        Ok(())
    }

    /// Records the frame's canvas size; preparation itself runs at the lifecycle hook.
    pub(crate) fn frame(
        st: &mut EngineState,
        _gpu: &GpuContext,
        input: &FrameInput,
    ) -> Result<(), RaaError> {
        st.features.insert(BillboardFrameSize(input.size));
        Ok(())
    }

    /// The backing-pixel size of `view`: its viewport rectangle of the canvas size.
    fn viewport_size(st: &EngineState, view: usize) -> (u32, u32) {
        let canvas = st
            .features
            .get::<BillboardFrameSize>()
            .map_or((0, 0), |size| size.0);
        let Some(rect) = st
            .features
            .get::<ViewsState>()
            .and_then(|views| views.viewports().get(view))
        else {
            return canvas;
        };
        let (w, h) = (f64::from(canvas.0), f64::from(canvas.1));
        let x = (rect.x * w).round() as u32;
        let y = (rect.y * h).round() as u32;
        let right = ((rect.x + rect.width) * w).round() as u32;
        let bottom = ((rect.y + rect.height) * h).round() as u32;
        (right.saturating_sub(x), bottom.saturating_sub(y))
    }

    struct GroupResources {
        root_world: Mat4,
        proxy_world: Mat4,
        bounds: Aabb,
    }

    fn resolve_resources(
        nodes: &HandleMap<NodeId>,
        instances: &HandleMap<InstanceGroup>,
        lit: &LitRenderer,
        scene: &Scene,
        config: &BillboardGroupConfig,
    ) -> Result<GroupResources, String> {
        let root = *nodes
            .get_ref(config.root)
            .map_err(|_| format!("root {} is not a live scene node", config.root))?;
        let root_world = scene
            .world(root)
            .map_err(|_| format!("root {} has no world transform", config.root))?;
        let proxy = instances
            .get_ref(config.proxy)
            .map_err(|_| format!("proxy instance {} is not live", config.proxy))?;
        if proxy.stale {
            return Err(format!("proxy instance {} is stale", config.proxy));
        }
        let proxy_world = scene
            .world(proxy.node)
            .map_err(|_| format!("proxy instance {} has no world transform", config.proxy))?;
        let mut bounds = Aabb::EMPTY;
        for &handle in &config.full {
            let group = instances
                .get_ref(handle)
                .map_err(|_| format!("LOD0 instance {handle} is not live"))?;
            if group.stale {
                return Err(format!("LOD0 instance {handle} is stale"));
            }
            for &id in &group.instances {
                let Some(slot) = lit.handle().instance_slot(id) else {
                    continue;
                };
                if let Some(Some(world)) = scene.instance_index().bounds(slot) {
                    bounds = bounds.union(&world);
                }
            }
        }
        if !bounds.diagonal().is_finite() {
            let p = proxy_world.transform_point(Vec3::ZERO);
            bounds = Aabb { min: p, max: p };
        }
        Ok(GroupResources {
            root_world,
            proxy_world,
            bounds,
        })
    }

    /// Switches the whole group: every live LOD0 instance is shown and the proxy hidden
    /// for LOD0, and the reverse for LOD1. The caller visibility saved by `set` is restored
    /// when the group is detached.
    fn apply_level(
        scene: &mut Scene,
        instances: &mut HandleMap<InstanceGroup>,
        record: &BillboardGroupRecord,
        level: u8,
    ) {
        for &handle in &record.config.full {
            if let Ok(group) = instances.get_mut(handle) {
                let _ = group.set_visible(scene, level == 0);
            }
        }
        let proxy = record.config.proxy;
        if let Ok(group) = instances.get_mut(proxy) {
            let _ = group.set_visible(scene, level == 1);
        }
    }

    /// Writes the 16 reserved parameter words of the proxy instance, keeping tint, squash,
    /// jiggle and the S23 tint flag.
    fn apply_params(
        lit: &mut LitRenderer,
        instances: &HandleMap<InstanceGroup>,
        handle: u32,
        params: [f32; 16],
    ) {
        let Ok(group) = instances.get_ref(handle) else {
            return;
        };
        let mut extra = lit.instance_extra(group.tag);
        extra.params = [
            [params[0], params[1], params[2], params[3]],
            [params[4], params[5], params[6], params[7]],
            [params[8], params[9], params[10], params[11]],
            [params[12], params[13], params[14], params[15]],
        ];
        lit.set_instance_extra(group.tag, extra);
    }

    /// Second share of `m6-group-registry-api`: prepares every group for `view` after the
    /// view's camera/viewport are installed and before `lit.prepare`.
    ///
    /// It reads the registry records created by [`BillboardGroupRegistry::set`], resolves
    /// the live root, LOD0 and proxy instances, computes the facing basis, projected bounds,
    /// inverse-transpose normal matrix and direction in Rust, calls `select_billboard`,
    /// updates only this view's hysteresis record and the last status, then switches LOD0
    /// instances against the proxy and writes the proxy parameter words. An unresolved bake
    /// forces LOD0 with `ready == false`; missing/stale scene resources keep the live LOD0
    /// instances visible and record a diagnostic. Invalid view indices refuse before any
    /// mutation. Bake targets and atlas images are never touched here.
    pub(crate) fn prepare_billboard_groups(st: &mut EngineState, view: u8) -> Result<(), RaaError> {
        if view > 3 {
            return Err(invalid(format!("view {view} must be within 0..=3")));
        }
        let viewport = viewport_size(st, usize::from(view));
        let camera = st.scene.camera;
        st.scene.update_world();
        let EngineState {
            scene,
            instances,
            nodes,
            lit,
            features,
            ..
        } = st;
        let Some(registry) = features.get_mut::<BillboardGroupRegistry>() else {
            return Ok(());
        };
        for record in registry.groups.values_mut() {
            match resolve_resources(nodes, instances, lit, scene, &record.config) {
                Ok(resources) => {
                    let recipe = BillboardPrepareRecipe {
                        eye: camera.eye,
                        target: camera.target,
                        up: camera.up,
                        fov_y_rad: camera.fov_y_rad,
                        near: camera.near,
                        ortho_height: camera.ortho_height,
                        viewport,
                        root_world: resources.root_world,
                        proxy_world: resources.proxy_world,
                        bounds: resources.bounds,
                    };
                    // The resolved-bake store is attached by "Expose asynchronous
                    // billboard baking in RAA"; until then every bake is unresolved.
                    let ready = false;
                    match prepare_billboard_record(record, view, &recipe, ready) {
                        Ok(prepared) => {
                            apply_level(scene, instances, record, prepared.level);
                            apply_params(lit, instances, record.config.proxy, prepared.params);
                        }
                        Err(error) => {
                            let _ = fail_billboard_record(record, view, error.to_string());
                            apply_level(scene, instances, record, 0);
                        }
                    }
                }
                Err(message) => {
                    let _ = fail_billboard_record(record, view, message);
                    apply_level(scene, instances, record, 0);
                }
            }
        }
        Ok(())
    }

    impl EngineState {
        /// The engine's full-canvas hook: the free [`prepare_billboard_groups`] entry
        /// point, reachable from `engine.rs` across the private api module.
        pub(crate) fn prepare_billboard_groups(&mut self, view: u8) -> Result<(), RaaError> {
            prepare_billboard_groups(self, view)
        }
    }

    fn required(obj: &JsValue, key: &str) -> Result<JsValue, RaaError> {
        if !obj.is_object() {
            return Err(invalid("a billboard group description must be an object"));
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

    fn handle(obj: &JsValue, key: &str) -> Result<u32, RaaError> {
        handle_number(number(obj, key)?, key)
    }

    fn full_list(obj: &JsValue) -> Result<Vec<u32>, RaaError> {
        let v = required(obj, "full")?;
        if !Array::is_array(&v) {
            return Err(invalid("'full' must be an array of instance handles"));
        }
        let a: &Array = v.unchecked_ref();
        let mut out = Vec::with_capacity(a.length() as usize);
        for i in 0..a.length() {
            let item = a
                .get(i)
                .as_f64()
                .ok_or_else(|| invalid(format!("'full[{i}]' must be a number")))?;
            out.push(handle_number(item, "full[]")?);
        }
        Ok(out)
    }

    fn level(obj: &JsValue) -> Result<BillboardLevel, RaaError> {
        let v = required(obj, "level")?;
        if let Some(s) = v.as_string() {
            if s == "auto" {
                return Ok(BillboardLevel::Auto);
            }
        } else if let Some(n) = v.as_f64() {
            if n == 0.0 {
                return Ok(BillboardLevel::Full);
            }
            if n == 1.0 {
                return Ok(BillboardLevel::Proxy);
            }
        }
        Err(invalid("'level' must be 'auto', 0 or 1"))
    }

    fn parse_desc(desc: &JsValue) -> Result<Option<BillboardGroupConfig>, RaaError> {
        if desc.is_null() || desc.is_undefined() {
            return Ok(None);
        }
        let bias = number(desc, "bias")?;
        if bias.fract() != 0.0 || !(-3.0..=3.0).contains(&bias) {
            return Err(invalid(format!(
                "'bias' must be a whole number from -3 to 3, got {bias}"
            )));
        }
        Ok(Some(BillboardGroupConfig {
            root: handle(desc, "root")?,
            full: full_list(desc)?,
            proxy: handle(desc, "proxy")?,
            bake: handle(desc, "bake")?,
            switch_pixels: number(desc, "switchPixels")? as f32,
            bias: bias as i8,
            level: level(desc)?,
        }))
    }

    fn live_roots(st: &EngineState) -> BTreeSet<u32> {
        st.nodes.iter().map(|(handle, _)| handle).collect()
    }

    /// Caller visibility before any group override: the values saved by an earlier
    /// `set` win over the visibility the preparation currently applies.
    fn live_instances(st: &EngineState) -> BTreeMap<u32, bool> {
        let mut instances: BTreeMap<u32, bool> = st
            .instances
            .iter()
            .filter(|(_, group)| !group.stale)
            .map(|(handle, group)| (handle, group.visible))
            .collect();
        if let Some(registry) = st.features.get::<BillboardGroupRegistry>() {
            for record in registry.groups.values() {
                for (&handle, &visible) in &record.original_visibility {
                    if let Some(slot) = instances.get_mut(&handle) {
                        *slot = visible;
                    }
                }
            }
        }
        instances
    }

    /// Restores the caller visibility the registry saved for the prior group. An instance
    /// the caller has removed since has nothing to restore.
    fn restore(st: &mut EngineState, prior: Vec<(u32, bool)>) {
        for (handle, visible) in prior {
            if let Ok(group) = st.instances.get_mut(handle) {
                let _ = group.set_visible(&mut st.scene, visible);
            }
        }
    }

    fn apply(st: &mut EngineState, id: &str, desc: &JsValue) -> Result<(), RaaError> {
        let config = parse_desc(desc)?;
        let roots = live_roots(st);
        let instances = live_instances(st);
        let prior = st
            .features
            .require_mut::<BillboardGroupRegistry>("billboard groups")?
            .set(id, config, &roots, &instances)?;
        restore(st, prior);
        Ok(())
    }

    fn set_field(obj: &Object, key: &str, value: JsValue) {
        let _ = Reflect::set(obj, &JsValue::from_str(key), &value);
    }

    fn state_value(record: &BillboardGroupRecord) -> JsValue {
        let obj = Object::new();
        set_field(
            &obj,
            "level",
            JsValue::from_f64(f64::from(record.status.level)),
        );
        set_field(
            &obj,
            "direction",
            JsValue::from_f64(f64::from(record.status.direction)),
        );
        set_field(&obj, "ready", JsValue::from_bool(record.status.ready));
        set_field(
            &obj,
            "error",
            match &record.status.error {
                Some(error) => JsValue::from_str(error),
                None => JsValue::NULL,
            },
        );
        obj.into()
    }

    #[wasm_bindgen(js_class = Engine)]
    impl Engine {
        /// Registers or replaces the whole-tree billboard group `id`, or removes it with a
        /// null description. A refusal (`billboard.invalid`) changes nothing.
        #[wasm_bindgen(js_name = setBillboardGroup)]
        pub fn set_billboard_group(&mut self, id: &str, desc: JsValue) -> Result<(), JsValue> {
            self.with_state(|_, st| apply(st, id, &desc))
        }

        /// The last prepared status of `id`, or null when no such group is registered.
        #[wasm_bindgen(js_name = billboardGroupState)]
        pub fn billboard_group_state(&self, id: &str) -> Result<JsValue, JsValue> {
            self.with_state(|_, st| {
                Ok(st
                    .features
                    .get::<BillboardGroupRegistry>()
                    .and_then(|registry| registry.get(id))
                    .map_or(JsValue::NULL, state_value))
            })
        }
    }
}
