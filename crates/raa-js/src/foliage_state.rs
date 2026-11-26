//! M7 foliage generations and whole-tree representation groups (`m7-foliage-state`).
//!
//! This module owns the native, generation-keyed published-asset registry and the
//! string-keyed whole-tree groups of `m7-public-api`. Foliage asset ids come from their
//! own counter, so the m6 billboard bake ids in `m37_billboard_bake` are never reused.
//! The pure registry and preparation core below compile on the host; the wasm adapter is
//! only compiled for wasm32 and resolves live scene handles.
//!
//! LIFECYCLE (contracts m7-bake-service, m7-public-api): a generation is pending until its
//! preparation settles, then ready or failed; a stale completion is dropped. Removal
//! detaches dependent groups and the material association before the owned proxy mesh
//! and volume are released; the borrowed LOD1 material/data lease stays caller-owned.

use std::collections::{BTreeMap, BTreeSet};

use raa_core::RaaError;
use raa_math::{Aabb, Mat4, Vec3};

use crate::billboard_selection::select_billboard;

/// Stable code of every refusal from the foliage generation/group state.
pub const FOLIAGE_INVALID: &str = "foliage.invalid";

fn invalid(message: impl Into<String>) -> RaaError {
    RaaError::FrameGraph {
        code: FOLIAGE_INVALID,
        message: message.into(),
    }
}

/// The phase of one whole-tree foliage group (`FoliageState.phase`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum FoliagePhase {
    #[default]
    Pending,
    Ready,
    Failed,
}

/// The published state of one whole-tree foliage group (m7-public-api `FoliageState`).
///
/// Handoff: 'Expose asynchronous foliage preparation to JavaScript' reads
/// [`FoliageState`] in [`FoliageGroups::get`] to answer `foliageGroupState` and to report
/// the phase reached by a completed preparation. Today this carries the group's phase,
/// actual prepared generation, published asset id (or `None`), native mode rank, selected
/// level and failure diagnostic; it holds no GPU types so the API layer can map it to JS.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FoliageState {
    pub phase: FoliagePhase,
    pub generation: u32,
    pub asset: Option<u32>,
    pub mode: u32,
    pub level: u8,
    pub error: Option<String>,
}

/// One generation-owned published foliage asset (`FoliageAssetInfo` minus the JS mapping).
///
/// `proxy` is the scene instance of the asset's proxy mesh, published by the asynchronous
/// preparation card; groups hide it until the asset is ready. `mesh` and `material` are
/// the native handles `FoliageAsset` carries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoliageAssetEntry {
    pub id: u32,
    pub generation: u32,
    pub phase: FoliagePhase,
    pub mode: u32,
    pub volume_resolution: u32,
    pub mesh: u32,
    pub proxy: u32,
    pub material: u32,
    pub triangle_count: u32,
    pub error: Option<String>,
}

/// The generation-owned asset registry. Ids are allocated here and independent of the m6
/// billboard bake namespace.
#[derive(Debug)]
pub struct FoliageAssets {
    entries: BTreeMap<u32, FoliageAssetEntry>,
    next_id: u32,
}

impl Default for FoliageAssets {
    fn default() -> Self {
        Self::new()
    }
}

impl FoliageAssets {
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            next_id: 1,
        }
    }

    pub fn get(&self, id: u32) -> Option<&FoliageAssetEntry> {
        self.entries.get(&id)
    }

    pub fn contains(&self, id: u32) -> bool {
        self.entries.contains_key(&id)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Starts a new generation and returns its freshly allocated asset id.
    pub fn begin(
        &mut self,
        generation: u32,
        mode: u32,
        volume_resolution: u32,
        material: u32,
    ) -> u32 {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        self.entries.insert(
            id,
            FoliageAssetEntry {
                id,
                generation,
                phase: FoliagePhase::Pending,
                mode,
                volume_resolution,
                mesh: 0,
                proxy: 0,
                material,
                triangle_count: 0,
                error: None,
            },
        );
        id
    }

    /// Replaces the generation of an existing id with a fresh pending generation.
    pub fn replace(
        &mut self,
        id: u32,
        generation: u32,
        mode: u32,
        volume_resolution: u32,
        material: u32,
    ) -> Result<(), RaaError> {
        let entry = self
            .entries
            .get_mut(&id)
            .ok_or_else(|| invalid(format!("foliage asset {id} is not published")))?;
        *entry = FoliageAssetEntry {
            id,
            generation,
            phase: FoliagePhase::Pending,
            mode,
            volume_resolution,
            mesh: 0,
            proxy: 0,
            material,
            triangle_count: 0,
            error: None,
        };
        Ok(())
    }

    /// Publishes the result of `generation`. A completion for a replaced or unknown
    /// generation is stale: it is ignored and returns `false`.
    pub fn complete(
        &mut self,
        id: u32,
        generation: u32,
        mesh: u32,
        proxy: u32,
        triangle_count: u32,
    ) -> bool {
        let Some(entry) = self.entries.get_mut(&id) else {
            return false;
        };
        if entry.generation != generation || entry.phase != FoliagePhase::Pending {
            return false;
        }
        entry.phase = FoliagePhase::Ready;
        entry.mesh = mesh;
        entry.proxy = proxy;
        entry.triangle_count = triangle_count;
        entry.error = None;
        true
    }

    /// Records a failure of `generation`; a replaced generation is stale and ignored.
    pub fn fail(&mut self, id: u32, generation: u32, error: impl Into<String>) -> bool {
        let Some(entry) = self.entries.get_mut(&id) else {
            return false;
        };
        if entry.generation != generation || entry.phase != FoliagePhase::Pending {
            return false;
        }
        entry.phase = FoliagePhase::Failed;
        entry.error = Some(error.into());
        true
    }

    pub fn remove(&mut self, id: u32) -> Option<FoliageAssetEntry> {
        self.entries.remove(&id)
    }
}

/// The requested level of a whole-tree group: automatic switching, forced source or forced
/// proxy. A forced proxy keeps the source visible while the asset is not ready.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FoliageLevel {
    Auto,
    Full,
    Proxy,
}

/// The validated settings of one whole-tree group (`FoliageGroupDesc` native form).
#[derive(Clone, Debug, PartialEq)]
pub struct FoliageGroupConfig {
    pub asset: u32,
    /// The live source foliage instance handles; never contains the proxy.
    pub full: Vec<u32>,
    /// The live source-root scene node handle.
    pub root: u32,
    /// Positive projected-size threshold in pixels.
    pub switch_pixels: f32,
    pub level: FoliageLevel,
}

/// The visibility one group applies to its source and proxy instances.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FoliageVisibility {
    pub source: bool,
    pub proxy: bool,
}

/// Source and proxy are mutually exclusive only when the asset is ready: while pending or
/// failed the source draws stay visible and the proxy stays hidden.
pub fn foliage_visibility(phase: FoliagePhase, level: u8) -> FoliageVisibility {
    let proxy = phase == FoliagePhase::Ready && level == 1;
    FoliageVisibility {
        source: !proxy,
        proxy,
    }
}

/// One registered group: its settings, last prepared state, per-view level record and the
/// caller visibility saved before any override.
#[derive(Clone, Debug, PartialEq)]
pub struct FoliageGroupRecord {
    pub config: FoliageGroupConfig,
    pub state: FoliageState,
    pub views: BTreeMap<u8, u8>,
    pub original_visibility: BTreeMap<u32, bool>,
}

/// The string-keyed registry of whole-tree foliage groups.
#[derive(Debug, Default)]
pub struct FoliageGroups {
    groups: BTreeMap<String, FoliageGroupRecord>,
}

impl FoliageGroups {
    pub fn new() -> Self {
        Self {
            groups: BTreeMap::new(),
        }
    }

    /// Registers or replaces `id`, or removes it with `None`. Validation runs before any
    /// mutation and returns the prior handles with their saved visibility in ascending
    /// handle order for the caller to restore.
    ///
    /// # Errors
    ///
    /// The root must be live, `full` nonempty, unique and live, `switch_pixels` finite and
    /// positive; every refusal carries [`FOLIAGE_INVALID`].
    pub fn set(
        &mut self,
        id: &str,
        config: Option<FoliageGroupConfig>,
        live_roots: &BTreeSet<u32>,
        live_instances: &BTreeMap<u32, bool>,
    ) -> Result<Vec<(u32, bool)>, RaaError> {
        if id.is_empty() {
            return Err(invalid("a foliage group id must not be empty"));
        }
        let Some(config) = config else {
            return Ok(self.remove(id));
        };
        validate_group(&config, live_roots, live_instances)?;
        let restored = self.remove(id);
        let original_visibility = config
            .full
            .iter()
            .map(|&handle| (handle, live_instances[&handle]))
            .collect();
        self.groups.insert(
            id.to_owned(),
            FoliageGroupRecord {
                config,
                state: FoliageState::default(),
                views: BTreeMap::new(),
                original_visibility,
            },
        );
        Ok(restored)
    }

    pub fn get(&self, id: &str) -> Option<&FoliageGroupRecord> {
        self.groups.get(id)
    }

    pub fn get_mut(&mut self, id: &str) -> Option<&mut FoliageGroupRecord> {
        self.groups.get_mut(id)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &FoliageGroupRecord)> {
        self.groups.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&String, &mut FoliageGroupRecord)> {
        self.groups.iter_mut()
    }

    /// Removes `id` and returns its saved visibility pairs, or an empty list.
    pub fn remove(&mut self, id: &str) -> Vec<(u32, bool)> {
        self.groups.remove(id).map_or_else(Vec::new, |record| {
            record.original_visibility.into_iter().collect()
        })
    }

    /// Removes every group that references `asset` and returns their saved visibility.
    /// The caller releases the asset only after this detach, so no group keeps a dead id.
    pub fn remove_asset(&mut self, asset: u32) -> Vec<(u32, bool)> {
        let ids: Vec<String> = self
            .groups
            .iter()
            .filter(|(_, record)| record.config.asset == asset)
            .map(|(id, _)| id.clone())
            .collect();
        let mut restored = Vec::new();
        for id in ids {
            restored.extend(self.remove(&id));
        }
        restored
    }
}

fn validate_group(
    config: &FoliageGroupConfig,
    live_roots: &BTreeSet<u32>,
    live_instances: &BTreeMap<u32, bool>,
) -> Result<(), RaaError> {
    if !live_roots.contains(&config.root) {
        return Err(invalid(format!(
            "root {} is not a live scene node",
            config.root
        )));
    }
    if config.full.is_empty() {
        return Err(invalid("full must hold at least one source instance"));
    }
    let mut seen = BTreeSet::new();
    for &handle in &config.full {
        if !live_instances.contains_key(&handle) {
            return Err(invalid(format!("source instance {handle} is not live")));
        }
        if !seen.insert(handle) {
            return Err(invalid(format!("source instance {handle} is listed twice")));
        }
    }
    if !config.switch_pixels.is_finite() || config.switch_pixels <= 0.0 {
        return Err(invalid("switch_pixels must be finite and positive"));
    }
    Ok(())
}

/// The installed camera, viewport and source-root frame one per-view preparation reads.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FoliagePrepareRecipe {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y_rad: f32,
    pub near: f32,
    pub ortho_height: Option<f32>,
    /// Viewport size in backing pixels.
    pub viewport: (u32, u32),
    /// World transform of the source-root node.
    pub root_world: Mat4,
    /// World bounds of the group's real source instances.
    pub bounds: Aabb,
}

fn check_view(view: u8) -> Result<(), RaaError> {
    if view > 3 {
        return Err(invalid(format!("view {view} must be within 0..=3")));
    }
    Ok(())
}

/// Projects the source bounds to a diameter in backing pixels for `recipe.viewport`.
fn projected_diameter(recipe: &FoliagePrepareRecipe) -> f32 {
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

/// Prepares one group for one view: the level from the shared whole-object projected-size
/// selection, then the state dictated by the published asset. The selected billboard
/// direction is deliberately discarded: hull proxies never receive m6 facing rotation.
pub fn prepare_foliage_record(
    record: &mut FoliageGroupRecord,
    view: u8,
    recipe: &FoliagePrepareRecipe,
    asset: Option<&FoliageAssetEntry>,
) -> Result<u8, RaaError> {
    check_view(view)?;
    let root_inverse = recipe
        .root_world
        .inverse()
        .ok_or_else(|| invalid("the foliage group root frame is not invertible"))?;
    let camera_local = root_inverse.transform_point(recipe.eye).to_array();
    let projected = projected_diameter(recipe);
    let forced = match record.config.level {
        FoliageLevel::Auto => None,
        FoliageLevel::Full => Some(0),
        FoliageLevel::Proxy => Some(1),
    };
    let previous = record.views.get(&view).copied().unwrap_or(0);
    let (selected, _direction) = select_billboard(
        camera_local,
        projected,
        record.config.switch_pixels,
        0,
        previous,
        forced,
    )?;
    let level = match asset {
        None => {
            record.state = FoliageState {
                phase: FoliagePhase::Failed,
                generation: record.state.generation,
                asset: None,
                mode: record.state.mode,
                level: 0,
                error: Some(format!(
                    "foliage asset {} is not published",
                    record.config.asset
                )),
            };
            0
        }
        Some(entry) => {
            let level = if entry.phase == FoliagePhase::Ready {
                selected
            } else {
                0
            };
            record.state = FoliageState {
                phase: entry.phase,
                generation: entry.generation,
                asset: (entry.phase == FoliagePhase::Ready).then_some(entry.id),
                mode: entry.mode,
                level,
                error: entry.error.clone(),
            };
            level
        }
    };
    record.views.insert(view, level);
    Ok(level)
}

/// Records a failed preparation for one view: the live source stays selected and the
/// diagnostic is kept for `foliageGroupState`.
pub fn fail_foliage_record(
    record: &mut FoliageGroupRecord,
    view: u8,
    message: impl Into<String>,
) -> Result<(), RaaError> {
    check_view(view)?;
    record.views.insert(view, 0);
    record.state = FoliageState {
        phase: FoliagePhase::Failed,
        generation: record.state.generation,
        asset: None,
        mode: record.state.mode,
        level: 0,
        error: Some(message.into()),
    };
    Ok(())
}

/// The one feature value the engine installs: generation-owned assets plus the groups.
#[derive(Debug, Default)]
pub struct FoliageStateStore {
    pub assets: FoliageAssets,
    pub groups: FoliageGroups,
}

impl FoliageStateStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn state(&self, id: &str) -> Option<&FoliageState> {
        self.groups.get(id).map(|record| &record.state)
    }

    /// Removes `asset` and detaches every dependent group, returning the removed asset and
    /// the visibility pairs the caller restores before releasing the proxy mesh.
    pub fn remove_asset(&mut self, asset: u32) -> (Option<FoliageAssetEntry>, Vec<(u32, bool)>) {
        let removed = self.assets.remove(asset);
        let restored = self.groups.remove_asset(asset);
        (removed, restored)
    }
}

#[cfg(target_arch = "wasm32")]
#[allow(
    unused_imports,
    reason = "the wasm lifecycle hooks are called by the engine card"
)]
pub(crate) use web::{prepare_foliage_groups, set_foliage_frame_size};

#[cfg(target_arch = "wasm32")]
#[allow(dead_code)]
mod web {
    use super::*;
    use crate::api::ViewsState;
    use crate::convert::HandleMap;
    use crate::state::{EngineState, InstanceGroup};
    use raa_core::RaaError;
    use raa_lit::LitRenderer;
    use raa_scene::Scene;
    use raa_web::FrameInput;

    /// The canvas backing size of the current invocation, recorded by the api frame hook.
    #[derive(Copy, Clone, Debug, Default)]
    pub(crate) struct FoliageFrameSize(pub (u32, u32));

    /// Records the frame canvas size and installs the foliage store on first use.
    pub(crate) fn set_foliage_frame_size(st: &mut EngineState, input: &FrameInput) {
        st.features.insert(FoliageFrameSize(input.size));
        st.foliage_state();
    }

    /// The backing-pixel size of `view`: its viewport rectangle of the canvas size.
    fn viewport_size(st: &EngineState, view: usize) -> (u32, u32) {
        let canvas = st
            .features
            .get::<FoliageFrameSize>()
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
        bounds: Aabb,
    }

    fn resolve_resources(
        nodes: &HandleMap<raa_scene::NodeId>,
        instances: &HandleMap<InstanceGroup>,
        lit: &LitRenderer,
        scene: &Scene,
        config: &FoliageGroupConfig,
    ) -> Result<GroupResources, String> {
        let root = *nodes
            .get_ref(config.root)
            .map_err(|_| format!("root {} is not a live scene node", config.root))?;
        let root_world = scene
            .world(root)
            .map_err(|_| format!("root {} has no world transform", config.root))?;
        let mut bounds = Aabb::EMPTY;
        for &handle in &config.full {
            let group = instances
                .get_ref(handle)
                .map_err(|_| format!("source instance {handle} is not live"))?;
            if group.stale {
                return Err(format!("source instance {handle} is stale"));
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
            let p = root_world.transform_point(Vec3::ZERO);
            bounds = Aabb { min: p, max: p };
        }
        Ok(GroupResources { root_world, bounds })
    }

    /// Applies the phase/level visibility. The source and the proxy are never both shown;
    /// an unresolved or failed asset keeps every live source draw.
    fn apply_visibility(
        scene: &mut Scene,
        instances: &mut HandleMap<InstanceGroup>,
        record: &mut FoliageGroupRecord,
        proxy: u32,
    ) {
        let visibility = foliage_visibility(record.state.phase, record.state.level);
        for &handle in &record.config.full {
            if let Ok(group) = instances.get_mut(handle) {
                let _ = group.set_visible(scene, visibility.source);
            }
        }
        if proxy != 0 && !record.config.full.contains(&proxy) {
            if visibility.proxy
                && !record.original_visibility.contains_key(&proxy)
                && let Ok(group) = instances.get_ref(proxy)
            {
                record.original_visibility.insert(proxy, group.visible);
            }
            if let Ok(group) = instances.get_mut(proxy) {
                let _ = group.set_visible(scene, visibility.proxy);
            }
        }
    }

    impl EngineState {
        /// The generation-owned foliage assets and whole-tree groups, installed on first
        /// use. The asynchronous preparation card and [`prepare_foliage_groups`] share it.
        pub(crate) fn foliage_state(&mut self) -> &mut FoliageStateStore {
            if self.features.get::<FoliageStateStore>().is_none() {
                self.features.insert(FoliageStateStore::new());
            }
            self.features
                .get_mut::<FoliageStateStore>()
                .expect("the foliage state was just inserted")
        }
    }

    /// Handoff: 'Install foliage groups before frame preparation' calls
    /// [`prepare_foliage_groups`] for each view after that view's camera is installed and
    /// before `lit.prepare`. Today it reads the [`FoliageStateStore`] asset registry and
    /// string-keyed groups, resolves the live source root and bounds, reuses the
    /// whole-object projected-size selection (`select_billboard`) for the level only, then
    /// sets source/proxy visibility from the asset's pending, failed or ready phase. It
    /// leaves every published asset and its proxy resources in place, so a pending or
    /// failed generation never releases the published resources.
    pub(crate) fn prepare_foliage_groups(st: &mut EngineState, view: u8) -> Result<(), RaaError> {
        if view > 3 {
            return Err(invalid(format!("view {view} must be within 0..=3")));
        }
        let viewport = viewport_size(st, usize::from(view));
        let camera = st.scene.camera;
        st.scene.update_world();
        st.foliage_state();
        let EngineState {
            scene,
            instances,
            nodes,
            lit,
            features,
            ..
        } = st;
        let Some(store) = features.get_mut::<FoliageStateStore>() else {
            return Ok(());
        };
        let groups: Vec<(String, u32)> = store
            .groups
            .iter()
            .map(|(id, record)| (id.clone(), record.config.asset))
            .collect();
        for (id, asset_id) in groups {
            let asset = store.assets.get(asset_id).cloned();
            let proxy = asset.as_ref().map_or(0, |entry| entry.proxy);
            let Some(record) = store.groups.get_mut(&id) else {
                continue;
            };
            match resolve_resources(nodes, instances, lit, scene, &record.config) {
                Ok(resources) => {
                    let recipe = FoliagePrepareRecipe {
                        eye: camera.eye,
                        target: camera.target,
                        up: camera.up,
                        fov_y_rad: camera.fov_y_rad,
                        near: camera.near,
                        ortho_height: camera.ortho_height,
                        viewport,
                        root_world: resources.root_world,
                        bounds: resources.bounds,
                    };
                    match prepare_foliage_record(record, view, &recipe, asset.as_ref()) {
                        Ok(_) => apply_visibility(scene, instances, record, proxy),
                        Err(error) => {
                            let _ = fail_foliage_record(record, view, error.to_string());
                            apply_visibility(scene, instances, record, proxy);
                        }
                    }
                }
                Err(message) => {
                    let _ = fail_foliage_record(record, view, message);
                    apply_visibility(scene, instances, record, proxy);
                }
            }
        }
        Ok(())
    }
}
