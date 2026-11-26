//! m38 asynchronous foliage preparation (M7): the `prepareFoliage`, `removeFoliage`,
//! `setFoliageGroup` and `foliageGroupState` Engine methods and the install/frame hooks
//! of the auto-discovered api module.
//!
//! A preparation starts as a pending generation of the [`FoliageAssets`] registry.
//! [`Engine::prepare_foliage`] validates the descriptor, resolves the live source draws
//! (through `LitHandle::foliage_source`), the source root and the caller-owned compiled
//! LOD1 material in Rust, then awaits [`raa_lit::LitRenderer::prepare_foliage`]. The future
//! owns its submission and no `EngineState` borrow crosses the await. A completion
//! installs the proxy instance and publishes only while its generation is current; a
//! removed or superseded completion releases its prepared mesh and resources again.
//!
//! LIFECYCLE (contracts m7-bake-service and m7-public-api): the source instances are
//! caller-owned and never touched by removal. `FoliageDesc.material` is a caller-owned,
//! generation-exclusive compiled LOD1 lease; it reaches `FoliageBakeSettings.material` and
//! `FoliageAssetInfo.material` unchanged. Removing a generation detaches dependent groups
//! and the material association before the owned mesh/volume are released, and never
//! removes the material or its material-data block.
//!
//! The pure descriptor, mapping, ledger and removal core compiles on the host:
//! `tests/foliage_api.rs` includes this file with `#[path]` and drives it without a GPU.

use std::collections::{BTreeMap, BTreeSet};

use raa_core::{MaterialId, RaaError};
use raa_lit::foliage_bake::{FoliageBakeSettings, IMPOSTOR_VIEWS};
use raa_lit::foliage_resources::FoliageParams;

use crate::foliage_state::{
    FOLIAGE_INVALID, FoliageAssetEntry, FoliageAssets, FoliagePhase, FoliageStateStore,
};

/// Stable code of a preparation that failed while baking or publishing.
pub const FOLIAGE_BAKE_FAILED: &str = "foliage.bake_failed";
/// Reference-layer texture extent of one preparation (a native, non-descriptor value).
pub const FOLIAGE_TEXTURE_RESOLUTION: u32 = 128;
/// Lowest accepted `FoliageDesc.views`.
pub const FOLIAGE_VIEWS_MIN: u32 = 64;
/// Highest accepted `FoliageDesc.views`.
pub const FOLIAGE_VIEWS_MAX: u32 = 128;
/// The two accepted native volume extents.
pub const FOLIAGE_RESOLUTIONS: [u32; 2] = [16, 32];

/// Refuses with [`FOLIAGE_INVALID`].
pub fn invalid(message: impl Into<String>) -> RaaError {
    RaaError::FrameGraph {
        code: FOLIAGE_INVALID,
        message: message.into(),
    }
}

/// Refuses with [`FOLIAGE_BAKE_FAILED`].
pub fn bake_failed(message: impl Into<String>) -> RaaError {
    RaaError::FrameGraph {
        code: FOLIAGE_BAKE_FAILED,
        message: message.into(),
    }
}

/// Keeps a `foliage.invalid` refusal and turns every other failure into
/// [`FOLIAGE_BAKE_FAILED`], with the underlying message preserved.
pub fn translate_error(error: &RaaError) -> RaaError {
    if error.code() == FOLIAGE_INVALID {
        invalid(error.message())
    } else {
        bake_failed(error.message())
    }
}

/// One of the six published foliage modes and its native rank.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FoliageMode {
    Spom,
    Pom,
    Offset,
    Shell,
    Stochastic,
    Differentiable,
}

impl FoliageMode {
    /// The six ranks in order, spom 0 through differentiable 5.
    pub const ALL: [Self; 6] = [
        Self::Spom,
        Self::Pom,
        Self::Offset,
        Self::Shell,
        Self::Stochastic,
        Self::Differentiable,
    ];

    /// Parses the `FoliageDesc.mode` text; anything else refuses [`FOLIAGE_INVALID`].
    pub fn parse(name: &str) -> Result<Self, RaaError> {
        match name {
            "spom" => Ok(Self::Spom),
            "pom" => Ok(Self::Pom),
            "offset" => Ok(Self::Offset),
            "shell" => Ok(Self::Shell),
            "stochastic" => Ok(Self::Stochastic),
            "differentiable" => Ok(Self::Differentiable),
            _ => Err(invalid(format!(
                "'mode' must be spom, pom, offset, shell, stochastic or differentiable, got '{name}'"
            ))),
        }
    }

    /// The native rank written to the asset and group state.
    pub const fn rank(self) -> u32 {
        self as u32
    }

    /// The published mode name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Spom => "spom",
            Self::Pom => "pom",
            Self::Offset => "offset",
            Self::Shell => "shell",
            Self::Stochastic => "stochastic",
            Self::Differentiable => "differentiable",
        }
    }

    /// The mode of native rank `rank`, or None above the differentiable rank.
    pub fn from_rank(rank: u32) -> Option<Self> {
        Self::ALL.into_iter().find(|mode| mode.rank() == rank)
    }

    /// Whether the mode uses the eight packed impostor reference layers.
    pub const fn is_impostor(self) -> bool {
        (self as u32) <= Self::Shell as u32
    }
}

/// The raw, optional `FoliageDesc` fields: a missing field stays `None` so that the
/// host-visible [`FoliageSpec::validate`] reports every malformed subset.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FoliageFields {
    pub instances: Option<Vec<u32>>,
    pub foliage_instances: Option<Vec<u32>>,
    pub root: Option<u32>,
    pub generation: Option<u32>,
    pub mode: Option<String>,
    pub resolution: Option<u32>,
    pub triangles: Option<u32>,
    pub seed: Option<u32>,
    pub views: Option<u32>,
    pub iterations: Option<u32>,
    pub silhouette_weight: Option<f32>,
    pub normal_weight: Option<f32>,
    pub edge_weight: Option<f32>,
    pub translucency: Option<f32>,
    pub cutoff: Option<f32>,
    pub material: Option<u32>,
}

/// The validated `FoliageDesc`; every field is required and checked before any mutation.
#[derive(Clone, Debug, PartialEq)]
pub struct FoliageSpec {
    pub instances: Vec<u32>,
    pub foliage_instances: Vec<u32>,
    pub root: u32,
    pub generation: u32,
    pub mode: FoliageMode,
    pub resolution: u32,
    pub triangles: u32,
    pub seed: u32,
    pub views: u32,
    pub iterations: u32,
    pub silhouette_weight: f32,
    pub normal_weight: f32,
    pub edge_weight: f32,
    pub translucency: f32,
    pub cutoff: f32,
    pub material: u32,
}

fn required<T>(value: Option<T>, name: &str) -> Result<T, RaaError> {
    value.ok_or_else(|| invalid(format!("'{name}' is required")))
}

fn instance_list(list: Vec<u32>, name: &str) -> Result<Vec<u32>, RaaError> {
    if list.is_empty() {
        return Err(invalid(format!(
            "'{name}' must hold at least one instance handle"
        )));
    }
    let mut seen = BTreeSet::new();
    for &handle in &list {
        if !seen.insert(handle) {
            return Err(invalid(format!("'{name}' lists instance {handle} twice")));
        }
    }
    Ok(list)
}

fn weight(value: f32, name: &str) -> Result<f32, RaaError> {
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(invalid(format!(
            "'{name}' must be finite and nonnegative, got {value}"
        )))
    }
}

fn unit(value: f32, name: &str) -> Result<f32, RaaError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err(invalid(format!(
            "'{name}' must be within 0..=1, got {value}"
        )))
    }
}

impl FoliageSpec {
    /// Validates the complete descriptor. A missing, duplicated, empty, out-of-range or
    /// inconsistent field refuses [`FOLIAGE_INVALID`] before anything is allocated.
    pub fn validate(fields: FoliageFields) -> Result<Self, RaaError> {
        let instances = instance_list(required(fields.instances, "instances")?, "instances")?;
        let foliage_instances = instance_list(
            required(fields.foliage_instances, "foliageInstances")?,
            "foliageInstances",
        )?;
        let live: BTreeSet<u32> = instances.iter().copied().collect();
        for &handle in &foliage_instances {
            if !live.contains(&handle) {
                return Err(invalid(format!(
                    "'foliageInstances' names {handle}, which is not in 'instances'"
                )));
            }
        }
        let mode = FoliageMode::parse(&required(fields.mode, "mode")?)?;
        let resolution = required(fields.resolution, "resolution")?;
        if !FOLIAGE_RESOLUTIONS.contains(&resolution) {
            return Err(invalid(format!(
                "'resolution' must be 16 or 32, got {resolution}"
            )));
        }
        let triangles = required(fields.triangles, "triangles")?;
        if triangles == 0 {
            return Err(invalid("'triangles' must be a positive integer"));
        }
        let views = required(fields.views, "views")?;
        if !(FOLIAGE_VIEWS_MIN..=FOLIAGE_VIEWS_MAX).contains(&views) {
            return Err(invalid(format!(
                "'views' must be within {FOLIAGE_VIEWS_MIN}..={FOLIAGE_VIEWS_MAX}, got {views}"
            )));
        }
        let iterations = required(fields.iterations, "iterations")?;
        if iterations == 0 {
            return Err(invalid("'iterations' must be a positive integer"));
        }
        let silhouette_weight = weight(
            required(fields.silhouette_weight, "silhouetteWeight")?,
            "silhouetteWeight",
        )?;
        let normal_weight = weight(
            required(fields.normal_weight, "normalWeight")?,
            "normalWeight",
        )?;
        let edge_weight = weight(required(fields.edge_weight, "edgeWeight")?, "edgeWeight")?;
        if silhouette_weight == 0.0 && normal_weight == 0.0 && edge_weight == 0.0 {
            return Err(invalid("the foliage weights must not all be zero"));
        }
        Ok(Self {
            instances,
            foliage_instances,
            root: required(fields.root, "root")?,
            generation: required(fields.generation, "generation")?,
            mode,
            resolution,
            triangles,
            seed: required(fields.seed, "seed")?,
            views,
            iterations,
            silhouette_weight,
            normal_weight,
            edge_weight,
            translucency: unit(
                required(fields.translucency, "translucency")?,
                "translucency",
            )?,
            cutoff: unit(required(fields.cutoff, "cutoff")?, "cutoff")?,
            material: required(fields.material, "material")?,
        })
    }
}

/// Maps the validated descriptor onto the renderer settings. `material` is the resolved
/// compiled LOD1 handle and reaches [`FoliageBakeSettings::material`] unchanged; it is
/// never replaced by a source triangle material. The impostor routes always bake the
/// eight native reference layers, so their `view_count` is [`IMPOSTOR_VIEWS`] while the
/// distillation routes use `FoliageDesc.views`.
pub fn bake_settings(
    spec: &FoliageSpec,
    material: MaterialId,
    min: [f32; 3],
    max: [f32; 3],
) -> FoliageBakeSettings {
    FoliageBakeSettings {
        params: FoliageParams {
            min,
            max,
            resolution: spec.resolution,
            mode: spec.mode.rank(),
            translucency: spec.translucency,
            generation: spec.generation,
            cutoff: spec.cutoff,
            view_count: if spec.mode.is_impostor() {
                IMPOSTOR_VIEWS
            } else {
                spec.views
            },
            texture_resolution: FOLIAGE_TEXTURE_RESOLUTION,
        },
        material,
        count: spec.triangles,
        seed: spec.seed,
        iterations: spec.iterations,
        silhouette_weight: spec.silhouette_weight,
        normal_weight: spec.normal_weight,
        edge_weight: spec.edge_weight,
    }
}

/// The LOD0 material of one source draw: the group material wins, then the sub-mesh
/// material, then the lit default. This is the per-triangle appearance input of the
/// source snapshot, never the compiled LOD1 `FoliageDesc.material`.
pub fn source_draw_material(group: Option<MaterialId>, submesh: Option<MaterialId>) -> MaterialId {
    group.or(submesh).unwrap_or(MaterialId(0))
}

/// The JS-facing result of one published preparation (`FoliageAssetInfo`). `mesh` is the
/// native proxy mesh handle and `material` echoes the caller-owned lease unchanged.
#[derive(Clone, Debug, PartialEq)]
pub struct FoliageAssetInfo {
    pub id: u32,
    pub generation: u32,
    pub mode: FoliageMode,
    pub triangles: u32,
    pub volume_resolution: u32,
    pub mesh: u32,
    pub material: u32,
}

impl FoliageAssetInfo {
    /// Reports the published entry: the actual prepared generation, mode, output
    /// triangle count, volume extent, proxy mesh and the same material handle.
    pub fn from_entry(entry: &FoliageAssetEntry) -> Self {
        Self {
            id: entry.id,
            generation: entry.generation,
            mode: FoliageMode::from_rank(entry.mode).unwrap_or(FoliageMode::Spom),
            triangles: entry.triangle_count,
            volume_resolution: entry.volume_resolution,
            mesh: entry.mesh,
            material: entry.material,
        }
    }
}

/// The `FoliageState.phase` text.
pub fn phase_name(phase: FoliagePhase) -> &'static str {
    match phase {
        FoliagePhase::Pending => "pending",
        FoliagePhase::Ready => "ready",
        FoliagePhase::Failed => "failed",
    }
}

/// The `FoliageState.mode` text of native rank `rank`.
pub fn mode_name(rank: u32) -> &'static str {
    FoliageMode::from_rank(rank).map_or(FoliageMode::Spom.name(), FoliageMode::name)
}

/// What a completed native preparation may do against the live store.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CompletionTarget {
    /// The engine was disposed while the preparation was in flight.
    Disposed,
    /// The pending generation is still the published one.
    Current,
    /// The entry was removed or a newer generation replaced it.
    Stale,
}

/// Classifies a completion: only a pending entry of the same generation is current; a
/// missing engine state is disposed and every other case is stale.
pub fn completion_target(
    live: Option<&FoliageAssets>,
    id: u32,
    generation: u32,
) -> CompletionTarget {
    let Some(assets) = live else {
        return CompletionTarget::Disposed;
    };
    match assets.get(id) {
        Some(entry) if entry.generation == generation && entry.phase == FoliagePhase::Pending => {
            CompletionTarget::Current
        }
        _ => CompletionTarget::Stale,
    }
}

/// The live native assets and the generation-exclusive material claims of the API.
/// `assets` keeps every prepared [`FoliageAsset`](raa_lit::FoliageAsset) until
/// `removeFoliage` releases it; `materials` maps a material handle to its asset id so a
/// second live generation of the same lease refuses before mutation.
#[derive(Debug)]
pub struct FoliageLedger<A> {
    assets: BTreeMap<u32, A>,
    materials: BTreeMap<u32, u32>,
}

impl<A> Default for FoliageLedger<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A> FoliageLedger<A> {
    pub fn new() -> Self {
        Self {
            assets: BTreeMap::new(),
            materials: BTreeMap::new(),
        }
    }

    /// The asset id holding `material`, if any live generation claims it.
    pub fn holder(&self, material: u32) -> Option<u32> {
        self.materials.get(&material).copied()
    }

    /// Claims `material` for `id`; another holder refuses [`FOLIAGE_INVALID`].
    pub fn claim(&mut self, material: u32, id: u32) -> Result<(), RaaError> {
        match self.materials.get(&material) {
            Some(&holder) if holder != id => Err(invalid(format!(
                "material {material} is already used by foliage asset {holder}"
            ))),
            _ => {
                self.materials.insert(material, id);
                Ok(())
            }
        }
    }

    /// Releases the claim only while `id` still holds it.
    pub fn release_material(&mut self, material: u32, id: u32) -> bool {
        if self.materials.get(&material).copied() == Some(id) {
            self.materials.remove(&material);
            true
        } else {
            false
        }
    }

    /// Keeps a published native asset for its later release.
    pub fn insert(&mut self, id: u32, asset: A) -> Option<A> {
        self.assets.insert(id, asset)
    }

    /// Takes the native asset of `id` for release; unknown ids give None.
    pub fn take(&mut self, id: u32) -> Option<A> {
        self.assets.remove(&id)
    }
}

/// One step of a foliage removal, in the contract's order: the dependent groups detach
/// first, then the proxy instance and the material association, then the owned mesh and
/// volume. The caller-owned material and its data block are never a step.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FoliageRemovalStep {
    DetachGroups,
    RemoveProxy(u32),
    DetachAssociation(u32),
    ReleaseMesh(u32),
}

/// The removed entry plus the visibility pairs the caller restores before the proxy goes.
#[derive(Clone, Debug, PartialEq)]
pub struct FoliageRemoval {
    pub entry: FoliageAssetEntry,
    pub restored: Vec<(u32, bool)>,
}

impl FoliageRemoval {
    /// The four removal steps of this entry, in order.
    pub fn steps(&self) -> [FoliageRemovalStep; 4] {
        [
            FoliageRemovalStep::DetachGroups,
            FoliageRemovalStep::RemoveProxy(self.entry.proxy),
            FoliageRemovalStep::DetachAssociation(self.entry.material),
            FoliageRemovalStep::ReleaseMesh(self.entry.mesh),
        ]
    }
}

/// Removes `asset` from the store: dependent groups detach and their saved visibility is
/// returned. An unknown id refuses [`FOLIAGE_INVALID`] before any mutation.
pub fn plan_removal(store: &mut FoliageStateStore, asset: u32) -> Result<FoliageRemoval, RaaError> {
    if !store.assets.contains(asset) {
        return Err(invalid(format!(
            "foliage asset {asset} is unknown or was removed"
        )));
    }
    let (entry, restored) = store.remove_asset(asset);
    match entry {
        Some(entry) => Ok(FoliageRemoval { entry, restored }),
        None => Err(invalid(format!(
            "foliage asset {asset} is unknown or was removed"
        ))),
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) use web::{frame, install};

#[cfg(target_arch = "wasm32")]
mod web {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::*;
    use crate::convert::{DISPOSED, api_error};
    use crate::engine::{Engine, InstallCtx};
    use crate::foliage_state::{
        FoliageGroupConfig, FoliageLevel, FoliageState, set_foliage_frame_size,
    };
    use crate::state::{EngineState, group_draw_materials};
    use js_sys::{Array, Object, Reflect};
    use raa_core::{GpuContext, MeshId};
    use raa_lit::FoliageAsset;
    use raa_lit::foliage_resources::FoliageSource;
    use raa_scene::NodeId;
    use raa_web::FrameInput;
    use wasm_bindgen::prelude::*;

    type SharedFoliageLedger = Rc<RefCell<FoliageLedger<FoliageAsset>>>;

    /// Installs the generation-owned asset/group store and the native asset ledger. The
    /// frame hook records the canvas size that the per-view group preparation reads.
    pub(crate) fn install(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
        ctx.features.insert(FoliageStateStore::new());
        ctx.features
            .insert(Rc::new(RefCell::new(FoliageLedger::<FoliageAsset>::new())));
        Ok(())
    }

    /// Records the frame's canvas size; the group preparation itself runs at the engine's
    /// per-view lifecycle hook.
    pub(crate) fn frame(
        st: &mut EngineState,
        _gpu: &GpuContext,
        input: &FrameInput,
    ) -> Result<(), RaaError> {
        set_foliage_frame_size(st, input);
        Ok(())
    }

    fn error_to_js(error: &RaaError) -> JsValue {
        raa_web::error_to_js(error)
    }

    fn disposed() -> JsValue {
        error_to_js(&api_error(DISPOSED, "the engine was disposed"))
    }

    fn field(obj: &JsValue, key: &str) -> Result<Option<JsValue>, RaaError> {
        if !obj.is_object() {
            return Err(invalid("a foliage description must be an object"));
        }
        let value = Reflect::get(obj, &JsValue::from_str(key))
            .map_err(|_| invalid(format!("cannot read '{key}'")))?;
        Ok(if value.is_undefined() || value.is_null() {
            None
        } else {
            Some(value)
        })
    }

    fn number(obj: &JsValue, key: &str) -> Result<Option<f64>, RaaError> {
        let Some(value) = field(obj, key)? else {
            return Ok(None);
        };
        value
            .as_f64()
            .map(Some)
            .ok_or_else(|| invalid(format!("'{key}' must be a number")))
    }

    fn whole_u32(obj: &JsValue, key: &str) -> Result<Option<u32>, RaaError> {
        let Some(value) = number(obj, key)? else {
            return Ok(None);
        };
        if value.fract() != 0.0 || !(0.0..=f64::from(u32::MAX)).contains(&value) {
            return Err(invalid(format!(
                "'{key}' must be a whole handle number, got {value}"
            )));
        }
        Ok(Some(value as u32))
    }

    fn f32_field(obj: &JsValue, key: &str) -> Result<Option<f32>, RaaError> {
        Ok(number(obj, key)?.map(|value| value as f32))
    }

    fn string_field(obj: &JsValue, key: &str) -> Result<Option<String>, RaaError> {
        let Some(value) = field(obj, key)? else {
            return Ok(None);
        };
        value
            .as_string()
            .map(Some)
            .ok_or_else(|| invalid(format!("'{key}' must be a string")))
    }

    fn handles(obj: &JsValue, key: &str) -> Result<Option<Vec<u32>>, RaaError> {
        let Some(value) = field(obj, key)? else {
            return Ok(None);
        };
        if !Array::is_array(&value) {
            return Err(invalid(format!(
                "'{key}' must be an array of instance handles"
            )));
        }
        let array: &Array = value.unchecked_ref();
        let mut out = Vec::with_capacity(array.length() as usize);
        for index in 0..array.length() {
            let item = array
                .get(index)
                .as_f64()
                .ok_or_else(|| invalid(format!("'{key}[{index}]' must be a number")))?;
            if item.fract() != 0.0 || !(0.0..=f64::from(u32::MAX)).contains(&item) {
                return Err(invalid(format!(
                    "'{key}[{index}]' must be a whole handle number, got {item}"
                )));
            }
            out.push(item as u32);
        }
        Ok(Some(out))
    }

    fn parse_desc(desc: &JsValue) -> Result<FoliageSpec, RaaError> {
        FoliageSpec::validate(FoliageFields {
            instances: handles(desc, "instances")?,
            foliage_instances: handles(desc, "foliageInstances")?,
            root: whole_u32(desc, "root")?,
            generation: whole_u32(desc, "generation")?,
            mode: string_field(desc, "mode")?,
            resolution: whole_u32(desc, "resolution")?,
            triangles: whole_u32(desc, "triangles")?,
            seed: whole_u32(desc, "seed")?,
            views: whole_u32(desc, "views")?,
            iterations: whole_u32(desc, "iterations")?,
            silhouette_weight: f32_field(desc, "silhouetteWeight")?,
            normal_weight: f32_field(desc, "normalWeight")?,
            edge_weight: f32_field(desc, "edgeWeight")?,
            translucency: f32_field(desc, "translucency")?,
            cutoff: f32_field(desc, "cutoff")?,
            material: whole_u32(desc, "material")?,
        })
    }

    fn parse_level(value: Option<JsValue>) -> Result<FoliageLevel, RaaError> {
        let Some(value) = value else {
            return Err(invalid("'level' is required"));
        };
        if let Some(name) = value.as_string() {
            if name == "auto" {
                return Ok(FoliageLevel::Auto);
            }
        } else if let Some(number) = value.as_f64() {
            if number == 0.0 {
                return Ok(FoliageLevel::Full);
            }
            if number == 1.0 {
                return Ok(FoliageLevel::Proxy);
            }
        }
        Err(invalid("'level' must be 'auto', 0 or 1"))
    }

    fn parse_group(desc: &JsValue) -> Result<FoliageGroupConfig, RaaError> {
        let level = parse_level(field(desc, "level")?)?;
        Ok(FoliageGroupConfig {
            asset: required(whole_u32(desc, "asset")?, "asset")?,
            full: required(handles(desc, "full")?, "full")?,
            root: required(whole_u32(desc, "root")?, "root")?,
            switch_pixels: required(f32_field(desc, "switchPixels")?, "switchPixels")?,
            level,
        })
    }

    fn resolve_material(st: &EngineState, spec: &FoliageSpec) -> Result<MaterialId, RaaError> {
        let material = st.material(spec.material).map_err(|_| {
            invalid(format!(
                "foliage material {} is not a live material handle",
                spec.material
            ))
        })?;
        let base = st.lit.materials().get(material).map(|record| record.data);
        let words = base.and_then(|base| st.lit.material_data(base));
        raa_lit::foliage_bake::validate_material_data(words.as_deref())
            .map_err(|error| invalid(error.message()))?;
        Ok(material)
    }

    struct SourceSnapshot {
        root: NodeId,
        source: FoliageSource,
    }

    fn resolve_source(
        st: &mut EngineState,
        spec: &FoliageSpec,
    ) -> Result<SourceSnapshot, RaaError> {
        st.scene.update_world();
        let root = *st
            .nodes
            .get_ref(spec.root)
            .map_err(|_| invalid(format!("root {} is not a live scene node", spec.root)))?;
        let root_world = st
            .scene
            .world(root)
            .map_err(|_| invalid(format!("root {} has no world transform", spec.root)))?;
        let root_inverse = root_world
            .inverse()
            .ok_or_else(|| invalid(format!("root {} has no invertible world frame", spec.root)))?;
        let foliage: BTreeSet<u32> = spec.foliage_instances.iter().copied().collect();
        let single = st.single_material;
        let mut draws = Vec::new();
        for &handle in &spec.instances {
            let group = st
                .instances
                .get_ref(handle)
                .map_err(|_| invalid(format!("source instance {handle} is not live")))?;
            if group.stale {
                return Err(invalid(format!("source instance {handle} is stale")));
            }
            let world = st
                .scene
                .world(group.node)
                .map_err(|_| invalid(format!("source instance {handle} has no world transform")))?;
            let model_to_root = root_inverse * world;
            let info = st
                .lit
                .info(group.mesh)
                .ok_or_else(|| invalid(format!("source instance {handle} has no mesh")))?;
            let ids = st
                .lit
                .submesh_ids(group.mesh)
                .ok_or_else(|| invalid(format!("source instance {handle} has no sub-meshes")))?;
            let parts: Vec<(MeshId, Option<MaterialId>, u32)> = ids
                .iter()
                .zip(&info.submeshes)
                .map(|(&id, sub)| (id, sub.material, sub.index_count))
                .collect();
            for (mesh, submesh) in group_draw_materials(group, &parts, single) {
                draws.push((
                    mesh,
                    source_draw_material(group.material, submesh),
                    group.tag,
                    model_to_root,
                    foliage.contains(&handle),
                ));
            }
        }
        let source = st
            .lit
            .handle()
            .foliage_source(&draws, root_world, spec.generation)?;
        Ok(SourceSnapshot { root, source })
    }

    fn attach_proxy(
        st: &mut EngineState,
        root: NodeId,
        asset: &FoliageAsset,
    ) -> Result<u32, RaaError> {
        let proxy = st.add_instance_group(root, asset.mesh, Some(asset.material))?;
        if let Ok(group) = st.instances.get_mut(proxy) {
            let _ = group.set_visible(&mut st.scene, false);
        }
        Ok(proxy)
    }

    fn ledger_of(st: &EngineState) -> Result<SharedFoliageLedger, RaaError> {
        st.features
            .get::<SharedFoliageLedger>()
            .cloned()
            .ok_or_else(|| invalid("the foliage feature is not installed"))
    }

    fn fail_preparation(
        st: &mut EngineState,
        ledger: &SharedFoliageLedger,
        spec: &FoliageSpec,
        id: u32,
        error: &RaaError,
    ) {
        let _ = st
            .foliage_state()
            .assets
            .fail(id, spec.generation, error.to_string());
        ledger.borrow_mut().release_material(spec.material, id);
    }

    fn live_roots(st: &EngineState) -> BTreeSet<u32> {
        st.nodes.iter().map(|(handle, _)| handle).collect()
    }

    fn live_instances(st: &EngineState) -> BTreeMap<u32, bool> {
        let mut instances: BTreeMap<u32, bool> = st
            .instances
            .iter()
            .filter(|(_, group)| !group.stale)
            .map(|(handle, group)| (handle, group.visible))
            .collect();
        if let Some(store) = st.features.get::<FoliageStateStore>() {
            for (_, record) in store.groups.iter() {
                for (&handle, &visible) in &record.original_visibility {
                    if let Some(slot) = instances.get_mut(&handle) {
                        *slot = visible;
                    }
                }
            }
        }
        instances
    }

    fn restore(st: &mut EngineState, prior: Vec<(u32, bool)>) {
        for (handle, visible) in prior {
            if let Ok(group) = st.instances.get_mut(handle) {
                let _ = group.set_visible(&mut st.scene, visible);
            }
        }
    }

    fn set_field(obj: &Object, key: &str, value: JsValue) {
        let _ = Reflect::set(obj, &JsValue::from_str(key), &value);
    }

    fn state_value(state: &FoliageState) -> JsValue {
        let obj = Object::new();
        set_field(&obj, "phase", JsValue::from_str(phase_name(state.phase)));
        set_field(&obj, "generation", JsValue::from(state.generation));
        set_field(
            &obj,
            "asset",
            state.asset.map_or(JsValue::NULL, JsValue::from),
        );
        set_field(&obj, "mode", JsValue::from_str(mode_name(state.mode)));
        set_field(&obj, "level", JsValue::from(state.level));
        set_field(
            &obj,
            "error",
            state
                .error
                .as_ref()
                .map_or(JsValue::NULL, |error| JsValue::from_str(error)),
        );
        obj.into()
    }

    fn asset_info_value(info: &FoliageAssetInfo) -> JsValue {
        let obj = Object::new();
        set_field(&obj, "id", JsValue::from(info.id));
        set_field(&obj, "generation", JsValue::from(info.generation));
        set_field(&obj, "mode", JsValue::from_str(info.mode.name()));
        set_field(&obj, "triangles", JsValue::from(info.triangles));
        set_field(
            &obj,
            "volumeResolution",
            JsValue::from(info.volume_resolution),
        );
        set_field(&obj, "mesh", JsValue::from(info.mesh));
        set_field(&obj, "material", JsValue::from(info.material));
        obj.into()
    }

    #[wasm_bindgen(js_class = Engine)]
    impl Engine {
        /// Prepares one whole-tree foliage generation for `desc` and resolves with its
        /// `FoliageAssetInfo` once the native asset is published.
        ///
        /// The descriptor is validated before any mutation; the live source instances,
        /// source root and compiled LOD1 material are resolved in Rust, and the source
        /// snapshot is taken through `LitHandle::foliage_source` with the LOD0 draw
        /// materials. Only the awaited `LitRenderer::prepare_foliage` runs off the engine
        /// state; the future owns its submission. LIFECYCLE (m7-bake-service): a completed
        /// generation installs its proxy instance and publishes only while it is current;
        /// a removed, superseded or disposed completion releases its prepared mesh and
        /// resources and never touches the caller-owned material or its data.
        ///
        /// Handoff: 'Coordinate foliage presentations in the game engine' consumes
        /// `Engine::prepare_foliage` (this comment is anchored on its definition),
        /// `Engine::remove_foliage`, `Engine::set_foliage_group` and
        /// `Engine::foliage_group_state`. Today they publish the validated native
        /// generation through `LitRenderer::prepare_foliage`
        /// (`crates/raa-lit/src/renderer.rs:867 prepare_foliage`), carry
        /// `FoliageDesc.material` unchanged into `FoliageAssetInfo.material` and retire an
        /// asset in the order of `FoliageRemoval::steps` without removing that material or
        /// its data block. Groups read the state published by `FoliageStateStore.state`
        /// (`crates/raa-js/src/foliage_state.rs:48 FoliageState`). Earlier handoffs this
        /// rests on: `LitHandle::foliage_source`
        /// (`crates/raa-lit/src/handle.rs:1757 foliage_source`) and `FoliageState`
        /// (`crates/raa-js/src/foliage_state.rs:48 FoliageState`).
        #[wasm_bindgen(js_name = prepareFoliage)]
        pub fn prepare_foliage(&self, desc: JsValue) -> js_sys::Promise {
            let submission = (|| {
                let spec = parse_desc(&desc).map_err(|error| error_to_js(&error))?;
                let (gpu, rc) = self.bake_context()?;
                let mut guard = rc.borrow_mut();
                let st = guard.as_mut().ok_or_else(disposed)?;
                let material = resolve_material(st, &spec).map_err(|error| error_to_js(&error))?;
                let ledger = ledger_of(st).map_err(|error| error_to_js(&error))?;
                if let Some(holder) = ledger.borrow().holder(spec.material) {
                    return Err(error_to_js(&invalid(format!(
                        "material {} is already used by foliage asset {holder}",
                        spec.material
                    ))));
                }
                let snapshot = resolve_source(st, &spec).map_err(|error| error_to_js(&error))?;
                let SourceSnapshot { root, source } = snapshot;
                if source.triangles.is_empty() {
                    return Err(error_to_js(&invalid("foliage source is empty")));
                }
                let settings = bake_settings(&spec, material, source.min, source.max);
                settings.validate().map_err(|error| error_to_js(&error))?;
                let id = st.foliage_state().assets.begin(
                    spec.generation,
                    spec.mode.rank(),
                    spec.resolution,
                    spec.material,
                );
                if let Err(error) = ledger.borrow_mut().claim(spec.material, id) {
                    let _ = st.foliage_state().assets.remove(id);
                    return Err(error_to_js(&error));
                }
                let triangles = source.triangles.len();
                let completion = st.lit.prepare_foliage(&gpu, source, settings);
                drop(guard);
                Ok::<_, JsValue>((gpu, rc, ledger, id, spec, root, completion, triangles))
            })();
            wasm_bindgen_futures::future_to_promise(async move {
                let (gpu, rc, ledger, id, spec, root, completion, triangles) = submission?;
                let result = completion.await;
                // The stage lines of `raa_lit::foliage_bake::report_foliage_stage` end here;
                // `crate::alloc_log::report_stage` writes them to `console.error`.
                raa_lit::foliage_bake::report_foliage_stage(
                    if result.is_ok() {
                        "preparation finished"
                    } else {
                        "preparation failed"
                    },
                    0,
                    triangles,
                );
                let mut guard = rc.borrow_mut();
                let Some(st) = guard.as_mut() else {
                    return Err(disposed());
                };
                let target =
                    completion_target(Some(&st.foliage_state().assets), id, spec.generation);
                match result {
                    Ok(asset) if target == CompletionTarget::Current => {
                        if let Err(error) = gpu.ensure_live() {
                            fail_preparation(st, &ledger, &spec, id, &error);
                            st.lit.release_foliage(asset);
                            return Err(error_to_js(&translate_error(&error)));
                        }
                        let proxy = match attach_proxy(st, root, &asset) {
                            Ok(proxy) => proxy,
                            Err(error) => {
                                fail_preparation(st, &ledger, &spec, id, &error);
                                st.lit.release_foliage(asset);
                                return Err(error_to_js(&translate_error(&error)));
                            }
                        };
                        let published = st.foliage_state().assets.complete(
                            id,
                            spec.generation,
                            asset.mesh.0,
                            proxy,
                            asset.triangle_count,
                        );
                        if !published {
                            let _ = st.remove_instance_group(proxy);
                            st.lit.release_foliage(asset);
                            ledger.borrow_mut().release_material(spec.material, id);
                            return Err(error_to_js(&invalid(format!(
                                "foliage asset {id} was superseded before it completed"
                            ))));
                        }
                        ledger.borrow_mut().insert(id, asset);
                        let info = st
                            .foliage_state()
                            .assets
                            .get(id)
                            .map(FoliageAssetInfo::from_entry)
                            .ok_or_else(|| {
                                error_to_js(&invalid(format!(
                                    "foliage asset {id} is not published"
                                )))
                            })?;
                        Ok(asset_info_value(&info))
                    }
                    Ok(asset) => {
                        st.lit.release_foliage(asset);
                        ledger.borrow_mut().release_material(spec.material, id);
                        Err(error_to_js(&invalid(format!(
                            "foliage asset {id} was removed before it completed"
                        ))))
                    }
                    Err(error) => {
                        let translated = translate_error(&error);
                        if target == CompletionTarget::Current {
                            fail_preparation(st, &ledger, &spec, id, &error);
                        } else {
                            ledger.borrow_mut().release_material(spec.material, id);
                        }
                        Err(error_to_js(&translated))
                    }
                }
            })
        }

        /// Releases foliage asset `id`: dependent groups detach and their caller
        /// visibility returns, the proxy instance and the material association go, then
        /// the owned mesh and volume are freed. The caller-owned material and its data
        /// block stay; an unknown or already removed id refuses `foliage.invalid`.
        #[wasm_bindgen(js_name = removeFoliage)]
        pub fn remove_foliage(&mut self, id: u32) -> Result<(), JsValue> {
            self.with_state(|_, st| {
                let ledger = ledger_of(st)?;
                let removal = plan_removal(st.foliage_state(), id)?;
                let mut native = ledger.borrow_mut().take(id);
                for step in removal.steps() {
                    match step {
                        FoliageRemovalStep::DetachGroups => {
                            for &(handle, visible) in &removal.restored {
                                if let Ok(group) = st.instances.get_mut(handle) {
                                    let _ = group.set_visible(&mut st.scene, visible);
                                }
                            }
                        }
                        FoliageRemovalStep::RemoveProxy(proxy) if proxy != 0 => {
                            let _ = st.remove_instance_group(proxy);
                        }
                        FoliageRemovalStep::RemoveProxy(_) => {}
                        FoliageRemovalStep::DetachAssociation(_material) => {
                            if let Some(asset) = &native {
                                let _ = st.lit.handle().set_foliage(asset.material, None);
                            }
                        }
                        FoliageRemovalStep::ReleaseMesh(_mesh) => {
                            if let Some(asset) = native.take() {
                                st.lit.release_foliage(asset);
                            }
                        }
                    }
                }
                if removal.entry.phase != FoliagePhase::Pending {
                    ledger
                        .borrow_mut()
                        .release_material(removal.entry.material, id);
                }
                Ok(())
            })
        }

        /// Registers or replaces the whole-tree foliage group `id`, or removes it with a
        /// null description. The asset must be published; every other refusal
        /// (`foliage.invalid`) changes nothing.
        #[wasm_bindgen(js_name = setFoliageGroup)]
        pub fn set_foliage_group(&mut self, id: &str, desc: JsValue) -> Result<(), JsValue> {
            self.with_state(|_, st| {
                let config = if desc.is_null() || desc.is_undefined() {
                    None
                } else {
                    Some(parse_group(&desc)?)
                };
                if let Some(config) = &config
                    && !st.foliage_state().assets.contains(config.asset)
                {
                    return Err(invalid(format!(
                        "foliage asset {} is not published",
                        config.asset
                    )));
                }
                let roots = live_roots(st);
                let instances = live_instances(st);
                let prior = st
                    .foliage_state()
                    .groups
                    .set(id, config, &roots, &instances)?;
                restore(st, prior);
                Ok(())
            })
        }

        /// The last prepared state of foliage group `id`, or null when it is not registered.
        #[wasm_bindgen(js_name = foliageGroupState)]
        pub fn foliage_group_state(&self, id: &str) -> Result<JsValue, JsValue> {
            self.with_state(|_, st| {
                Ok(st
                    .foliage_state()
                    .state(id)
                    .map_or(JsValue::NULL, state_value))
            })
        }
    }
}
