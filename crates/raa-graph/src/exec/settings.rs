//! Settings accessors of the executor (contract `rg-settings-accessors`). For each feature the
//! graph configures this file holds its Setting keys (the members of its setter's input interface
//! in `packages/raa/types/*.d.ts`), the descriptor those keys read and write, and the handle call
//! that applies it. Pipeline, Taa, Other and the eight post features answer empty values here.

use raa_contact::{ContactDesc, ContactHandle};
use raa_core::RaaError;
use raa_gi::{
    DebugView as GiDebugView, GiHandle, GiQuality, GiQualityPatch, GiSettings, GiSettingsPatch,
    IndirectMode, SsgiQuality,
};
use raa_gtao::{AoDesc, GtaoHandle};
use raa_lit::{AreaShadowSettings, AreaShadowTier, StochasticMode, StochasticShadowSettings};
use raa_shadow::{ShadowDesc, ShadowHandle};
use raa_tonemap::{Look, ToneMapper, ToneMapping, TonemapControl};
use raa_vg::{DebugView as VgDebugView, VgHandle, VgSettings, VgSettingsPatch};

use super::FeatureHandles;
use crate::catalog::{FeatureId, ParamValue};

/// The `FrameGpu` feature id that selects the diffuse form factor for area lights: the
/// `formFactor` mode. `crates/raa-js/src/api/m28_lights.rs:230` declares it as
/// `AREA_FORM_FACTOR_FEATURE_ID` and `crates/raa-js/tests/local_lights.rs:254` pins it.
const AREA_FORM_FACTOR_FEATURE_ID: u32 = 39;

/// The JS names of the tone mapping curve, look and area-light shading mode (`true` is `formFactor`).
const MAPPERS: [(&str, ToneMapper); 3] = [
    ("agx", ToneMapper::Agx),
    ("aces", ToneMapper::Aces),
    ("linear", ToneMapper::Linear),
];
const LOOKS: [(&str, Look); 3] = [
    ("none", Look::None),
    ("punchy", Look::Punchy),
    ("golden", Look::Golden),
];
const SHADING: [(&str, bool); 2] = [("ltc", false), ("formFactor", true)];

/// The Setting keys of `feature` in catalog key order: the members of its setter's input
/// interface. Empty for a feature without Setting keys.
pub fn setting_keys(feature: &FeatureId) -> &'static [&'static str] {
    match feature {
        FeatureId::Tonemap => &["mapper", "exposureEv", "look", "dither"],
        FeatureId::Shadow => &["softness", "distance", "resolution"],
        FeatureId::Vg => &[
            "errorPixels",
            "softwareRaster",
            "occlusion",
            "shadows",
            "maxVisibleClusters",
            "debugView",
        ],
        FeatureId::GiQuality => &["resolution", "levels", "probeRays"],
        FeatureId::Gi => &[
            "intensity",
            "environmentDiffuse",
            "localLightShadows",
            "voxelFullRelight",
            "voxelHistoryMax",
            "voxelRays",
            "indirect",
            "screenTraces",
            "reflections",
            "historyMax",
            "historyMin",
            "clampGamma",
            "spatialPasses",
            "restirMCap",
            "debugView",
            "screenSpaceQuality",
        ],
        FeatureId::AreaShadows => &[
            "tier",
            "heroCount",
            "heroDistance",
            "fadeDistance",
            "band",
            "stochasticCount",
            "stochasticDistance",
            "stochasticCells",
            "stochasticMode",
        ],
        FeatureId::AreaShading => &["mode"],
        FeatureId::Gtao => &["radius", "intensity", "power"],
        FeatureId::Contact => &["distance", "thickness", "intensity"],
        _ => &[],
    }
}

/// [`setting_keys`] followed by the presence keys, which are not catalog keys: `enabled` for Vg;
/// `enabled` and `screenSpace` for Gi.
pub fn snapshot_keys(feature: &FeatureId) -> &'static [&'static str] {
    match feature {
        FeatureId::Vg => &[
            "errorPixels",
            "softwareRaster",
            "occlusion",
            "shadows",
            "maxVisibleClusters",
            "debugView",
            "enabled",
        ],
        FeatureId::Gi => &[
            "intensity",
            "environmentDiffuse",
            "localLightShadows",
            "voxelFullRelight",
            "voxelHistoryMax",
            "voxelRays",
            "indirect",
            "screenTraces",
            "reflections",
            "historyMax",
            "historyMin",
            "clampGamma",
            "spatialPasses",
            "restirMCap",
            "debugView",
            "screenSpaceQuality",
            "enabled",
            "screenSpace",
        ],
        other => setting_keys(other),
    }
}

/// The default value of every snapshot key of `feature`: the feature's default descriptor read
/// through the getters [`read_values`] uses. Empty for a feature without keys.
///
/// Handoff: `default_values` (crates/raa-graph/src/exec/settings.rs:133) and the other dispatch functions of this file (`setting_keys`, `snapshot_keys`, `merge_values`, `read_values`, `write_values`) and `AppliedDescs` are extended by card 'raa-graph exec 3/5: settings accessors for ssr to cel', which adds the post-feature match arms and the post fields of `AppliedDescs`. Today the eight post features (Ssr to Cel) answer empty values in these functions; the state passed is the per-feature key tables (FeatureId to keys in catalog order, then presence keys).
///
/// Handoff: `default_values` (crates/raa-graph/src/exec/settings.rs:133) and the same dispatch functions are called by card 'raa-graph exec 4/5: GraphExecutor apply, rollback and clear' to take the snapshot, merge targets, write values and capture applied descriptors. Today `read_values` and `write_values` read and write through the live handles, and `AppliedDescs::capture` reads every installed feature; the state passed is the key tables and `AppliedDescs`.
pub fn default_values(feature: &FeatureId) -> Vec<(String, ParamValue)> {
    default_desc(feature).map_or_else(Vec::new, |desc| values_of(&desc, snapshot_keys(feature)))
}

/// Merges `base` then `changes` over the default descriptor of `feature` and validates the result
/// with the validation the feature's handle runs. Returns every snapshot key in [`snapshot_keys`]
/// order. `Err((key, message))` names the key whose value the descriptor cannot hold (a value of
/// the wrong kind, an unknown key, a value outside the field's type) or, when the validation
/// refuses, the first write after which the descriptor is refused.
pub fn merge_values(
    feature: &FeatureId,
    base: &[(String, ParamValue)],
    changes: &[(String, ParamValue)],
) -> Result<Vec<(String, ParamValue)>, (String, String)> {
    let Some(default) = default_desc(feature) else {
        return Ok(Vec::new());
    };
    let mut desc = default;
    for (key, value) in base.iter().chain(changes) {
        desc_set(&mut desc, key, value).map_err(|message| (key.clone(), message))?;
    }
    match desc_check(&desc) {
        Ok(()) => Ok(values_of(&desc, snapshot_keys(feature))),
        Err(error) => Err(refusal(default, base, changes, &error)),
    }
}

/// The key and message of the first prefix of the writes that the validation refuses. A rule that
/// relates two fields can refuse a prefix that a later write would repair, so the prefix is named,
/// not the whole set; `error` is the refusal of the complete set.
fn refusal(
    default: Desc,
    base: &[(String, ParamValue)],
    changes: &[(String, ParamValue)],
    error: &RaaError,
) -> (String, String) {
    let mut desc = default;
    let mut last = String::new();
    for (key, value) in base.iter().chain(changes) {
        // The same writes already succeeded in `merge_values`, so this cannot fail.
        let _ = desc_set(&mut desc, key, value);
        if let Err(prefix) = desc_check(&desc) {
            return (key.clone(), prefix.message().to_owned());
        }
        last.clone_from(key);
    }
    (last, error.message().to_owned())
}

/// The live values of `feature`'s snapshot keys, in [`snapshot_keys`] order. An empty vector for a
/// feature without keys; `None` while the feature's handle (or `lit` for the area-light features)
/// is absent.
pub fn read_values(
    h: &FeatureHandles<'_>,
    feature: &FeatureId,
) -> Option<Vec<(String, ParamValue)>> {
    let keys = snapshot_keys(feature);
    if keys.is_empty() {
        return Some(Vec::new());
    }
    let desc = desc_of(&AppliedDescs::capture(h), feature)?;
    Some(values_of(&desc, keys))
}

/// Writes `values` over the live descriptor of `feature` and applies the result through the
/// feature's handle, which validates it. A refusal changes nothing. An absent handle, or a
/// feature without keys, is `Ok(())` with nothing written.
pub fn write_values(
    h: &mut FeatureHandles<'_>,
    feature: &FeatureId,
    values: &[(String, ParamValue)],
) -> Result<(), RaaError> {
    let Some(mut desc) = desc_of(&AppliedDescs::capture(h), feature) else {
        return Ok(());
    };
    for (key, value) in values {
        desc_set(&mut desc, key, value).map_err(|message| apply_refusal(key, &message))?;
    }
    desc_apply(h, &desc)
}

/// The values of `keys` read from `desc`, in `keys` order.
fn values_of(desc: &Desc, keys: &[&str]) -> Vec<(String, ParamValue)> {
    keys.iter()
        .filter_map(|key| Some(((*key).to_owned(), desc_get(desc, key)?)))
        .collect()
}

/// The refusal of a value the descriptor cannot hold: the S52 `graph.apply` code, naming the key.
fn apply_refusal(key: &str, message: &str) -> RaaError {
    RaaError::FrameGraph {
        code: "graph.apply",
        message: format!("{key}: {message}"),
    }
}

/// One feature's descriptor as the accessors hold it.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Desc {
    Tonemap(ToneMapping),
    Shadow(ShadowDesc),
    Vg(VgSettings),
    GiQuality(GiQuality),
    Gi(GiSettings),
    AreaShadows((AreaShadowSettings, StochasticShadowSettings)),
    AreaShading(bool),
    Gtao(AoDesc),
    Contact(ContactDesc),
}

/// The default descriptor of `feature`; `None` for a feature without keys.
fn default_desc(feature: &FeatureId) -> Option<Desc> {
    Some(match feature {
        FeatureId::Tonemap => Desc::Tonemap(ToneMapping::default()),
        FeatureId::Shadow => Desc::Shadow(ShadowDesc::default()),
        FeatureId::Vg => Desc::Vg(VgSettings::default()),
        FeatureId::GiQuality => Desc::GiQuality(GiQuality::default()),
        FeatureId::Gi => Desc::Gi(GiSettings::default()),
        FeatureId::AreaShadows => Desc::AreaShadows((
            AreaShadowSettings::default(),
            StochasticShadowSettings::standard(),
        )),
        FeatureId::AreaShading => Desc::AreaShading(false),
        FeatureId::Gtao => Desc::Gtao(AoDesc::default()),
        FeatureId::Contact => Desc::Contact(ContactDesc::default()),
        _ => return None,
    })
}

/// The captured descriptor of `feature`; `None` when the feature has none or is absent.
fn desc_of(applied: &AppliedDescs, feature: &FeatureId) -> Option<Desc> {
    Some(match feature {
        FeatureId::Tonemap => Desc::Tonemap(applied.tonemap?),
        FeatureId::Shadow => Desc::Shadow(applied.shadow?),
        FeatureId::Vg => Desc::Vg(applied.vg?),
        FeatureId::GiQuality => Desc::GiQuality(applied.gi_quality?),
        FeatureId::Gi => Desc::Gi(applied.gi?),
        FeatureId::AreaShadows => Desc::AreaShadows(applied.area_shadows?),
        FeatureId::AreaShading => Desc::AreaShading(applied.area_shading?),
        FeatureId::Gtao => Desc::Gtao(applied.gtao?),
        FeatureId::Contact => Desc::Contact(applied.contact?),
        _ => return None,
    })
}

/// The value of `key` in `desc`; `None` for a key the feature does not have.
fn desc_get(desc: &Desc, key: &str) -> Option<ParamValue> {
    match desc {
        Desc::Tonemap(d) => get_tonemap(d, key),
        Desc::Shadow(d) => get_shadow(d, key),
        Desc::Vg(d) => get_vg(d, key),
        Desc::GiQuality(d) => get_gi_quality(d, key),
        Desc::Gi(d) => get_gi(d, key),
        Desc::AreaShadows(d) => get_area_shadows(d, key),
        Desc::AreaShading(on) => get_area_shading(*on, key),
        Desc::Gtao(d) => get_gtao(d, key),
        Desc::Contact(d) => get_contact(d, key),
    }
}

/// Writes `value` to `key` of `desc`. A value of the wrong kind or outside the field's type is
/// refused with the reason; a key the feature does not have is refused as unknown.
fn desc_set(desc: &mut Desc, key: &str, value: &ParamValue) -> Result<(), String> {
    match desc {
        Desc::Tonemap(d) => set_tonemap(d, key, value),
        Desc::Shadow(d) => set_shadow(d, key, value),
        Desc::Vg(d) => set_vg(d, key, value),
        Desc::GiQuality(d) => set_gi_quality(d, key, value),
        Desc::Gi(d) => set_gi(d, key, value),
        Desc::AreaShadows(d) => set_area_shadows(d, key, value),
        Desc::AreaShading(on) => set_area_shading(on, key, value),
        Desc::Gtao(d) => set_gtao(d, key, value),
        Desc::Contact(d) => set_contact(d, key, value),
    }
}

/// The validation the feature's handle runs before it stores `desc`: the descriptor's own
/// `validate`, or for Vg, GiQuality and Gi the patch apply of the full patch over the defaults.
fn desc_check(desc: &Desc) -> Result<(), RaaError> {
    match desc {
        Desc::Tonemap(d) => d.validate(),
        Desc::Shadow(d) => d.validate(),
        Desc::Vg(d) => vg_patch(d).apply(&VgSettings::default()).map(|_| ()),
        Desc::GiQuality(d) => gi_quality_patch(d).apply(&GiQuality::default()).map(|_| ()),
        Desc::Gi(d) => gi_patch(d).apply(&GiSettings::default()).map(|_| ()),
        Desc::AreaShadows((a, s)) => {
            a.validate()?;
            s.validate().map(|_| ())
        }
        Desc::AreaShading(_) => Ok(()),
        Desc::Gtao(d) => d.validate(),
        Desc::Contact(d) => d.validate(),
    }
}

/// Applies `desc` through the feature's live handle, which validates it and refuses without
/// changing anything. An absent handle is `Ok(())`.
fn desc_apply(h: &mut FeatureHandles<'_>, desc: &Desc) -> Result<(), RaaError> {
    match desc {
        Desc::Tonemap(d) => if_installed(h.handles.tonemap.as_ref(), |t| t.set(*d)),
        Desc::Shadow(d) => if_installed(h.handles.shadow.as_ref(), |s| s.set(*d)),
        Desc::Vg(d) => if_installed(h.handles.vg.as_ref(), |vg| vg.set(vg_patch(d))),
        Desc::GiQuality(d) => if_installed(h.handles.gi.as_ref(), |gi| {
            gi.set_quality(gi_quality_patch(d))
        }),
        Desc::Gi(d) => if_installed(h.handles.gi.as_ref(), |gi| gi.set(gi_patch(d))),
        Desc::AreaShadows((a, s)) => match h.lit.as_deref_mut() {
            Some(lit) => {
                // Both settings are validated before either is stored, so a refusal changes nothing.
                let (a, s) = (a.validate()?, s.validate()?);
                lit.set_area_shadows(a)?;
                lit.set_area_stochastic(s)
            }
            None => Ok(()),
        },
        Desc::AreaShading(on) => {
            if let Some(lit) = h.lit.as_deref_mut() {
                lit.set_frame_feature(AREA_FORM_FACTOR_FEATURE_ID, *on);
            }
            Ok(())
        }
        Desc::Gtao(d) => if_installed(h.handles.gtao.as_ref(), |g| g.set(*d)),
        Desc::Contact(d) => if_installed(h.handles.contact.as_ref(), |c| c.set(*d)),
    }
}

/// `apply` on the installed handle; an absent one is `Ok(())`.
fn if_installed<T>(
    handle: Option<&T>,
    apply: impl FnOnce(&T) -> Result<(), RaaError>,
) -> Result<(), RaaError> {
    handle.map_or(Ok(()), apply)
}

/// The full Vg patch of `s`: every field is set, so the patch apply validates the whole
/// descriptor. `runtime_bake` and `instance_default` are not keys; they keep their live values.
fn vg_patch(s: &VgSettings) -> VgSettingsPatch {
    VgSettingsPatch {
        enabled: Some(s.enabled),
        error_pixels: Some(s.error_pixels),
        software_raster: Some(s.software_raster),
        occlusion: Some(s.occlusion),
        shadows: Some(s.shadows),
        runtime_bake: Some(s.runtime_bake),
        instance_default: Some(s.instance_default),
        max_visible_clusters: Some(s.max_visible_clusters),
        debug_view: Some(s.debug_view),
    }
}

/// The full Gi patch of `g`: every field is set, so the patch apply validates the whole
/// descriptor.
fn gi_patch(g: &GiSettings) -> GiSettingsPatch {
    GiSettingsPatch {
        enabled: Some(g.enabled),
        intensity: Some(f64::from(g.intensity)),
        environment_diffuse: Some(f64::from(g.environment_diffuse)),
        local_light_shadows: Some(g.local_light_shadows),
        reflections: Some(g.reflections),
        screen_traces: Some(g.screen_traces),
        debug_view: Some(g.debug_view),
        screen_space: Some(g.screen_space),
        screen_space_quality: Some(g.screen_space_quality),
        indirect: Some(g.indirect),
        history_max: Some(f64::from(g.history_max)),
        clamp_gamma: Some(f64::from(g.clamp_gamma)),
        spatial_passes: Some(f64::from(g.spatial_passes)),
        restir_m_cap: Some(f64::from(g.restir_m_cap)),
        voxel_full_relight: Some(g.voxel_full_relight),
        history_min: Some(f64::from(g.history_min)),
        voxel_history_max: Some(f64::from(g.voxel_history_max)),
        voxel_rays: Some(f64::from(g.voxel_rays)),
    }
}

/// The full GiQuality patch of `q`, validated as a whole by the patch apply.
fn gi_quality_patch(q: &GiQuality) -> GiQualityPatch {
    GiQualityPatch {
        resolution: Some(f64::from(q.resolution)),
        levels: Some(f64::from(q.levels)),
        probe_rays: Some(f64::from(q.probe_rays)),
    }
}

/// The descriptors the executor applied, one `Option` per feature: `None` while the feature is not
/// installed. [`AppliedDescs::capture`] reads the live descriptors and [`AppliedDescs::changed`]
/// lists the features whose live descriptor no longer equals the captured one.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AppliedDescs {
    pub tonemap: Option<ToneMapping>,
    pub shadow: Option<ShadowDesc>,
    pub vg: Option<VgSettings>,
    pub gi_quality: Option<GiQuality>,
    pub gi: Option<GiSettings>,
    pub area_shadows: Option<(AreaShadowSettings, StochasticShadowSettings)>,
    pub area_shading: Option<bool>,
    pub gtao: Option<AoDesc>,
    pub contact: Option<ContactDesc>,
}

impl AppliedDescs {
    /// The live descriptors of the installed features.
    pub fn capture(h: &FeatureHandles<'_>) -> AppliedDescs {
        AppliedDescs {
            tonemap: h.handles.tonemap.as_ref().map(TonemapControl::get),
            shadow: h.handles.shadow.as_ref().map(ShadowHandle::get),
            vg: h.handles.vg.as_ref().map(VgHandle::get),
            gi_quality: h.handles.gi.as_ref().map(GiHandle::quality),
            gi: h.handles.gi.as_ref().map(GiHandle::get),
            area_shadows: h
                .lit
                .as_deref()
                .map(|lit| (lit.area_shadows(), lit.area_stochastic())),
            area_shading: h
                .lit
                .as_deref()
                .map(|lit| lit.frame_feature(AREA_FORM_FACTOR_FEATURE_ID)),
            gtao: h.handles.gtao.as_ref().map(GtaoHandle::get),
            contact: h.handles.contact.as_ref().map(ContactHandle::get),
        }
    }

    /// Pushes, in S53 rank order, each feature whose live descriptor differs from this one. `out`
    /// is not cleared. Nothing is allocated when nothing differs: the descriptors are compared.
    pub fn changed(&self, h: &FeatureHandles<'_>, out: &mut Vec<FeatureId>) {
        let live = AppliedDescs::capture(h);
        if self.tonemap != live.tonemap {
            out.push(FeatureId::Tonemap);
        }
        if self.shadow != live.shadow {
            out.push(FeatureId::Shadow);
        }
        if self.vg != live.vg {
            out.push(FeatureId::Vg);
        }
        if self.gi_quality != live.gi_quality {
            out.push(FeatureId::GiQuality);
        }
        if self.gi != live.gi {
            out.push(FeatureId::Gi);
        }
        if self.area_shadows != live.area_shadows {
            out.push(FeatureId::AreaShadows);
        }
        if self.area_shading != live.area_shading {
            out.push(FeatureId::AreaShading);
        }
        if self.gtao != live.gtao {
            out.push(FeatureId::Gtao);
        }
        if self.contact != live.contact {
            out.push(FeatureId::Contact);
        }
    }
}

fn get_tonemap(d: &ToneMapping, key: &str) -> Option<ParamValue> {
    match key {
        "mapper" => name_in(&MAPPERS, d.mapper),
        "exposureEv" => Some(f32_value(d.exposure_ev)),
        "look" => name_in(&LOOKS, d.look),
        "dither" => Some(ParamValue::Bool(d.dither)),
        _ => None,
    }
}

fn set_tonemap(d: &mut ToneMapping, key: &str, value: &ParamValue) -> Result<(), String> {
    match key {
        "mapper" => d.mapper = choice(key, value, |text| lookup(&MAPPERS, text))?,
        "exposureEv" => d.exposure_ev = float(key, value)?,
        "look" => d.look = choice(key, value, |text| lookup(&LOOKS, text))?,
        "dither" => d.dither = flag(key, value)?,
        _ => return Err(unknown(key)),
    }
    Ok(())
}

fn get_shadow(d: &ShadowDesc, key: &str) -> Option<ParamValue> {
    match key {
        "softness" => Some(f32_value(d.softness)),
        "distance" => Some(f32_value(d.distance_m)),
        "resolution" => Some(u32_value(d.resolution)),
        _ => None,
    }
}

fn set_shadow(d: &mut ShadowDesc, key: &str, value: &ParamValue) -> Result<(), String> {
    match key {
        "softness" => d.softness = float(key, value)?,
        "distance" => d.distance_m = float(key, value)?,
        "resolution" => d.resolution = count(key, value)?,
        _ => return Err(unknown(key)),
    }
    Ok(())
}

fn get_vg(d: &VgSettings, key: &str) -> Option<ParamValue> {
    match key {
        "enabled" => Some(ParamValue::Bool(d.enabled)),
        "errorPixels" => Some(f32_value(d.error_pixels)),
        "softwareRaster" => Some(ParamValue::Bool(d.software_raster)),
        "occlusion" => Some(ParamValue::Bool(d.occlusion)),
        "shadows" => Some(ParamValue::Bool(d.shadows)),
        "maxVisibleClusters" => Some(u32_value(d.max_visible_clusters)),
        "debugView" => Some(named(d.debug_view.name())),
        _ => None,
    }
}

fn set_vg(d: &mut VgSettings, key: &str, value: &ParamValue) -> Result<(), String> {
    match key {
        "enabled" => d.enabled = flag(key, value)?,
        "errorPixels" => d.error_pixels = float(key, value)?,
        "softwareRaster" => d.software_raster = flag(key, value)?,
        "occlusion" => d.occlusion = flag(key, value)?,
        "shadows" => d.shadows = flag(key, value)?,
        "maxVisibleClusters" => d.max_visible_clusters = count(key, value)?,
        "debugView" => d.debug_view = choice(key, value, VgDebugView::from_name)?,
        _ => return Err(unknown(key)),
    }
    Ok(())
}

fn get_gi_quality(d: &GiQuality, key: &str) -> Option<ParamValue> {
    match key {
        "resolution" => Some(u32_value(d.resolution)),
        "levels" => Some(u32_value(d.levels)),
        "probeRays" => Some(u32_value(d.probe_rays)),
        _ => None,
    }
}

fn set_gi_quality(d: &mut GiQuality, key: &str, value: &ParamValue) -> Result<(), String> {
    match key {
        "resolution" => d.resolution = count(key, value)?,
        "levels" => d.levels = count(key, value)?,
        "probeRays" => d.probe_rays = count(key, value)?,
        _ => return Err(unknown(key)),
    }
    Ok(())
}

fn get_gi(d: &GiSettings, key: &str) -> Option<ParamValue> {
    match key {
        "enabled" => Some(ParamValue::Bool(d.enabled)),
        "screenSpace" => Some(ParamValue::Bool(d.screen_space)),
        "intensity" => Some(f32_value(d.intensity)),
        "environmentDiffuse" => Some(f32_value(d.environment_diffuse)),
        "localLightShadows" => Some(ParamValue::Bool(d.local_light_shadows)),
        "voxelFullRelight" => Some(ParamValue::Bool(d.voxel_full_relight)),
        "voxelHistoryMax" => Some(u32_value(d.voxel_history_max)),
        "voxelRays" => Some(u32_value(d.voxel_rays)),
        "indirect" => Some(named(d.indirect.as_str())),
        "screenTraces" => Some(ParamValue::Bool(d.screen_traces)),
        "reflections" => Some(ParamValue::Bool(d.reflections)),
        "historyMax" => Some(u32_value(d.history_max)),
        "historyMin" => Some(u32_value(d.history_min)),
        "clampGamma" => Some(f32_value(d.clamp_gamma)),
        "spatialPasses" => Some(u32_value(d.spatial_passes)),
        "restirMCap" => Some(f32_value(d.restir_m_cap)),
        "debugView" => Some(named(d.debug_view.as_str())),
        "screenSpaceQuality" => Some(named(d.screen_space_quality.as_str())),
        _ => None,
    }
}

fn set_gi(d: &mut GiSettings, key: &str, value: &ParamValue) -> Result<(), String> {
    match key {
        "enabled" => d.enabled = flag(key, value)?,
        "screenSpace" => d.screen_space = flag(key, value)?,
        "intensity" => d.intensity = float(key, value)?,
        "environmentDiffuse" => d.environment_diffuse = float(key, value)?,
        "localLightShadows" => d.local_light_shadows = flag(key, value)?,
        "voxelFullRelight" => d.voxel_full_relight = flag(key, value)?,
        "voxelHistoryMax" => d.voxel_history_max = count(key, value)?,
        "voxelRays" => d.voxel_rays = count(key, value)?,
        "indirect" => d.indirect = choice(key, value, IndirectMode::parse)?,
        "screenTraces" => d.screen_traces = flag(key, value)?,
        "reflections" => d.reflections = flag(key, value)?,
        "historyMax" => d.history_max = count(key, value)?,
        "historyMin" => d.history_min = count(key, value)?,
        "clampGamma" => d.clamp_gamma = float(key, value)?,
        "spatialPasses" => d.spatial_passes = count(key, value)?,
        "restirMCap" => d.restir_m_cap = float(key, value)?,
        "debugView" => d.debug_view = choice(key, value, GiDebugView::parse)?,
        "screenSpaceQuality" => d.screen_space_quality = choice(key, value, SsgiQuality::parse)?,
        _ => return Err(unknown(key)),
    }
    Ok(())
}

fn get_area_shadows(
    d: &(AreaShadowSettings, StochasticShadowSettings),
    key: &str,
) -> Option<ParamValue> {
    let (a, s) = d;
    match key {
        "tier" => Some(named(a.tier.as_str())),
        "heroCount" => Some(u32_value(a.hero_count)),
        "heroDistance" => Some(f32_value(a.hero_distance)),
        "fadeDistance" => Some(f32_value(a.fade_distance)),
        "band" => Some(f32_value(a.band)),
        "stochasticCount" => Some(u32_value(s.count)),
        "stochasticDistance" => Some(f32_value(s.distance)),
        "stochasticCells" => Some(u32_value(s.cells)),
        "stochasticMode" => Some(named(s.mode.as_str())),
        _ => None,
    }
}

fn set_area_shadows(
    d: &mut (AreaShadowSettings, StochasticShadowSettings),
    key: &str,
    value: &ParamValue,
) -> Result<(), String> {
    let (a, s) = d;
    match key {
        "tier" => a.tier = choice(key, value, AreaShadowTier::from_name)?,
        "heroCount" => a.hero_count = count(key, value)?,
        "heroDistance" => a.hero_distance = float(key, value)?,
        "fadeDistance" => a.fade_distance = float(key, value)?,
        "band" => a.band = float(key, value)?,
        "stochasticCount" => s.count = count(key, value)?,
        "stochasticDistance" => s.distance = float(key, value)?,
        "stochasticCells" => s.cells = count(key, value)?,
        "stochasticMode" => s.mode = choice(key, value, StochasticMode::from_name)?,
        _ => return Err(unknown(key)),
    }
    Ok(())
}

fn get_area_shading(on: bool, key: &str) -> Option<ParamValue> {
    match key {
        "mode" => name_in(&SHADING, on),
        _ => None,
    }
}

fn set_area_shading(on: &mut bool, key: &str, value: &ParamValue) -> Result<(), String> {
    match key {
        "mode" => *on = choice(key, value, |text| lookup(&SHADING, text))?,
        _ => return Err(unknown(key)),
    }
    Ok(())
}

fn get_gtao(d: &AoDesc, key: &str) -> Option<ParamValue> {
    match key {
        "radius" => Some(f32_value(d.radius_m)),
        "intensity" => Some(f32_value(d.intensity)),
        "power" => Some(f32_value(d.power)),
        _ => None,
    }
}

fn set_gtao(d: &mut AoDesc, key: &str, value: &ParamValue) -> Result<(), String> {
    match key {
        "radius" => d.radius_m = float(key, value)?,
        "intensity" => d.intensity = float(key, value)?,
        "power" => d.power = float(key, value)?,
        _ => return Err(unknown(key)),
    }
    Ok(())
}

fn get_contact(d: &ContactDesc, key: &str) -> Option<ParamValue> {
    match key {
        "distance" => Some(f32_value(d.distance_m)),
        "thickness" => Some(f32_value(d.thickness_m)),
        "intensity" => Some(f32_value(d.intensity)),
        _ => None,
    }
}

fn set_contact(d: &mut ContactDesc, key: &str, value: &ParamValue) -> Result<(), String> {
    match key {
        "distance" => d.distance_m = float(key, value)?,
        "thickness" => d.thickness_m = float(key, value)?,
        "intensity" => d.intensity = float(key, value)?,
        _ => return Err(unknown(key)),
    }
    Ok(())
}

fn f32_value(value: f32) -> ParamValue {
    ParamValue::Number(f64::from(value))
}

fn u32_value(value: u32) -> ParamValue {
    ParamValue::Int(i64::from(value))
}

fn named(name: &str) -> ParamValue {
    ParamValue::Enum(name.to_owned())
}

/// The JS name of `value` in `table`.
fn name_in<T: Copy + PartialEq>(table: &[(&'static str, T)], value: T) -> Option<ParamValue> {
    table
        .iter()
        .find(|(_, candidate)| *candidate == value)
        .map(|(name, _)| named(name))
}

/// The value named `text` in `table`.
fn lookup<T: Copy>(table: &[(&'static str, T)], text: &str) -> Option<T> {
    table
        .iter()
        .find(|(name, _)| *name == text)
        .map(|(_, value)| *value)
}

fn float(key: &str, value: &ParamValue) -> Result<f32, String> {
    match value {
        ParamValue::Number(number) => Ok(*number as f32),
        other => Err(expected(key, "a number", other)),
    }
}

fn count(key: &str, value: &ParamValue) -> Result<u32, String> {
    match value {
        ParamValue::Int(int) => u32::try_from(*int)
            .map_err(|_| format!("{key}: {int} is not an unsigned 32-bit integer")),
        other => Err(expected(key, "an integer", other)),
    }
}

fn flag(key: &str, value: &ParamValue) -> Result<bool, String> {
    match value {
        ParamValue::Bool(flag) => Ok(*flag),
        other => Err(expected(key, "a boolean", other)),
    }
}

/// The enum value `parse` reads from the name in `value`.
fn choice<T>(
    key: &str,
    value: &ParamValue,
    parse: impl Fn(&str) -> Option<T>,
) -> Result<T, String> {
    match value {
        ParamValue::Enum(text) => {
            parse(text.as_str()).ok_or_else(|| format!("{key}: '{text}' is not an allowed value"))
        }
        other => Err(expected(key, "an enum name", other)),
    }
}

fn expected(key: &str, what: &str, value: &ParamValue) -> String {
    format!("{key}: expected {what}, got {value:?}")
}

fn unknown(key: &str) -> String {
    format!("{key}: not a setting of this feature")
}

#[cfg(test)]
mod tests {
    use super::*;

    const FEATURES: [FeatureId; 9] = [
        FeatureId::Tonemap,
        FeatureId::Shadow,
        FeatureId::Vg,
        FeatureId::GiQuality,
        FeatureId::Gi,
        FeatureId::AreaShadows,
        FeatureId::AreaShading,
        FeatureId::Gtao,
        FeatureId::Contact,
    ];

    fn pair(key: &str, value: ParamValue) -> (String, ParamValue) {
        (key.to_owned(), value)
    }

    #[test]
    fn gtao_defaults_are_the_ao_desc_defaults() {
        assert_eq!(
            default_values(&FeatureId::Gtao),
            vec![
                pair("radius", ParamValue::Number(0.8f32 as f64)),
                pair("intensity", ParamValue::Number(0.6f32 as f64)),
                pair("power", ParamValue::Number(1.0)),
            ]
        );
    }

    #[test]
    fn radius_25_is_refused_under_its_key() {
        let changes = [pair("radius", ParamValue::Number(25.0))];
        match merge_values(&FeatureId::Gtao, &[], &changes) {
            Err((key, message)) => {
                assert_eq!(key, "radius");
                assert!(message.contains("radius_m"), "message: {message}");
            }
            other => panic!("radius 25 must be refused, got {other:?}"),
        }
    }

    #[test]
    fn enum_into_a_number_key_is_refused_with_that_key() {
        let changes = [pair("radius", ParamValue::Enum("long".to_owned()))];
        match merge_values(&FeatureId::Gtao, &[], &changes) {
            Err((key, _)) => assert_eq!(key, "radius"),
            other => panic!("an enum into radius must be refused, got {other:?}"),
        }
    }

    #[test]
    fn unknown_key_is_refused_under_its_name() {
        let changes = [pair("nope", ParamValue::Number(1.0))];
        match merge_values(&FeatureId::Gtao, &[], &changes) {
            Err((key, _)) => assert_eq!(key, "nope"),
            other => panic!("an unknown key must be refused, got {other:?}"),
        }
    }

    #[test]
    fn changes_apply_over_the_base_values() {
        let base = [pair("radius", ParamValue::Number(2.0))];
        let changes = [pair("power", ParamValue::Number(3.0))];
        assert_eq!(
            merge_values(&FeatureId::Gtao, &base, &changes),
            Ok(vec![
                pair("radius", ParamValue::Number(2.0)),
                pair("intensity", ParamValue::Number(0.6f32 as f64)),
                pair("power", ParamValue::Number(3.0)),
            ])
        );
    }

    #[test]
    fn area_shading_mode_takes_its_enum_names() {
        let changes = [pair("mode", ParamValue::Enum("formFactor".to_owned()))];
        assert_eq!(
            merge_values(&FeatureId::AreaShading, &[], &changes),
            Ok(vec![pair(
                "mode",
                ParamValue::Enum("formFactor".to_owned())
            )])
        );
    }

    #[test]
    fn vg_snapshot_ends_with_enabled() {
        assert_eq!(snapshot_keys(&FeatureId::Vg).last(), Some(&"enabled"));
        assert!(snapshot_keys(&FeatureId::Gi).ends_with(&["enabled", "screenSpace"]));
    }

    #[test]
    fn every_feature_round_trips_its_defaults() {
        for feature in &FEATURES {
            let defaults = default_values(feature);
            let keys: Vec<&str> = defaults.iter().map(|(key, _)| key.as_str()).collect();
            assert_eq!(keys, snapshot_keys(feature), "{feature:?}");
            assert_eq!(
                merge_values(feature, &[], &defaults),
                Ok(defaults.clone()),
                "{feature:?}"
            );
        }
    }

    #[test]
    fn features_without_keys_answer_empty_values() {
        for feature in [FeatureId::Pipeline, FeatureId::Taa, FeatureId::Other("x")] {
            assert!(snapshot_keys(&feature).is_empty());
            assert!(default_values(&feature).is_empty());
            assert_eq!(merge_values(&feature, &[], &[]), Ok(Vec::new()));
        }
    }
}
