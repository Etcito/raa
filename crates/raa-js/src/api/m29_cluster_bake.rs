//! m29 cluster bake, shares 1 and 2: the pure helpers of the cluster-bake API (S31), the section
//! wire validation (S30) and the state report, and the wasm32 runtime of the shared cluster store
//! (`install`, `frame`).
//!
//! The helpers are plain Rust, so `tests/cluster_bake.rs` includes this file and exercises
//! them natively: the refusal codes (`vg.invalid`, `vg.mesh_empty`, `vg.mesh_too_large`,
//! `vg.mesh_vertex_colors`), the option patch and its S31 domains, the bake input of the flat
//! JS arrays (S13 lengths), the budget domain and the stage names. Errors are
//! `RaaError::FrameGraph` values, because the included file cannot name `crate::convert`.
//!
//! The wasm32 part (`web`) inserts one shared `ClusterStore` and the bake budget in the feature
//! map. Each Engine render invocation steps the baking meshes once, within the budget, on the
//! clock of `globalThis.performance.now` (the js-sys date clock as fallback). The same module
//! holds the JS conversions of the section and of the bake input (S30, S31) and the exports
//! `bakeClusters`, `validateClusters`, `clusterCutMesh` and the sliced class `ClusterBake`.

// Some helpers are reached only by the host tests or only by the Engine methods of a later card,
// so they are dead in one of the two builds.
#![allow(dead_code)]

use raa_cluster::error::{EMPTY, TOO_LARGE};
use raa_cluster::{
    BakeStage, BuildOptions, ClusterError, ClusterMesh, ClusterSource, ClusterState,
    MeshClusterStatus, OwnedClusterInput, PACKED_CLUSTER_SIZE, PACKED_NODE_SIZE, SectionHeader,
};
use raa_core::RaaError;
use raa_geom::MeshData;

/// A malformed option, array, handle or budget (S31).
pub(crate) const VG_INVALID: &str = "vg.invalid";
/// No triangle to bake, or no triangle survives quantisation (S31).
pub(crate) const MESH_EMPTY: &str = "vg.mesh_empty";
/// The mesh exceeds the S29 limits (S31).
pub(crate) const MESH_TOO_LARGE: &str = "vg.mesh_too_large";
/// The mesh has a colour stream, which the 16 B cluster vertex cannot carry (S31).
pub(crate) const MESH_VERTEX_COLORS: &str = "vg.mesh_vertex_colors";

/// The default bake budget of one Engine render invocation, in milliseconds (VG_BAKE_BUDGET).
const DEFAULT_BAKE_BUDGET_MS: f64 = 4.0;
/// The smallest accepted Engine bake budget, in milliseconds (VG_BAKE_BUDGET).
const BAKE_BUDGET_MIN_MS: f64 = 0.5;
/// The largest accepted Engine bake budget, in milliseconds (VG_BAKE_BUDGET).
const BAKE_BUDGET_MAX_MS: f64 = 100.0;

/// The `vg.invalid` refusal with the field path first: `<field>: <reason>`.
pub(crate) fn vg_invalid(field: &str, reason: impl std::fmt::Display) -> RaaError {
    RaaError::FrameGraph {
        code: VG_INVALID,
        message: format!("{field}: {reason}"),
    }
}

/// The refusal of a cluster error: `cluster.empty` and `cluster.too_large` keep their vg
/// meaning, every other code is `vg.invalid`. The message keeps the path and the message.
pub(crate) fn cluster_error(e: &ClusterError) -> RaaError {
    let code = match e.code {
        EMPTY => MESH_EMPTY,
        TOO_LARGE => MESH_TOO_LARGE,
        _ => VG_INVALID,
    };
    let message = if e.path.is_empty() {
        e.message.clone()
    } else {
        format!("{}: {}", e.path, e.message)
    };
    RaaError::FrameGraph { code, message }
}

/// The vg code of a stored code (`MeshClusterStatus.code`): the table of [`cluster_error`],
/// and a code that already starts with `vg.` is returned unchanged.
pub(crate) fn vg_code(code: &'static str) -> &'static str {
    match code {
        EMPTY => MESH_EMPTY,
        TOO_LARGE => MESH_TOO_LARGE,
        _ if code.starts_with("vg.") => code,
        _ => VG_INVALID,
    }
}

/// The build options as JS sends them; an absent (undefined) field takes the S31 default.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct OptionsPatch {
    pub max_cluster_triangles: Option<f64>,
    pub max_cluster_vertices: Option<f64>,
    pub min_reduction: Option<f64>,
    pub target_ratio: Option<f64>,
}

/// The `BuildOptions` of a patch: integers 8..=128 and 16..=255, fractions in (0, 1), all
/// finite. Each refusal is `vg.invalid` naming `options.<camelCase field>`.
pub(crate) fn build_options(patch: &OptionsPatch) -> Result<BuildOptions, RaaError> {
    let base = BuildOptions::default();
    Ok(BuildOptions {
        max_cluster_triangles: integer(
            patch.max_cluster_triangles,
            (8.0, 128.0),
            base.max_cluster_triangles,
            "options.maxClusterTriangles",
        )?,
        max_cluster_vertices: integer(
            patch.max_cluster_vertices,
            (16.0, 255.0),
            base.max_cluster_vertices,
            "options.maxClusterVertices",
        )?,
        min_reduction: fraction(
            patch.min_reduction,
            base.min_reduction,
            "options.minReduction",
        )?,
        target_ratio: fraction(patch.target_ratio, base.target_ratio, "options.targetRatio")?,
    })
}

fn integer(
    value: Option<f64>,
    (min, max): (f64, f64),
    default: usize,
    field: &str,
) -> Result<usize, RaaError> {
    let Some(v) = value else {
        return Ok(default);
    };
    if v.is_finite() && v.fract() == 0.0 && (min..=max).contains(&v) {
        Ok(v as usize)
    } else {
        Err(vg_invalid(
            field,
            format!("must be an integer in {min}..={max}, got {v}"),
        ))
    }
}

fn fraction(value: Option<f64>, default: f32, field: &str) -> Result<f32, RaaError> {
    let Some(v) = value else {
        return Ok(default);
    };
    // The check is on the f32 the options carry, so a value that rounds to 1 is refused.
    let f = v as f32;
    if v.is_finite() && f > 0.0 && f < 1.0 {
        Ok(f)
    } else {
        Err(vg_invalid(field, format!("must be in (0, 1), got {v}")))
    }
}

/// The bake input of the flat JS arrays: positions 3n, normals 3n, uvs 2n (optional) and
/// indices 3m, then `ClusterInput::validate`. No triangle is `vg.mesh_empty`.
pub(crate) fn bake_input(
    positions: &[f32],
    normals: &[f32],
    uvs: Option<&[f32]>,
    indices: &[u32],
) -> Result<OwnedClusterInput, RaaError> {
    if !positions.len().is_multiple_of(3) {
        return Err(vg_invalid(
            "positions",
            format!("length {} is not a multiple of 3", positions.len()),
        ));
    }
    let count = positions.len() / 3;
    if normals.len() != positions.len() {
        return Err(vg_invalid(
            "normals",
            format!(
                "length {} does not match the length {} of positions",
                normals.len(),
                positions.len()
            ),
        ));
    }
    if let Some(uvs) = uvs.filter(|uvs| uvs.len() != 2 * count) {
        return Err(vg_invalid(
            "uvs",
            format!(
                "length {} is not 2 per vertex for {count} vertices",
                uvs.len()
            ),
        ));
    }
    if !indices.len().is_multiple_of(3) {
        return Err(vg_invalid(
            "indices",
            format!("length {} is not a multiple of 3", indices.len()),
        ));
    }
    if let Some(index) = indices.iter().find(|&&i| i as usize >= count) {
        return Err(vg_invalid(
            "indices",
            format!("index {index} is out of range for {count} vertices"),
        ));
    }
    if indices.is_empty() {
        return Err(RaaError::FrameGraph {
            code: MESH_EMPTY,
            message: "indices: the mesh has no triangle".to_string(),
        });
    }
    let owned = OwnedClusterInput {
        mesh: MeshData {
            positions: positions
                .chunks_exact(3)
                .map(|p| [p[0], p[1], p[2]])
                .collect(),
            normals: normals
                .chunks_exact(3)
                .map(|n| [n[0], n[1], n[2]])
                .collect(),
            indices: indices.to_vec(),
        },
        uvs: uvs.map(|uvs| uvs.chunks_exact(2).map(|uv| [uv[0], uv[1]]).collect()),
    };
    owned.as_input().validate().map_err(|e| cluster_error(&e))?;
    Ok(owned)
}

/// A budget in milliseconds: finite and inside `[min, max]`, else `vg.invalid` naming `field`.
pub(crate) fn check_budget_ms(ms: f64, min: f64, max: f64, field: &str) -> Result<f64, RaaError> {
    if ms.is_finite() && (min..=max).contains(&ms) {
        Ok(ms)
    } else {
        Err(vg_invalid(
            field,
            format!("must be a finite number in {min}..={max} milliseconds, got {ms}"),
        ))
    }
}

/// The S31 name of a bake stage.
pub(crate) fn stage_name(stage: BakeStage) -> &'static str {
    match stage {
        BakeStage::Weld => "weld",
        BakeStage::Partition => "partition",
        BakeStage::Level => "level",
        BakeStage::Hierarchy => "hierarchy",
        BakeStage::Pack => "pack",
        BakeStage::Done => "done",
    }
}

/// The saved-mesh cluster section as the JS conversion reads it (S30): the arrays as they are and
/// every number as f64. `version` and `algorithm` are the S30 literals 1.
pub(crate) struct SectionWire<'a> {
    pub version: f64,
    pub algorithm: f64,
    pub vertices: &'a [u16],
    pub triangles: &'a [u32],
    pub clusters: &'a [u8],
    pub nodes: &'a [u8],
    pub root_node: f64,
    pub hierarchy_depth: f64,
    pub levels: f64,
    pub quant_origin: [f64; 3],
    pub quant_step: f64,
    pub uv_origin: [f64; 2],
    pub uv_step: [f64; 2],
    pub bounds_center: [f64; 3],
    pub bounds_radius: f64,
    pub source_triangles: f64,
}

/// A refusal of one section field: `vg.invalid` naming `<at><field>`.
fn wire_refusal(at: &str, field: &str, reason: &str) -> RaaError {
    vg_invalid(&format!("{at}{field}"), reason)
}

/// A whole number in the inclusive range, returned as u64 (the caller narrows it to its type).
fn whole(at: &str, field: &str, value: f64, (min, max): (f64, f64)) -> Result<u64, RaaError> {
    if value.is_finite() && value.fract() == 0.0 && (min..=max).contains(&value) {
        Ok(value as u64)
    } else {
        let reason = format!("must be an integer in {min}..={max}, got {value}");
        Err(wire_refusal(at, field, &reason))
    }
}

/// Every value stays finite when narrowed to f32, the precision the section carries.
fn finite32(values: &[f64]) -> bool {
    values.iter().all(|&v| (v as f32).is_finite())
}

/// The S30 wire rules on the version, the array lengths and the scalars, then the complete
/// `ClusterMesh::from_section_parts` validation. Every refusal is `vg.invalid` with the field
/// named as `<at><camelCase field or ClusterError path>`; `at` is empty or `sections[i].`.
pub(crate) fn section_from_wire(w: &SectionWire<'_>, at: &str) -> Result<ClusterMesh, RaaError> {
    if w.version != 1.0 {
        return Err(wire_refusal(at, "version", "must be 1"));
    }
    if w.algorithm != 1.0 {
        return Err(wire_refusal(at, "algorithm", "must be 1"));
    }
    if w.vertices.is_empty() || !w.vertices.len().is_multiple_of(8) {
        let reason = "must be a non-empty multiple of 8 u16";
        return Err(wire_refusal(at, "vertices", reason));
    }
    if w.triangles.is_empty() {
        return Err(wire_refusal(at, "triangles", "must not be empty"));
    }
    if w.clusters.is_empty() || !w.clusters.len().is_multiple_of(PACKED_CLUSTER_SIZE) {
        let reason = "must be a non-empty multiple of 48 bytes";
        return Err(wire_refusal(at, "clusters", reason));
    }
    if w.nodes.is_empty() || !w.nodes.len().is_multiple_of(PACKED_NODE_SIZE) {
        let reason = "must be a non-empty multiple of 48 bytes";
        return Err(wire_refusal(at, "nodes", reason));
    }
    let node_count = (w.nodes.len() / PACKED_NODE_SIZE) as u64;
    let root = whole(at, "rootNode", w.root_node, (0.0, u32::MAX as f64))?;
    if root >= node_count {
        let reason = format!("must be below the node count {node_count}");
        return Err(wire_refusal(at, "rootNode", &reason));
    }
    let depth = whole(at, "hierarchyDepth", w.hierarchy_depth, (1.0, 64.0))?;
    let levels = whole(at, "levels", w.levels, (1.0, 256.0))?;
    // The largest f64 below 2^64: every whole number of f64 under 2^64 is at most this.
    let source = whole(
        at,
        "sourceTriangles",
        w.source_triangles,
        (0.0, 18446744073709549568.0),
    )?;
    if !finite32(&w.quant_origin) {
        return Err(wire_refusal(at, "quantOrigin", "must be finite"));
    }
    if !(finite32(&[w.quant_step]) && w.quant_step > 0.0) {
        return Err(wire_refusal(
            at,
            "quantStep",
            "must be finite and above zero",
        ));
    }
    if !finite32(&w.uv_origin) {
        return Err(wire_refusal(at, "uvOrigin", "must be finite"));
    }
    if !(finite32(&w.uv_step) && w.uv_step.iter().all(|&v| v >= 0.0)) {
        return Err(wire_refusal(
            at,
            "uvStep",
            "must be finite and not negative",
        ));
    }
    if !finite32(&w.bounds_center) {
        return Err(wire_refusal(at, "boundsCenter", "must be finite"));
    }
    if !(finite32(&[w.bounds_radius]) && w.bounds_radius >= 0.0) {
        return Err(wire_refusal(
            at,
            "boundsRadius",
            "must be finite and not negative",
        ));
    }
    let header = SectionHeader {
        root_node: root as u32,
        hierarchy_depth: depth as u32,
        levels: levels as u32,
        quant_origin: w.quant_origin.map(|v| v as f32),
        quant_step: w.quant_step as f32,
        uv_origin: w.uv_origin.map(|v| v as f32),
        uv_step: w.uv_step.map(|v| v as f32),
        bounds_center: w.bounds_center.map(|v| v as f32),
        bounds_radius: w.bounds_radius as f32,
        source_triangles: source,
    };
    ClusterMesh::from_section_parts(w.vertices, w.triangles, w.clusters, w.nodes, header)
        .map_err(|e| vg_invalid(&format!("{at}{}", e.path), e.message))
}

/// The state of one mesh as the JS `meshClusters` reports it (S31).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ClusterReport {
    pub state: &'static str,
    pub progress: f32,
    pub source: Option<&'static str>,
    pub code: Option<&'static str>,
    pub message: Option<String>,
    pub sections: u32,
    pub clusters: u32,
    pub source_triangles: u64,
    pub levels: u32,
}

/// The report of a key: the store state with its code through [`vg_code`]. The section counts
/// (summed clusters and source triangles, the maximum of the levels) are zero unless ready.
pub(crate) fn cluster_report(
    status: &MeshClusterStatus,
    sections: Option<&[ClusterMesh]>,
) -> ClusterReport {
    let state = match status.state {
        ClusterState::None => "none",
        ClusterState::Baking => "baking",
        ClusterState::Ready => "ready",
        ClusterState::Refused => "refused",
    };
    let source = status.source.map(|source| match source {
        ClusterSource::Attached => "attached",
        ClusterSource::Runtime => "runtime",
    });
    let mut report = ClusterReport {
        state,
        progress: status.progress,
        source,
        code: status.code.map(vg_code),
        message: status.message.clone(),
        sections: 0,
        clusters: 0,
        source_triangles: 0,
        levels: 0,
    };
    if let (ClusterState::Ready, Some(sections)) = (status.state, sections) {
        report.sections = sections.len() as u32;
        for section in sections {
            report.clusters = report
                .clusters
                .saturating_add(section.clusters.len() as u32);
            report.source_triangles = report
                .source_triangles
                .saturating_add(section.source_triangles);
            report.levels = report.levels.max(section.levels);
        }
    }
    report
}

/// The Engine bake budget (`setClusterBakeBudget`): finite and inside [0.5, 100] milliseconds
/// per render invocation (VG_BAKE_BUDGET). A refusal is `vg.invalid` naming `ms`.
pub(crate) fn check_bake_budget(ms: f64) -> Result<f64, RaaError> {
    check_budget_ms(ms, BAKE_BUDGET_MIN_MS, BAKE_BUDGET_MAX_MS, "ms")
}

/// Refuses a mesh with a colour stream: its baked vertex cannot carry the four linear
/// components, so the code is `vg.mesh_vertex_colors`. `field` names the argument.
pub(crate) fn check_vertex_colors(has_colors: bool, field: &str) -> Result<(), RaaError> {
    if has_colors {
        Err(RaaError::FrameGraph {
            code: MESH_VERTEX_COLORS,
            message: format!(
                "{field}: the mesh has a vertex colour stream, which the 16 B cluster vertex cannot carry"
            ),
        })
    } else {
        Ok(())
    }
}

/// Plans an atomic attach from already-converted section results: every entry must have
/// converted, and there must be one section per sub-mesh (`sections` names a count refusal).
/// An invalid entry refuses the whole list and returns nothing to attach.
pub(crate) fn plan_attach(
    expected: usize,
    sections: impl IntoIterator<Item = Result<ClusterMesh, RaaError>>,
) -> Result<Vec<ClusterMesh>, RaaError> {
    let converted = sections.into_iter().collect::<Result<Vec<_>, _>>()?;
    if converted.len() != expected {
        return Err(vg_invalid(
            "sections",
            format!(
                "must have one section per sub-mesh: expected {expected}, got {}",
                converted.len()
            ),
        ));
    }
    Ok(converted)
}

#[cfg(target_arch = "wasm32")]
pub(crate) use web::{frame, install};

#[cfg(target_arch = "wasm32")]
mod web {
    use super::*;
    use crate::engine::{Engine, InstallCtx};
    use crate::state::{EngineState, MeshSources};
    use js_sys::{
        Array, Float32Array, Function, Object, Reflect, Uint8Array, Uint16Array, Uint32Array,
    };
    use raa_cluster::{
        BakeProgress, BakeStatus, ClusterBake as SlicedBake, ClusterStore, WorkBudget,
        cut_triangles, select_cut,
    };
    use raa_core::GpuContext;
    use raa_web::{FrameInput, error_to_js};
    use std::cell::RefCell;
    use std::rc::Rc;
    use wasm_bindgen::{JsCast, JsValue, prelude::wasm_bindgen};

    /// Milliseconds of CPU per Engine render invocation for the runtime bake (S31).
    pub(crate) struct ClusterBakeBudget(pub f64);

    /// Inserts the shared cluster store, the default bake budget and the mesh-source table
    /// in the feature map.
    pub(crate) fn install(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
        ctx.features
            .insert(Rc::new(RefCell::new(ClusterStore::new())));
        ctx.features
            .insert(ClusterBakeBudget(DEFAULT_BAKE_BUDGET_MS));
        ctx.features.insert(MeshSources::default());
        Ok(())
    }

    /// Steps the baking meshes once per invocation. The clock is read once for the start; each
    /// unit runs while the clock is below the start plus the budget. No baking key, no clock.
    pub(crate) fn frame(
        st: &mut EngineState,
        _gpu: &GpuContext,
        _input: &FrameInput,
    ) -> Result<(), RaaError> {
        let Some(store) = st.features.get::<Rc<RefCell<ClusterStore>>>() else {
            return Ok(());
        };
        if store.borrow().baking() == 0 {
            return Ok(());
        }
        let budget = st
            .features
            .get::<ClusterBakeBudget>()
            .map_or(DEFAULT_BAKE_BUDGET_MS, |b| b.0);
        let deadline = now_ms() + budget;
        let mut clock = || now_ms() < deadline;
        store.borrow_mut().step_all(&mut clock);
        Ok(())
    }

    /// Milliseconds of `globalThis.performance.now`, read through reflection and a function
    /// call; the js-sys date clock when `performance` is absent (browser or Node).
    pub(crate) fn now_ms() -> f64 {
        let global = js_sys::global();
        if let Ok(perf) = Reflect::get(&global, &JsValue::from_str("performance"))
            && let Ok(now) = Reflect::get(&perf, &JsValue::from_str("now"))
            && let Some(now) = now.dyn_ref::<Function>()
            && let Ok(ms) = now.call0(&perf)
            && let Some(ms) = ms.as_f64()
            && ms.is_finite()
        {
            return ms;
        }
        js_sys::Date::now()
    }

    /// The S30 keys in canonical order.
    const SECTION_KEYS: [&str; 16] = [
        "version",
        "algorithm",
        "vertices",
        "triangles",
        "clusters",
        "nodes",
        "rootNode",
        "hierarchyDepth",
        "levels",
        "quantOrigin",
        "quantStep",
        "uvOrigin",
        "uvStep",
        "boundsCenter",
        "boundsRadius",
        "sourceTriangles",
    ];

    /// The keys of `ClusterBakeOptions` (S31).
    const OPTION_KEYS: [&str; 4] = [
        "maxClusterTriangles",
        "maxClusterVertices",
        "minReduction",
        "targetRatio",
    ];

    /// The own property `key` of a JS value; undefined when it is absent.
    fn prop(value: &JsValue, key: &str) -> JsValue {
        Reflect::get(value, &JsValue::from_str(key)).unwrap_or(JsValue::UNDEFINED)
    }

    /// The own enumerable keys of a JS object, in order.
    fn own_keys(value: &JsValue) -> Vec<String> {
        Object::keys(value.unchecked_ref::<Object>())
            .iter()
            .filter_map(|key| key.as_string())
            .collect()
    }

    /// An object that is not an array; a typed array is refused by its keys.
    fn plain(value: &JsValue) -> bool {
        value.is_object() && !Array::is_array(value)
    }

    /// The field prefix of a section value; `section` names the whole value.
    fn subject(at: &str) -> &str {
        if at.is_empty() {
            "section"
        } else {
            at.trim_end_matches('.')
        }
    }

    /// The own property `key` as a `T` (a typed array), or a refusal naming `<at><key>`.
    fn typed<T: JsCast>(value: &JsValue, key: &str, at: &str, what: &str) -> Result<T, RaaError> {
        prop(value, key)
            .dyn_into::<T>()
            .map_err(|_| wire_refusal(at, key, &format!("must be {what}")))
    }

    /// As [`typed`] for an optional key: undefined is absent.
    fn optional<T: JsCast>(value: &JsValue, key: &str, what: &str) -> Result<Option<T>, RaaError> {
        if prop(value, key).is_undefined() {
            return Ok(None);
        }
        typed(value, key, "", what).map(Some)
    }

    /// A JS number, or a refusal naming `field`.
    fn as_number(value: &JsValue, field: &str) -> Result<f64, RaaError> {
        value
            .as_f64()
            .ok_or_else(|| vg_invalid(field, "must be a number"))
    }

    /// The number at `<at><key>`.
    fn number(value: &JsValue, key: &str, at: &str) -> Result<f64, RaaError> {
        as_number(&prop(value, key), &format!("{at}{key}"))
    }

    /// The array at `<at><key>` of exactly N numbers.
    fn numbers<const N: usize>(value: &JsValue, key: &str, at: &str) -> Result<[f64; N], RaaError> {
        let refusal = || wire_refusal(at, key, &format!("must be an array of {N} numbers"));
        let array = prop(value, key)
            .dyn_into::<Array>()
            .ok()
            .filter(|array| array.length() as usize == N)
            .ok_or_else(refusal)?;
        let mut out = [0.0; N];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = array.get(i as u32).as_f64().ok_or_else(refusal)?;
        }
        Ok(out)
    }

    /// Exactly the keys of `keys`: an extra key or a missing one is refused as `<at><key>`.
    fn exact_keys(value: &JsValue, keys: &[&str], at: &str) -> Result<(), RaaError> {
        let own = own_keys(value);
        if let Some(extra) = own.iter().find(|key| !keys.contains(&key.as_str())) {
            return Err(vg_invalid(
                &format!("{at}{extra}"),
                "is not a field of this object",
            ));
        }
        match keys.iter().find(|&&key| !own.iter().any(|o| o == key)) {
            Some(missing) => Err(vg_invalid(&format!("{at}{missing}"), "is missing")),
            None => Ok(()),
        }
    }

    /// The S30 section of a JS value: the plain object of rule 1 with its typed arrays, then the
    /// wire rules and the complete validation of [`section_from_wire`]. Every refusal is
    /// `vg.invalid` with its field prefixed by `at` (empty, or `sections[i].`).
    pub(crate) fn section_from_js(value: &JsValue, at: &str) -> Result<ClusterMesh, RaaError> {
        if !plain(value) {
            return Err(vg_invalid(subject(at), "must be a ClusterSection object"));
        }
        exact_keys(value, &SECTION_KEYS, at)?;
        let vertices = typed::<Uint16Array>(value, "vertices", at, "a Uint16Array")?.to_vec();
        let triangles = typed::<Uint32Array>(value, "triangles", at, "a Uint32Array")?.to_vec();
        let clusters = typed::<Uint8Array>(value, "clusters", at, "a Uint8Array")?.to_vec();
        let nodes = typed::<Uint8Array>(value, "nodes", at, "a Uint8Array")?.to_vec();
        let wire = SectionWire {
            version: number(value, "version", at)?,
            algorithm: number(value, "algorithm", at)?,
            vertices: &vertices,
            triangles: &triangles,
            clusters: &clusters,
            nodes: &nodes,
            root_node: number(value, "rootNode", at)?,
            hierarchy_depth: number(value, "hierarchyDepth", at)?,
            levels: number(value, "levels", at)?,
            quant_origin: numbers(value, "quantOrigin", at)?,
            quant_step: number(value, "quantStep", at)?,
            uv_origin: numbers(value, "uvOrigin", at)?,
            uv_step: numbers(value, "uvStep", at)?,
            bounds_center: numbers(value, "boundsCenter", at)?,
            bounds_radius: number(value, "boundsRadius", at)?,
            source_triangles: number(value, "sourceTriangles", at)?,
        };
        section_from_wire(&wire, at)
    }

    /// The S30 section object of a mesh, keys in canonical order; `clusters` and `nodes` are the
    /// bytes of `ClusterMesh::to_section_bytes`.
    pub(crate) fn section_to_js(mesh: &ClusterMesh) -> JsValue {
        let (clusters, nodes) = mesh.to_section_bytes();
        let vertices: Vec<u16> = mesh.vertices.iter().flat_map(|v| v.data).collect();
        object(vec![
            ("version", JsValue::from_f64(1.0)),
            ("algorithm", JsValue::from_f64(1.0)),
            ("vertices", Uint16Array::from(vertices.as_slice()).into()),
            (
                "triangles",
                Uint32Array::from(mesh.triangles.as_slice()).into(),
            ),
            ("clusters", Uint8Array::from(clusters.as_slice()).into()),
            ("nodes", Uint8Array::from(nodes.as_slice()).into()),
            ("rootNode", JsValue::from_f64(f64::from(mesh.root_node))),
            (
                "hierarchyDepth",
                JsValue::from_f64(f64::from(mesh.hierarchy_depth)),
            ),
            ("levels", JsValue::from_f64(f64::from(mesh.levels))),
            ("quantOrigin", numbers_js(&mesh.quant_origin)),
            ("quantStep", JsValue::from_f64(f64::from(mesh.quant_step))),
            ("uvOrigin", numbers_js(&mesh.uv_origin)),
            ("uvStep", numbers_js(&mesh.uv_step)),
            ("boundsCenter", numbers_js(&mesh.bounds_center)),
            (
                "boundsRadius",
                JsValue::from_f64(f64::from(mesh.bounds_radius)),
            ),
            (
                "sourceTriangles",
                JsValue::from_f64(mesh.source_triangles as f64),
            ),
        ])
    }

    /// A JS array of the f32 components.
    fn numbers_js<const N: usize>(values: &[f32; N]) -> JsValue {
        values
            .iter()
            .map(|&v| JsValue::from_f64(f64::from(v)))
            .collect::<Array>()
            .into()
    }

    /// The components of each `[f32; N]`, in order.
    fn flat<const N: usize>(values: &[[f32; N]]) -> Vec<f32> {
        values.iter().flatten().copied().collect()
    }

    /// A plain object of the entries, in order.
    fn object(entries: Vec<(&str, JsValue)>) -> JsValue {
        let out = Object::new();
        for (key, value) in entries {
            let _ = Reflect::set(&out, &JsValue::from_str(key), &value);
        }
        out.into()
    }

    /// The S31 progress object of a bake snapshot.
    fn progress_js(progress: &BakeProgress) -> JsValue {
        object(vec![
            ("stage", JsValue::from_str(stage_name(progress.stage))),
            ("level", JsValue::from_f64(f64::from(progress.level))),
            ("fraction", JsValue::from_f64(f64::from(progress.fraction))),
            (
                "done",
                JsValue::from_bool(progress.stage == BakeStage::Done),
            ),
        ])
    }

    /// The bake options and the arrays of a `ClusterBakeInput` (S31). Undefined `options` is an
    /// empty patch; see [`read_options`].
    fn read_input(value: &JsValue) -> Result<(OwnedClusterInput, BuildOptions), RaaError> {
        if !plain(value) {
            return Err(vg_invalid("input", "must be a ClusterBakeInput object"));
        }
        let positions = typed::<Float32Array>(value, "positions", "", "a Float32Array")?.to_vec();
        let normals = typed::<Float32Array>(value, "normals", "", "a Float32Array")?.to_vec();
        let uvs = optional::<Float32Array>(value, "uvs", "a Float32Array")?.map(|v| v.to_vec());
        let indices = typed::<Uint32Array>(value, "indices", "", "a Uint32Array")?.to_vec();
        let options = build_options(&read_options(&prop(value, "options"))?)?;
        let input = bake_input(&positions, &normals, uvs.as_deref(), &indices)?;
        Ok((input, options))
    }

    /// The options patch of `ClusterBakeOptions` (S31): undefined is empty; an unknown key, a null
    /// or a value that is not a number is `vg.invalid` naming `options.<key>`.
    pub(crate) fn read_options(value: &JsValue) -> Result<OptionsPatch, RaaError> {
        if value.is_undefined() {
            return Ok(OptionsPatch::default());
        }
        if !plain(value) {
            return Err(vg_invalid("options", "must be an object"));
        }
        if let Some(key) = own_keys(value)
            .iter()
            .find(|key| !OPTION_KEYS.contains(&key.as_str()))
        {
            return Err(vg_invalid(&format!("options.{key}"), "is not an option"));
        }
        let read = |key: &str| match prop(value, key) {
            v if v.is_undefined() => Ok(None),
            v => as_number(&v, &format!("options.{key}")).map(Some),
        };
        Ok(OptionsPatch {
            max_cluster_triangles: read("maxClusterTriangles")?,
            max_cluster_vertices: read("maxClusterVertices")?,
            min_reduction: read("minReduction")?,
            target_ratio: read("targetRatio")?,
        })
    }

    /// The `budgetMs` of `step`: undefined is the default; else finite and above zero.
    fn budget_of(value: &JsValue) -> Result<f64, RaaError> {
        let ms = if value.is_undefined() {
            DEFAULT_BAKE_BUDGET_MS
        } else {
            as_number(value, "budgetMs")?
        };
        if ms.is_finite() && ms > 0.0 {
            Ok(ms)
        } else {
            let reason = format!("must be finite and above zero, got {ms}");
            Err(vg_invalid("budgetMs", reason))
        }
    }

    /// A refusal as the JS `RaaError` (an `Error` named `RaaError` with its `code`).
    fn to_js<T>(result: Result<T, RaaError>) -> Result<T, JsValue> {
        result.map_err(|e| error_to_js(&e))
    }

    fn bake_section(input: &JsValue) -> Result<JsValue, RaaError> {
        let (owned, options) = read_input(input)?;
        let mesh = raa_cluster::bake(owned.as_input(), &options).map_err(|e| cluster_error(&e))?;
        Ok(section_to_js(&mesh))
    }

    fn cut_object(section: &JsValue, error_metres: &JsValue) -> Result<JsValue, RaaError> {
        let error = as_number(error_metres, "errorMetres")?;
        if !(error.is_finite() && error >= 0.0) {
            let reason = format!("must be finite and not negative, got {error}");
            return Err(vg_invalid("errorMetres", reason));
        }
        let mesh = section_from_js(section, "")?;
        let clusters = select_cut(&mesh, error as f32);
        let cut = cut_triangles(&mesh, &clusters);
        Ok(object(vec![
            (
                "positions",
                Float32Array::from(flat(&cut.positions).as_slice()).into(),
            ),
            (
                "normals",
                Float32Array::from(flat(&cut.normals).as_slice()).into(),
            ),
            ("uvs", Float32Array::from(flat(&cut.uvs).as_slice()).into()),
            ("indices", Uint32Array::from(cut.indices.as_slice()).into()),
            (
                "clusterOfTriangle",
                Uint32Array::from(cut.cluster_of_triangle.as_slice()).into(),
            ),
            ("clusters", JsValue::from_f64(clusters.len() as f64)),
            (
                "triangles",
                JsValue::from_f64((cut.indices.len() / 3) as f64),
            ),
        ]))
    }

    /// `bakeClusters(input)`: bakes one single-material mesh to the end (S31).
    #[wasm_bindgen(js_name = bakeClusters)]
    pub fn bake_clusters(input: JsValue) -> Result<JsValue, JsValue> {
        to_js(bake_section(&input))
    }

    /// `validateClusters(section)`: the complete structural validation of an S30 section.
    #[wasm_bindgen(js_name = validateClusters)]
    pub fn validate_clusters(section: JsValue) -> Result<(), JsValue> {
        to_js(section_from_js(&section, "").map(|_| ()))
    }

    /// `clusterCutMesh(section, errorMetres)`: the uniform cut at `errorMetres` >= 0 (S29, S31).
    #[wasm_bindgen(js_name = clusterCutMesh)]
    pub fn cluster_cut_mesh(section: JsValue, error_metres: JsValue) -> Result<JsValue, JsValue> {
        to_js(cut_object(&section, &error_metres))
    }

    /// `new ClusterBake(input)` (S31): a sliced bake, stepped within time budgets.
    #[wasm_bindgen]
    pub struct ClusterBake {
        bake: Option<SlicedBake>,
        progress: BakeProgress,
        outcome: Option<Result<ClusterMesh, ClusterError>>,
    }

    #[wasm_bindgen]
    impl ClusterBake {
        /// Copies and validates the input; the refusals are those of `bakeClusters`.
        #[wasm_bindgen(constructor)]
        pub fn new(input: JsValue) -> Result<ClusterBake, JsValue> {
            to_js(ClusterBake::start(&input))
        }

        /// Unit steps until `budgetMs` (default 4) has passed or the bake is done; at least one
        /// unit runs. A bake error is returned here and by [`ClusterBake::result`].
        pub fn step(&mut self, budget_ms: JsValue) -> Result<JsValue, JsValue> {
            let budget = budget_of(&budget_ms);
            to_js(
                budget
                    .and_then(|ms| self.advance(ms))
                    .map(|()| progress_js(&self.progress)),
            )
        }

        /// The stage, level and fraction after the last step.
        pub fn progress(&self) -> JsValue {
            progress_js(&self.progress)
        }

        /// The baked section; `vg.invalid` 'the bake is not done' before done.
        pub fn result(&self) -> Result<JsValue, JsValue> {
            to_js(self.section())
        }
    }

    impl ClusterBake {
        fn start(input: &JsValue) -> Result<ClusterBake, RaaError> {
            let (owned, options) = read_input(input)?;
            let bake =
                SlicedBake::new(owned.as_input(), &options).map_err(|e| cluster_error(&e))?;
            Ok(ClusterBake {
                progress: bake.progress(),
                bake: Some(bake),
                outcome: None,
            })
        }

        /// Runs units while the clock is below the deadline; a finished bake is finished here
        /// and its outcome kept. An error outcome is returned.
        fn advance(&mut self, budget_ms: f64) -> Result<(), RaaError> {
            if let Some(Err(e)) = &self.outcome {
                return Err(cluster_error(e));
            }
            if let Some(bake) = self.bake.as_mut() {
                let deadline = now_ms() + budget_ms;
                while bake.step(WorkBudget { units: 1 }) == BakeStatus::Working
                    && now_ms() < deadline
                {}
                self.progress = bake.progress();
            }
            if self.progress.stage == BakeStage::Done
                && let Some(bake) = self.bake.take()
            {
                self.outcome = Some(bake.finish());
            }
            match &self.outcome {
                Some(Err(e)) => Err(cluster_error(e)),
                _ => Ok(()),
            }
        }

        fn section(&self) -> Result<JsValue, RaaError> {
            match &self.outcome {
                Some(Ok(mesh)) => Ok(section_to_js(mesh)),
                Some(Err(e)) => Err(cluster_error(e)),
                None => Err(vg_invalid("result", "the bake is not done")),
            }
        }
    }

    /// The shared cluster store of the feature map.
    fn cluster_store(st: &EngineState) -> Result<Rc<RefCell<ClusterStore>>, RaaError> {
        st.features
            .get::<Rc<RefCell<ClusterStore>>>()
            .cloned()
            .ok_or_else(|| vg_invalid("store", "the cluster store is not installed"))
    }

    /// The retained bake inputs of a handle: an absent entry is `vg.invalid` naming `mesh`
    /// ('no retained geometry').
    fn retained_source<'a>(
        sources: Option<&'a MeshSources>,
        mesh: u32,
    ) -> Result<&'a crate::state::MeshSource, RaaError> {
        sources
            .and_then(|sources| sources.get(mesh))
            .ok_or_else(|| vg_invalid("mesh", "no retained geometry"))
    }

    /// `attachMeshClusters(mesh, sections)`: one section per sub-mesh, all converted and
    /// validated, then the mesh's clusters are replaced atomically (state ready, attached).
    fn attach_mesh(st: &mut EngineState, mesh: u32, sections: &JsValue) -> Result<(), RaaError> {
        st.mesh(mesh)?;
        let (expected, has_colors) = {
            let source = retained_source(st.features.get::<MeshSources>(), mesh)?;
            (source.submeshes.len(), source.has_colors)
        };
        check_vertex_colors(has_colors, "mesh")?;
        let array = sections
            .dyn_ref::<Array>()
            .ok_or_else(|| vg_invalid("sections", "must be an Array of ClusterSection objects"))?;
        let results = (0..array.length())
            .map(|i| section_from_js(&array.get(i), &format!("sections[{i}].")))
            .collect::<Vec<_>>();
        let sections = plan_attach(expected, results)?;
        cluster_store(st)?.borrow_mut().attach(mesh, sections);
        Ok(())
    }

    /// `bakeMeshClusters(mesh, options)`: starts the sliced runtime bake of every sub-mesh.
    fn bake_mesh(st: &mut EngineState, mesh: u32, options: &JsValue) -> Result<(), RaaError> {
        st.mesh(mesh)?;
        let has_colors = retained_source(st.features.get::<MeshSources>(), mesh)?.has_colors;
        check_vertex_colors(has_colors, "mesh")?;
        let options = build_options(&read_options(options)?)?;
        let inputs = retained_source(st.features.get::<MeshSources>(), mesh)?
            .cluster_inputs()
            .map_err(|e| cluster_error(&e))?;
        cluster_store(st)?
            .borrow_mut()
            .start_bake(mesh, inputs, &options)
            .map_err(|e| cluster_error(&e))
    }

    /// `clearMeshClusters(mesh)`: cancels a bake and drops the clusters of the mesh.
    fn clear_mesh(st: &mut EngineState, mesh: u32) -> Result<(), RaaError> {
        st.mesh(mesh)?;
        cluster_store(st)?.borrow_mut().clear(mesh);
        Ok(())
    }

    /// `meshClusters(mesh)`: the current state as a fresh `MeshClusterState` object.
    fn mesh_state(st: &EngineState, mesh: u32) -> Result<JsValue, RaaError> {
        st.mesh(mesh)?;
        let store = cluster_store(st)?;
        let store = store.borrow();
        let status = store.state(mesh);
        let report = cluster_report(&status, store.sections(mesh));
        Ok(report_to_js(&report))
    }

    /// The `MeshClusterState` object: an absent source, code or message is `null`.
    fn report_to_js(report: &ClusterReport) -> JsValue {
        let optional = |value: Option<&str>| value.map_or(JsValue::NULL, JsValue::from_str);
        object(vec![
            ("state", JsValue::from_str(report.state)),
            ("progress", JsValue::from_f64(f64::from(report.progress))),
            ("source", optional(report.source)),
            ("code", optional(report.code)),
            ("message", optional(report.message.as_deref())),
            ("sections", JsValue::from_f64(f64::from(report.sections))),
            ("clusters", JsValue::from_f64(f64::from(report.clusters))),
            (
                "sourceTriangles",
                JsValue::from_f64(report.source_triangles as f64),
            ),
            ("levels", JsValue::from_f64(f64::from(report.levels))),
        ])
    }

    /// `setClusterBakeBudget(ms)`: finite and inside [0.5, 100] milliseconds.
    fn set_budget(st: &mut EngineState, ms: &JsValue) -> Result<(), RaaError> {
        let ms = check_bake_budget(as_number(ms, "ms")?)?;
        if let Some(budget) = st.features.get_mut::<ClusterBakeBudget>() {
            budget.0 = ms;
        }
        Ok(())
    }

    /// The current bake budget in milliseconds per Engine render invocation.
    fn budget_ms(st: &EngineState) -> f64 {
        st.features
            .get::<ClusterBakeBudget>()
            .map_or(DEFAULT_BAKE_BUDGET_MS, |budget| budget.0)
    }

    #[wasm_bindgen(js_class = Engine)]
    impl Engine {
        /// Attaches one cluster section per sub-mesh of `mesh`, in sub-mesh order. All sections
        /// are validated before the mesh's clusters are replaced atomically (state ready,
        /// source attached); a refusal leaves them unchanged. An unknown handle throws
        /// `api.unknown_handle`, a colour-stream mesh `vg.mesh_vertex_colors` and a wrong
        /// section count or content `vg.invalid`.
        #[wasm_bindgen(js_name = attachMeshClusters)]
        pub fn attach_mesh_clusters(
            &mut self,
            mesh: u32,
            sections: JsValue,
        ) -> Result<(), JsValue> {
            self.with_state(|_, st| attach_mesh(st, mesh, &sections))
        }

        /// Starts a sliced runtime bake of every sub-mesh of `mesh`, spending at most
        /// `clusterBakeBudget()` milliseconds per Engine render invocation. The state becomes
        /// `baking` with `progress` in [0, 1), then `ready` with source runtime.
        #[wasm_bindgen(js_name = bakeMeshClusters)]
        pub fn bake_mesh_clusters(&mut self, mesh: u32, options: JsValue) -> Result<(), JsValue> {
            self.with_state(|_, st| bake_mesh(st, mesh, &options))
        }

        /// Cancels a bake of `mesh` and drops its clusters: state none, source null.
        #[wasm_bindgen(js_name = clearMeshClusters)]
        pub fn clear_mesh_clusters(&mut self, mesh: u32) -> Result<(), JsValue> {
            self.with_state(|_, st| clear_mesh(st, mesh))
        }

        /// Returns a fresh `MeshClusterState` of `mesh` (state, progress, source, code,
        /// message and the ready section, cluster, source-triangle and level counts).
        #[wasm_bindgen(js_name = meshClusters)]
        pub fn mesh_clusters(&self, mesh: u32) -> Result<JsValue, JsValue> {
            self.with_state(|_, st| mesh_state(st, mesh))
        }

        /// Sets the CPU budget of the runtime cluster bake per Engine render invocation, in
        /// milliseconds: finite and inside [0.5, 100], else `vg.invalid` naming `ms`.
        #[wasm_bindgen(js_name = setClusterBakeBudget)]
        pub fn set_cluster_bake_budget(&mut self, ms: JsValue) -> Result<(), JsValue> {
            self.with_state(|_, st| set_budget(st, &ms))
        }

        /// The runtime cluster-bake budget in milliseconds per Engine render invocation
        /// (default 4).
        #[wasm_bindgen(js_name = clusterBakeBudget)]
        pub fn cluster_bake_budget(&self) -> Result<f64, JsValue> {
            self.with_state(|_, st| Ok(budget_ms(st)))
        }
    }
}
