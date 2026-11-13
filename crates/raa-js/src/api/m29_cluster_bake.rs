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
//! clock of `globalThis.performance.now` (the js-sys date clock as fallback).

// The helpers of this share are consumed by the cards that follow it in this file (wasm32
// only ever reaches `install`, `frame` and the budget type), so they are dead there.
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

#[cfg(target_arch = "wasm32")]
pub(crate) use web::{frame, install};

#[cfg(target_arch = "wasm32")]
mod web {
    use super::DEFAULT_BAKE_BUDGET_MS;
    use crate::engine::InstallCtx;
    use crate::state::EngineState;
    use js_sys::{Function, Reflect};
    use raa_cluster::ClusterStore;
    use raa_core::{GpuContext, RaaError};
    use raa_web::FrameInput;
    use std::cell::RefCell;
    use std::rc::Rc;
    use wasm_bindgen::{JsCast, JsValue};

    /// Milliseconds of CPU per Engine render invocation for the runtime bake (S31).
    pub(crate) struct ClusterBakeBudget(pub f64);

    /// Inserts the shared cluster store and the default bake budget in the feature map.
    pub(crate) fn install(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
        ctx.features
            .insert(Rc::new(RefCell::new(ClusterStore::new())));
        ctx.features
            .insert(ClusterBakeBudget(DEFAULT_BAKE_BUDGET_MS));
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
}
