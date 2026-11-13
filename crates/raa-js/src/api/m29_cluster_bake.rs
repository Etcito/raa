//! m29 cluster bake, share 1: the pure helpers of the cluster-bake API (S31) and the wasm32
//! runtime of the shared cluster store (`install`, `frame`).
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
use raa_cluster::{BakeStage, BuildOptions, ClusterError, OwnedClusterInput};
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
