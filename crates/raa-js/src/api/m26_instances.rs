//! m26 instances API: per-instance detail chains (`setMeshLods`), the two-sided geometric
//! error of a level (`meshLodError`), the level bias of one instance
//! (`setInstanceLodBias`), removing a chain (`clearMeshLods`), the instance ray query
//! (`queryInstancesRay`), the detail options (`setLodOptions`) and the frame statistics of
//! the lit renderer (`instanceStats`).
//!
//! A level mesh pairs its sub-mesh k with sub-mesh k of the full mesh. Refusals:
//! `lod.unknown_mesh`, `lod.too_many_levels`, `lod.invalid` (raa-lit's `render.invalid_lod`
//! included), `api.invalid_argument` for a malformed argument, `api.unknown_handle` for a
//! stale instance. A posed instance always draws its full mesh (raa-lit rule).
//! The pure part below compiles on the host (tests/instances.rs includes this file).

use std::collections::HashMap;

use raa_core::RaaError;
use raa_geom::MeshData;
use raa_lit::{LodParams, MAX_LOD_LEVELS};

const INVALID: &str = "api.invalid_argument";
const UNKNOWN_MESH: &str = "lod.unknown_mesh";
const TOO_MANY_LEVELS: &str = "lod.too_many_levels";
const LOD_INVALID: &str = "lod.invalid";

const DEFAULT_RAY_HITS: usize = 64;
/// Largest `max` of `queryInstancesRay`, the default instance cap of the engine.
pub(crate) const MAX_RAY_HITS: usize = 16384;
const MAX_BIAS: i8 = 3;
const MAX_HYSTERESIS: f64 = 0.9;

fn refusal(code: &'static str, message: impl Into<String>) -> RaaError {
    RaaError::FrameGraph {
        code,
        message: message.into(),
    }
}

/// A raa-lit refusal under the JS `lod.invalid` code.
pub(crate) fn lod_refusal(e: RaaError) -> RaaError {
    match e {
        RaaError::FrameGraph { message, .. } => refusal(LOD_INVALID, message),
        other => other,
    }
}

/// The bias argument of `setInstanceLodBias`: a whole number from -3 to 3.
pub(crate) fn bias_arg(v: Option<f64>) -> Result<i8, RaaError> {
    let v = v.ok_or_else(|| refusal(INVALID, "'bias' is required"))?;
    if v.fract() != 0.0 || !(-f64::from(MAX_BIAS)..=f64::from(MAX_BIAS)).contains(&v) {
        return Err(refusal(
            INVALID,
            format!("'bias' must be a whole number from -3 to 3, got {v}"),
        ));
    }
    Ok(v as i8)
}

/// The `max` argument of `queryInstancesRay`: a whole number from 1 to `MAX_RAY_HITS`,
/// 64 when absent.
pub(crate) fn max_arg(v: Option<f64>) -> Result<usize, RaaError> {
    let Some(v) = v else {
        return Ok(DEFAULT_RAY_HITS);
    };
    if v.fract() != 0.0 || !(1.0..=MAX_RAY_HITS as f64).contains(&v) {
        return Err(refusal(
            INVALID,
            format!("'max' must be a whole number from 1 to {MAX_RAY_HITS}, got {v}"),
        ));
    }
    Ok(v as usize)
}

/// A handle number: a whole number in u32 range.
pub(crate) fn handle_arg(name: &str, v: f64) -> Result<u32, RaaError> {
    if v.fract() == 0.0 && (0.0..=f64::from(u32::MAX)).contains(&v) {
        Ok(v as u32)
    } else {
        Err(refusal(
            INVALID,
            format!("'{name}' must be a handle number, got {v}"),
        ))
    }
}

/// A ray origin or direction: three finite numbers that stay finite as f32.
pub(crate) fn vec3_arg(name: &str, v: &[f64]) -> Result<[f32; 3], RaaError> {
    let bad = || refusal(INVALID, format!("'{name}' must be three finite numbers"));
    if v.len() != 3 || v.iter().any(|x| !x.is_finite()) {
        return Err(bad());
    }
    let out = [v[0] as f32, v[1] as f32, v[2] as f32];
    if out.iter().all(|x| x.is_finite()) {
        Ok(out)
    } else {
        Err(bad())
    }
}

/// A finite bias value that stays finite as f32.
fn finite_bias(name: &str, v: f64) -> Result<f32, RaaError> {
    let v32 = v as f32;
    if v.is_finite() && v32.is_finite() {
        Ok(v32)
    } else {
        Err(refusal(
            INVALID,
            format!("'{name}' must be a finite number, got {v}"),
        ))
    }
}

/// The `setLodOptions` settings over the current ones. Camera parameters are
/// `{threshold_px: thresholdPx, hysteresis, bias}`; cascade parameters are
/// `{threshold_px: 1.0 (one shadow-map texel), hysteresis, bias: shadowBias}`. Absent keys
/// keep their current value.
pub(crate) fn lod_options(
    camera: LodParams,
    shadow: LodParams,
    threshold_px: Option<f64>,
    hysteresis: Option<f64>,
    bias: Option<f64>,
    shadow_bias: Option<f64>,
) -> Result<(LodParams, LodParams), RaaError> {
    let mut c = camera;
    let mut s = shadow;
    if let Some(e) = threshold_px {
        let e32 = e as f32;
        if !(e > 0.0 && e32.is_finite()) {
            return Err(refusal(
                INVALID,
                format!("'thresholdPx' must be a positive number, got {e}"),
            ));
        }
        c.threshold_px = e32;
    }
    if let Some(h) = hysteresis {
        if !(h.is_finite() && (0.0..=MAX_HYSTERESIS).contains(&h)) {
            return Err(refusal(
                INVALID,
                format!("'hysteresis' must be a number from 0 to 0.9, got {h}"),
            ));
        }
        c.hysteresis = h as f32;
        s.hysteresis = h as f32;
    }
    if let Some(b) = bias {
        c.bias = finite_bias("bias", b)?;
    }
    if let Some(b) = shadow_bias {
        s.bias = finite_bias("shadowBias", b)?;
    }
    // Shadows always count one shadow-map texel; thresholdPx is the camera threshold only.
    s.threshold_px = 1.0;
    Ok((c, s))
}

/// The number of coarser levels of a chain: at least one, and with the full mesh at most
/// `MAX_LOD_LEVELS` entries.
pub(crate) fn level_count(levels: usize) -> Result<(), RaaError> {
    if levels == 0 {
        return Err(refusal(
            LOD_INVALID,
            "a detail chain needs at least one coarser level",
        ));
    }
    if levels + 1 > MAX_LOD_LEVELS {
        return Err(refusal(
            TOO_MANY_LEVELS,
            format!(
                "a detail chain holds at most {} coarser levels, got {levels}",
                MAX_LOD_LEVELS - 1
            ),
        ));
    }
    Ok(())
}

/// A level must have as many sub-meshes as the full mesh.
pub(crate) fn same_submesh_count(full: usize, level: usize) -> Result<(), RaaError> {
    if full == level {
        Ok(())
    } else {
        Err(refusal(
            LOD_INVALID,
            format!("a level has {level} sub-meshes, the full mesh has {full}"),
        ))
    }
}

/// Flat xyz positions as points.
pub(crate) fn positions(flat: &[f32]) -> Result<Vec<[f32; 3]>, RaaError> {
    if !flat.len().is_multiple_of(3) {
        return Err(refusal(
            INVALID,
            "'positions' must hold a whole number of xyz triples",
        ));
    }
    Ok(flat.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect())
}

/// The geometry `lod_error` reads: positions and triangle indices.
pub(crate) fn mesh_data(flat: &[f32], indices: &[u32]) -> Result<MeshData, RaaError> {
    Ok(MeshData {
        positions: positions(flat)?,
        normals: Vec::new(),
        indices: indices.to_vec(),
    })
}

/// The ray hits as `(handle, t)`: the smallest t of each handle, sorted by t then handle,
/// at most `max` of them.
pub(crate) fn nearest_hits(hits: &[(u32, f32)], max: usize) -> Vec<(u32, f32)> {
    let mut best: HashMap<u32, f32> = HashMap::new();
    for &(handle, t) in hits {
        best.entry(handle)
            .and_modify(|b| *b = b.min(t))
            .or_insert(t);
    }
    let mut out: Vec<(u32, f32)> = best.into_iter().collect();
    out.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    out.truncate(max);
    out
}

#[cfg(target_arch = "wasm32")]
pub(crate) use web::{frame, install};

#[cfg(target_arch = "wasm32")]
mod web {
    use super::*;
    use crate::engine::{Engine, InstallCtx};
    use crate::state::EngineState;
    use js_sys::{Array, Float32Array, Object, Reflect, Uint32Array};
    use raa_core::{GpuContext, MeshId};
    use raa_lit::LodLevel;
    use raa_math::Vec3;
    use raa_web::FrameInput;
    use wasm_bindgen::{JsCast, prelude::*};

    pub(crate) fn install(_ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
        Ok(())
    }

    pub(crate) fn frame(
        _st: &mut EngineState,
        _gpu: &GpuContext,
        _input: &FrameInput,
    ) -> Result<(), RaaError> {
        Ok(())
    }

    /// The value at `key`, or None when it is missing, undefined or null.
    fn field(obj: &JsValue, key: &str) -> Result<Option<JsValue>, RaaError> {
        if !obj.is_object() {
            return Err(refusal(
                INVALID,
                format!("expected an object holding '{key}'"),
            ));
        }
        let v = Reflect::get(obj, &JsValue::from_str(key))
            .map_err(|_| refusal(INVALID, format!("cannot read '{key}'")))?;
        Ok((!v.is_undefined() && !v.is_null()).then_some(v))
    }

    /// An optional number argument: undefined and null are absent.
    fn optional(v: &JsValue, name: &str) -> Result<Option<f64>, RaaError> {
        if v.is_undefined() || v.is_null() {
            return Ok(None);
        }
        v.as_f64()
            .map(Some)
            .ok_or_else(|| refusal(INVALID, format!("'{name}' must be a number")))
    }

    fn number(obj: &JsValue, key: &str) -> Result<Option<f64>, RaaError> {
        field(obj, key)?.map_or(Ok(None), |v| optional(&v, key))
    }

    /// An array of numbers, or a Float32Array.
    fn numbers(v: &JsValue, name: &str) -> Result<Vec<f64>, RaaError> {
        if let Some(f) = v.dyn_ref::<Float32Array>() {
            return Ok(f.to_vec().into_iter().map(f64::from).collect());
        }
        if !Array::is_array(v) {
            return Err(refusal(
                INVALID,
                format!("'{name}' must be an array of numbers"),
            ));
        }
        let a: &Array = v.unchecked_ref();
        (0..a.length())
            .map(|i| {
                a.get(i)
                    .as_f64()
                    .ok_or_else(|| refusal(INVALID, format!("'{name}' must hold numbers")))
            })
            .collect()
    }

    /// The geometry of a `{positions: Float32Array, indices: Uint32Array}` object.
    fn mesh_data_of(desc: &JsValue) -> Result<MeshData, RaaError> {
        let pos =
            field(desc, "positions")?.ok_or_else(|| refusal(INVALID, "'positions' is required"))?;
        let idx =
            field(desc, "indices")?.ok_or_else(|| refusal(INVALID, "'indices' is required"))?;
        let pos = pos
            .dyn_ref::<Float32Array>()
            .ok_or_else(|| refusal(INVALID, "'positions' must be a Float32Array"))?
            .to_vec();
        let idx = idx
            .dyn_ref::<Uint32Array>()
            .ok_or_else(|| refusal(INVALID, "'indices' must be a Uint32Array"))?
            .to_vec();
        mesh_data(&pos, &idx)
    }

    fn lod_error_of(full: &JsValue, level: &JsValue) -> Result<f64, RaaError> {
        let full = mesh_data_of(full)?;
        let level = mesh_data_of(level)?;
        raa_lit::lod_error(&full, &level)
            .map(f64::from)
            .map_err(lod_refusal)
    }

    /// The `(mesh handle, error)` entries of `setMeshLods`' `levels` array.
    fn level_entries(levels: &JsValue) -> Result<Vec<(u32, f64)>, RaaError> {
        if !Array::is_array(levels) {
            return Err(refusal(
                INVALID,
                "'levels' must be an Array of {mesh, error}",
            ));
        }
        let a: &Array = levels.unchecked_ref();
        let mut out = Vec::with_capacity(a.length() as usize);
        for i in 0..a.length() {
            let item = a.get(i);
            let mesh = number(&item, "mesh")?
                .ok_or_else(|| refusal(INVALID, format!("'levels[{i}].mesh' is required")))?;
            let error = number(&item, "error")?
                .ok_or_else(|| refusal(INVALID, format!("'levels[{i}].error' is required")))?;
            out.push((handle_arg("levels[].mesh", mesh)?, error));
        }
        Ok(out)
    }

    /// The sub-mesh ids of mesh handle `mesh`, as addMeshInstance resolves them.
    fn submeshes(st: &EngineState, mesh: u32) -> Result<Vec<MeshId>, RaaError> {
        let unknown = || refusal(UNKNOWN_MESH, format!("mesh {mesh} does not exist"));
        let lit = st.mesh(mesh).map_err(|_| unknown())?.lit;
        st.lit
            .submesh_ids(lit)
            .map(|ids| ids.to_vec())
            .ok_or_else(unknown)
    }

    fn set(obj: &Object, key: &str, value: &JsValue) {
        let _ = Reflect::set(obj, &JsValue::from_str(key), value);
    }

    fn u32_array(values: &[u32]) -> JsValue {
        values
            .iter()
            .map(|&x| JsValue::from(x))
            .collect::<Array>()
            .into()
    }

    #[wasm_bindgen(js_class = Engine)]
    impl Engine {
        /// Geometric error of `level` against `full` (`{positions, indices}` objects, the
        /// MeshDesc fields read), in object units: the two-sided vertex-to-triangle distance.
        #[wasm_bindgen(js_name = meshLodError)]
        pub fn mesh_lod_error(&self, full: JsValue, level: JsValue) -> Result<f64, JsValue> {
            self.with_state(|_, _| lod_error_of(&full, &level))
        }

        /// Registers `levels` (`[{mesh, error}]`, coarser than the full mesh, finest first)
        /// as the detail chain of mesh `mesh`: sub-mesh k of each level pairs with sub-mesh
        /// k of `mesh`. A refusal on any sub-mesh clears the earlier ones.
        #[wasm_bindgen(js_name = setMeshLods)]
        pub fn set_mesh_lods(&mut self, mesh: u32, levels: JsValue) -> Result<(), JsValue> {
            self.with_state(|_, st| {
                let entries = level_entries(&levels)?;
                level_count(entries.len())?;
                let full = submeshes(st, mesh)?;
                let mut chains = Vec::with_capacity(entries.len());
                for (handle, error) in entries {
                    let subs = submeshes(st, handle)?;
                    same_submesh_count(full.len(), subs.len())?;
                    chains.push((subs, error));
                }
                for (k, &full_k) in full.iter().enumerate() {
                    let chain: Vec<LodLevel> = chains
                        .iter()
                        .map(|(subs, error)| LodLevel {
                            mesh: subs[k],
                            error: *error as f32,
                        })
                        .collect();
                    if let Err(e) = st.lit.set_mesh_lods(full_k, &chain) {
                        for &done in &full[..k] {
                            st.lit.clear_mesh_lods(done);
                        }
                        return Err(lod_refusal(e));
                    }
                }
                Ok(())
            })
        }

        /// Sets the level bias of `instance` (whole number from -3 to 3): positive bias
        /// draws coarser levels sooner.
        #[wasm_bindgen(js_name = setInstanceLodBias)]
        pub fn set_instance_lod_bias(
            &mut self,
            instance: u32,
            bias: Option<f64>,
        ) -> Result<(), JsValue> {
            self.with_state(|_, st| {
                let bias = bias_arg(bias)?;
                let tag = st.instance_group(instance)?.tag;
                st.lit.set_instance_lod_bias(tag, bias);
                Ok(())
            })
        }

        /// Removes the detail chain of `mesh` (every sub-mesh); true when any sub-mesh had
        /// a chain.
        #[wasm_bindgen(js_name = clearMeshLods)]
        pub fn clear_mesh_lods(&mut self, mesh: u32) -> Result<bool, JsValue> {
            self.with_state(|_, st| {
                let full = submeshes(st, mesh)?;
                let mut had = false;
                for id in full {
                    had |= st.lit.clear_mesh_lods(id);
                }
                Ok(had)
            })
        }

        /// Detail options: `{thresholdPx?, hysteresis?, bias?, shadowBias?}`; absent keys
        /// keep their value. Shadows use one texel and `shadowBias`.
        #[wasm_bindgen(js_name = setLodOptions)]
        pub fn set_lod_options(&mut self, settings: JsValue) -> Result<(), JsValue> {
            self.with_state(|_, st| {
                let threshold = number(&settings, "thresholdPx")?;
                let hysteresis = number(&settings, "hysteresis")?;
                let bias = number(&settings, "bias")?;
                let shadow_bias = number(&settings, "shadowBias")?;
                let (camera, shadow) = lod_options(
                    st.lit.lod_options(),
                    st.lit.shadow_lod_options(),
                    threshold,
                    hysteresis,
                    bias,
                    shadow_bias,
                )?;
                st.lit.set_lod_options(camera);
                st.lit.set_shadow_lod_options(shadow);
                Ok(())
            })
        }

        /// The instances hit by the ray `origin` + t·`dir`, as `[{instance, t}]` with the
        /// smallest t of each instance, ascending by t then handle, at most `max` (64).
        #[wasm_bindgen(js_name = queryInstancesRay)]
        pub fn query_instances_ray(
            &self,
            origin: JsValue,
            dir: JsValue,
            max: JsValue,
        ) -> Result<JsValue, JsValue> {
            self.with_state(|_, st| {
                let o = vec3_arg("origin", &numbers(&origin, "origin")?)?;
                let d = vec3_arg("dir", &numbers(&dir, "dir")?)?;
                let max = max_arg(optional(&max, "max")?)?;
                let mut raw = Vec::new();
                st.scene.ray_instances(
                    Vec3::new(o[0], o[1], o[2]),
                    Vec3::new(d[0], d[1], d[2]),
                    f32::MAX,
                    &mut raw,
                );
                let mut hits = Vec::with_capacity(raw.len());
                for (id, t) in raw {
                    let Ok(tag) = st.scene.instance_tag(id) else {
                        continue;
                    };
                    if st
                        .instance_group(tag)
                        .is_ok_and(|group| group.instances.contains(&id))
                    {
                        hits.push((tag, t));
                    }
                }
                let out = Array::new();
                for (instance, t) in nearest_hits(&hits, max) {
                    let hit = Object::new();
                    set(&hit, "instance", &JsValue::from(instance));
                    set(&hit, "t", &JsValue::from(f64::from(t)));
                    out.push(&JsValue::from(hit));
                }
                Ok(out.into())
            })
        }

        /// `{instances, camera, cascades[4], lods[8], uploadedBytes}`: the lit renderer's
        /// frame statistics. `lods[k]` counts camera instances on level k; posed instances
        /// count on level 0.
        #[wasm_bindgen(js_name = instanceStats)]
        pub fn instance_stats(&self) -> Result<JsValue, JsValue> {
            self.with_state(|_, st| {
                let s = st.lit.frame_stats();
                let out = Object::new();
                set(&out, "instances", &JsValue::from(s.instances));
                set(&out, "camera", &JsValue::from(s.camera));
                set(&out, "cascades", &u32_array(&s.cascades));
                set(&out, "lods", &u32_array(&s.lods));
                set(
                    &out,
                    "uploadedBytes",
                    &JsValue::from(s.uploaded_bytes as f64),
                );
                Ok(out.into())
            })
        }
    }
}
