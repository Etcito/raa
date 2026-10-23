//! m07 mesh API (S13): attributed meshes from JS typed arrays (`addMesh`, `updateMesh`),
//! their counts, sub-meshes and bounds (`meshInfo`) and instances drawing each sub-mesh
//! with its own material (`addMeshInstance`).
//!
//! Mesh refusals keep their `mesh.*` code; an unknown sub-mesh material handle throws
//! `api.unknown_handle`; a malformed `MeshDesc` (missing array, wrong array type,
//! non-number field) throws `api.invalid_argument`.

use js_sys::{Array, Float32Array, Object, Reflect, Uint8Array, Uint32Array};
use raa_core::{GpuContext, MaterialId, RaaError};
use raa_math::Vec3;
use raa_mesh::{AttributedMesh, Colors, MeshError, SubMesh, from_js_arrays};
use raa_web::FrameInput;
use wasm_bindgen::{JsCast, prelude::*};

use crate::convert::{INVALID_ARGUMENT, UNKNOWN_HANDLE, api_error};
use crate::engine::{Engine, InstallCtx};
use crate::state::{EngineState, MeshEntry, MeshOrigin};

pub(crate) fn install(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
    let _ = ctx;
    Ok(())
}

/// Nothing per frame: the groups of updated meshes are rebuilt by the engine
/// (`EngineState::rebuild_dirty_groups`) right after the api files' `frame`.
pub(crate) fn frame(
    _st: &mut EngineState,
    _gpu: &GpuContext,
    _input: &FrameInput,
) -> Result<(), RaaError> {
    Ok(())
}

fn invalid(message: String) -> RaaError {
    api_error(INVALID_ARGUMENT, message)
}

/// A `mesh.*` refusal with its code kept.
fn mesh_error(e: &MeshError) -> RaaError {
    api_error(e.code(), e.message.clone())
}

/// The value at `key`, or None when it is missing, undefined or null.
fn field(obj: &JsValue, key: &str) -> Result<Option<JsValue>, RaaError> {
    if !obj.is_object() {
        return Err(invalid(format!(
            "expected an object holding '{key}', got a non-object"
        )));
    }
    let v = Reflect::get(obj, &JsValue::from_str(key))
        .map_err(|_| invalid(format!("cannot read '{key}'")))?;
    Ok(if v.is_undefined() || v.is_null() {
        None
    } else {
        Some(v)
    })
}

fn optional_f32s(desc: &JsValue, key: &str) -> Result<Option<Vec<f32>>, RaaError> {
    field(desc, key)?
        .map(|v| {
            v.dyn_ref::<Float32Array>()
                .map(Float32Array::to_vec)
                .ok_or_else(|| invalid(format!("'{key}' must be a Float32Array")))
        })
        .transpose()
}

fn required_f32s(desc: &JsValue, key: &str) -> Result<Vec<f32>, RaaError> {
    optional_f32s(desc, key)?.ok_or_else(|| invalid(format!("'{key}' is required")))
}

fn indices(desc: &JsValue) -> Result<Vec<u32>, RaaError> {
    let v = field(desc, "indices")?.ok_or_else(|| invalid("'indices' is required".into()))?;
    v.dyn_ref::<Uint32Array>()
        .map(Uint32Array::to_vec)
        .ok_or_else(|| invalid("'indices' must be a Uint32Array".into()))
}

/// Vertex colours: a Float32Array is linear rgba, a Uint8Array sRGB rgba8.
enum OwnedColors {
    Linear(Vec<f32>),
    Srgb8(Vec<u8>),
}

impl OwnedColors {
    fn as_colors(&self) -> Colors<'_> {
        match self {
            Self::Linear(c) => Colors::Linear(c),
            Self::Srgb8(c) => Colors::Srgb8(c),
        }
    }
}

fn colors(desc: &JsValue) -> Result<Option<OwnedColors>, RaaError> {
    let Some(v) = field(desc, "colors")? else {
        return Ok(None);
    };
    if let Some(f) = v.dyn_ref::<Float32Array>() {
        return Ok(Some(OwnedColors::Linear(f.to_vec())));
    }
    if let Some(b) = v.dyn_ref::<Uint8Array>() {
        return Ok(Some(OwnedColors::Srgb8(b.to_vec())));
    }
    Err(invalid(
        "'colors' must be a Float32Array (linear) or a Uint8Array (sRGB rgba8)".into(),
    ))
}

/// A sub-mesh index field: a number that is a non-negative integer in u32 range
/// (`mesh.submesh_range` otherwise).
fn submesh_u32(sub: &JsValue, i: u32, key: &str) -> Result<u32, RaaError> {
    let x = field(sub, key)?
        .ok_or_else(|| invalid(format!("'subMeshes[{i}].{key}' is required")))?
        .as_f64()
        .ok_or_else(|| invalid(format!("'subMeshes[{i}].{key}' must be a number")))?;
    if x.fract() == 0.0 && (0.0..=f64::from(u32::MAX)).contains(&x) {
        Ok(x as u32)
    } else {
        Err(api_error(
            "mesh.submesh_range",
            format!("subMeshes[{i}].{key} must be a non-negative integer index, got {x}"),
        ))
    }
}

/// The material behind a JS material handle number (`api.unknown_handle`).
fn material_handle(st: &EngineState, i: u32, v: &JsValue) -> Result<MaterialId, RaaError> {
    let h = v
        .as_f64()
        .ok_or_else(|| invalid(format!("'subMeshes[{i}].material' must be a number")))?;
    if h.fract() == 0.0 && (0.0..=f64::from(u32::MAX)).contains(&h) {
        st.material(h as u32)
    } else {
        Err(api_error(
            UNKNOWN_HANDLE,
            format!("subMeshes[{i}].material: unknown material handle {h}"),
        ))
    }
}

/// `subMeshes` with material handles resolved; empty when absent (one sub-mesh over all
/// indices).
fn submeshes(st: &EngineState, desc: &JsValue) -> Result<Vec<SubMesh>, RaaError> {
    let Some(v) = field(desc, "subMeshes")? else {
        return Ok(Vec::new());
    };
    if !Array::is_array(&v) {
        return Err(invalid("'subMeshes' must be an Array".into()));
    }
    let a: &Array = v.unchecked_ref();
    let mut out = Vec::with_capacity(a.length() as usize);
    for i in 0..a.length() {
        let sub = a.get(i);
        if !sub.is_object() {
            return Err(invalid(format!("'subMeshes[{i}]' must be an object")));
        }
        let material = field(&sub, "material")?
            .map(|m| material_handle(st, i, &m))
            .transpose()?;
        out.push(SubMesh {
            first_index: submesh_u32(&sub, i, "first")?,
            index_count: submesh_u32(&sub, i, "count")?,
            material,
        });
    }
    Ok(out)
}

/// Reads an S13 `MeshDesc` into a validated attributed mesh (`raa_mesh::from_js_arrays`).
fn mesh_desc(st: &EngineState, desc: &JsValue) -> Result<AttributedMesh, RaaError> {
    let positions = required_f32s(desc, "positions")?;
    let normals = required_f32s(desc, "normals")?;
    let uvs = optional_f32s(desc, "uvs")?;
    let tangents = optional_f32s(desc, "tangents")?;
    let colors = colors(desc)?;
    let indices = indices(desc)?;
    let submeshes = submeshes(st, desc)?;
    from_js_arrays(
        &positions,
        &normals,
        uvs.as_deref(),
        tangents.as_deref(),
        colors.as_ref().map(OwnedColors::as_colors),
        &indices,
        &submeshes,
    )
    .map_err(|e| mesh_error(&e))
}

/// Sets `key` on a fresh plain object; this cannot throw, so the result is ignored.
fn set(obj: &Object, key: &str, value: &JsValue) {
    let _ = Reflect::set(obj, &JsValue::from_str(key), value);
}

fn vec3_to_js(v: Vec3) -> JsValue {
    [v.x, v.y, v.z]
        .iter()
        .map(|&c| JsValue::from(c))
        .collect::<Array>()
        .into()
}

/// The JS handle of material `id`, or null for a material without one (the default
/// material).
fn material_to_js(st: &EngineState, id: Option<MaterialId>) -> JsValue {
    id.and_then(|id| st.materials.iter().find(|&(_, &m)| m == id))
        .map_or(JsValue::NULL, |(h, _)| JsValue::from(h))
}

#[wasm_bindgen(js_class = Engine)]
impl Engine {
    /// Uploads an S13 `MeshDesc` (`positions`, `normals`, `indices`, optional `uvs`,
    /// `tangents`, `colors` (Float32Array linear or Uint8Array sRGB rgba8) and
    /// `subMeshes: [{first, count, material?}]`) and returns its mesh handle. Refusals:
    /// the `mesh.*` codes of S13, `api.unknown_handle` for a sub-mesh material.
    #[wasm_bindgen(js_name = addMesh)]
    pub fn add_mesh(&mut self, desc: JsValue) -> Result<u32, JsValue> {
        self.with_state(|gpu, st| {
            let mesh = mesh_desc(st, &desc)?;
            let lit = st.lit.upload(gpu, &mesh)?;
            Ok(st.meshes.insert(MeshEntry {
                lit,
                origin: MeshOrigin::M07,
            }))
        })
    }

    /// Replaces the geometry and sub-meshes of mesh `id` (any mesh handle) with a
    /// `MeshDesc`. Its instance groups are rebuilt at the next frame, keeping node,
    /// visibility and tag. A refused desc changes nothing.
    #[wasm_bindgen(js_name = updateMesh)]
    pub fn update_mesh(&mut self, id: u32, desc: JsValue) -> Result<(), JsValue> {
        self.with_state(|gpu, st| {
            let entry = st.mesh(id)?;
            let mesh = mesh_desc(st, &desc)?;
            st.lit.update(gpu, entry.lit, &mesh)?;
            st.mark_mesh_dirty(entry.lit);
            Ok(())
        })
    }

    /// `{vertices, triangles, subMeshes: [{first, count, material: handle | null}],
    /// bounds: {min: Vec3, max: Vec3}}` of mesh `id`.
    #[wasm_bindgen(js_name = meshInfo)]
    pub fn mesh_info(&self, id: u32) -> Result<JsValue, JsValue> {
        self.with_state(|_, st| {
            let entry = st.mesh(id)?;
            let info = st.lit.info(entry.lit).ok_or_else(|| {
                api_error(
                    "render.unknown_mesh",
                    format!(
                        "lit mesh {} of mesh handle {id} does not exist",
                        entry.lit.0
                    ),
                )
            })?;
            let subs: Array = info
                .submeshes
                .iter()
                .map(|s| {
                    let o = Object::new();
                    set(&o, "first", &JsValue::from(s.first_index));
                    set(&o, "count", &JsValue::from(s.index_count));
                    set(&o, "material", &material_to_js(st, s.material));
                    JsValue::from(o)
                })
                .collect();
            let bounds = Object::new();
            set(&bounds, "min", &vec3_to_js(info.bounds.min));
            set(&bounds, "max", &vec3_to_js(info.bounds.max));
            let out = Object::new();
            set(&out, "vertices", &JsValue::from(info.vertices));
            set(&out, "triangles", &JsValue::from(info.triangles));
            set(&out, "subMeshes", &subs);
            set(&out, "bounds", &bounds);
            Ok(out.into())
        })
    }

    /// Draws `mesh` at `node` with each sub-mesh's own material (the default material
    /// when it has none) and returns the instance handle (an instance group).
    #[wasm_bindgen(js_name = addMeshInstance)]
    pub fn add_mesh_instance(&mut self, node: u32, mesh: u32) -> Result<u32, JsValue> {
        self.with_state(|_, st| {
            let node = st.node(node)?;
            let mesh = st.mesh(mesh)?;
            st.add_instance_group(node, mesh.lit, None)
        })
    }
}
