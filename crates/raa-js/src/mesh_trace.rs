//! Bridge from the lit renderer's live mesh list to the optional mesh tracer.

use std::collections::BTreeMap;

use js_sys::{Object, Reflect};
use raa_core::RaaError;
use raa_lit::{DrawClass, FLAG_TRACE_MIRROR, FLAG_UNLIT, SurfaceModel};
use raa_math::Mat4;
use raa_sdf_trace::mesh::{
    MAX_BOUNCES, MESH_TRACE_NAME, MeshInput, MeshTraceFeature, MeshTraceHandle, MeshTraceMaterial,
    MeshTraceSettings,
};
use wasm_bindgen::prelude::*;

use crate::{
    convert::{INVALID_ARGUMENT, api_error, js},
    engine::Engine,
    state::{EngineState, MeshSources},
};

struct Confidence(raa_ssr::SsrHandle);

pub(crate) fn prepare(st: &EngineState) -> Result<(), RaaError> {
    let Some(handle) = st.features.get::<MeshTraceHandle>() else {
        return Ok(());
    };
    let settings = handle.borrow().settings;
    if !settings.active() {
        return Ok(());
    }
    let lit = st.lit.handle();
    if settings.reflections && !settings.reference {
        let mut jitter = Mat4::IDENTITY;
        let [x, y] = lit.camera_jitter_ndc();
        jitter.cols[3] = [x, y, 0.0, 1.0];
        if let Some(probe) = st.features.get::<Confidence>() {
            probe.0.set_camera(
                &st.scene.camera.view(),
                &(jitter * st.scene.camera.projection()),
            );
        }
    }

    let Some(sources) = st.features.get::<MeshSources>() else {
        return Ok(());
    };
    let mut parts = BTreeMap::new();
    for (handle, mesh) in st.meshes.iter() {
        let (Some(source), Some(ids)) = (sources.get(handle), st.lit.submesh_ids(mesh.lit)) else {
            continue;
        };
        for (&id, &(first, count)) in ids.iter().zip(&source.submeshes) {
            let Some(gpu) = lit.mesh_storage(id) else {
                continue;
            };
            let indices = &source.mesh.indices[first as usize..(first + count) as usize];
            parts.insert(id, (gpu.key, &source.mesh, indices));
        }
    }

    let library = st.lit.materials();
    let materials = (0..library.len() as u32)
        .map(|index| {
            let material = library
                .get(raa_core::MaterialId(index))
                .expect("dense lit materials");
            MeshTraceMaterial {
                base: material.base_color,
                roughness: material.roughness,
                emissive: material.emissive,
                metallic: material.metallic,
                mirror: material.flags & FLAG_TRACE_MIRROR != 0,
                unlit: material.flags & FLAG_UNLIT != 0 || material.model == SurfaceModel::Flat,
            }
        })
        .collect::<Vec<_>>();

    let inputs = st
        .list
        .items
        .iter()
        .filter_map(|item| {
            let material = library.get(item.material)?;
            if material.class() == DrawClass::Transparent {
                return None;
            }
            let (key, mesh, indices) = parts.get(&item.mesh)?;
            Some(MeshInput {
                key: *key,
                mesh: item.mesh.0,
                positions: &mesh.positions,
                normals: &mesh.normals,
                indices,
                world: item.world,
                material: item.material.0,
            })
        })
        .collect::<Vec<_>>();
    handle
        .borrow_mut()
        .update(&inputs, &materials, lit.frame_buffer())
}

fn integer(desc: &JsValue, key: &str, previous: u32) -> Result<u32, RaaError> {
    let Some(raw) = js::optional_f64(desc, key)? else {
        return Ok(previous);
    };
    if raw.fract() != 0.0 || !(0.0..=f64::from(MAX_BOUNCES)).contains(&raw) {
        return Err(api_error(
            "trace.invalid_settings",
            format!("{key} must be a whole number from 0 to {MAX_BOUNCES}"),
        ));
    }
    Ok(raw as u32)
}

#[wasm_bindgen]
impl Engine {
    /// Configures the reflection confidence gate and the optional path reference.
    #[wasm_bindgen(js_name = setMeshTracing)]
    pub fn set_mesh_tracing(&self, options: JsValue) -> Result<(), JsValue> {
        self.with_state(|gpu, st| {
            if !options.is_object() || options.is_null() || js_sys::Array::is_array(&options) {
                return Err(api_error(
                    INVALID_ARGUMENT,
                    "mesh tracing options must be an object",
                ));
            }
            let existing = st.features.get::<MeshTraceHandle>().cloned();
            let previous = existing
                .as_ref()
                .map_or_else(MeshTraceSettings::default, |h| h.borrow().settings);
            let confidence_threshold = js::optional_f64(&options, "confidenceThreshold")?
                .unwrap_or(f64::from(previous.confidence_threshold));
            if !(0.0..=1.0).contains(&confidence_threshold) {
                return Err(api_error(
                    "trace.invalid_settings",
                    "confidenceThreshold must be in [0, 1]",
                ));
            }
            let settings = MeshTraceSettings {
                reflections: js::optional_bool(&options, "reflections")?
                    .unwrap_or(previous.reflections),
                reference: js::optional_bool(&options, "reference")?.unwrap_or(previous.reference),
                confidence_threshold: confidence_threshold as f32,
                reflection_bounces: integer(
                    &options,
                    "reflectionBounces",
                    previous.reflection_bounces,
                )?,
                path_bounces: integer(&options, "pathBounces", previous.path_bounces)?,
            }
            .validate()?;
            let hybrid = settings.reflections && !settings.reference;
            let mut pipeline = st.pipeline.borrow_mut();
            let handle = if let Some(handle) = existing {
                handle
            } else {
                let handle = MeshTraceFeature::install(&mut pipeline)?;
                st.features.insert(handle.clone());
                handle
            };
            if hybrid && st.features.get::<Confidence>().is_none() {
                let confidence = raa_ssr::SsrFeature::install_confidence(gpu, &mut pipeline)?;
                st.features.insert(Confidence(confidence));
            }
            if st.features.get::<Confidence>().is_some() {
                pipeline.set_enabled(raa_ssr::CONFIDENCE_NAME, hybrid)?;
            }
            pipeline.set_enabled(MESH_TRACE_NAME, settings.active())?;
            handle.borrow_mut().configure(settings);
            Ok(())
        })
    }

    /// The latest completed GPU counter sample, or null until one is available.
    #[wasm_bindgen(js_name = meshTracingStats)]
    pub fn mesh_tracing_stats(&self) -> Result<JsValue, JsValue> {
        self.with_state(|_, st| {
            let Some(handle) = st.features.get::<MeshTraceHandle>() else {
                return Ok(JsValue::NULL);
            };
            let Some(stats) = handle.borrow_mut().stats() else {
                return Ok(JsValue::NULL);
            };
            let result = Object::new();
            let names = [
                "eligible",
                "acceptedScreen",
                "traced",
                "screenMiss",
                "offscreen",
                "disocclusion",
                "lowConfidence",
                "nestedMirror",
                "geometryHit",
                "mirrorRays",
                "pathRays",
                "referencePixels",
            ];
            for (name, value) in names.into_iter().zip(stats.counts) {
                Reflect::set(&result, &name.into(), &f64::from(value).into())
                    .expect("fresh object");
            }
            let controls = stats.settings;
            for (name, value) in [
                ("frame", stats.frame as f64),
                ("width", f64::from(stats.size.0)),
                ("height", f64::from(stats.size.1)),
                (
                    "confidenceThreshold",
                    f64::from(controls.confidence_threshold),
                ),
                ("reflectionBounces", f64::from(controls.reflection_bounces)),
                ("pathBounces", f64::from(controls.path_bounces)),
            ] {
                Reflect::set(&result, &name.into(), &value.into()).expect("fresh object");
            }
            Ok(result.into())
        })
    }
}
