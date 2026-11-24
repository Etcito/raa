//! Bridge from the lit renderer's live mesh list to the optional mesh tracer.

use std::collections::BTreeMap;

use js_sys::{Object, Reflect};
use raa_core::RaaError;
use raa_lit::{DrawClass, FLAG_TRACE_MIRROR, FLAG_UNLIT, SurfaceModel};
use raa_math::Mat4;
use raa_sdf_trace::mesh::{
    DiffuseOutput, LightingOutput, MAX_BOUNCES, MAX_LIGHTING_DEPTH, MESH_TRACE_NAME,
    MeshIndirectGrid, MeshInput, MeshTraceFeature, MeshTraceHandle, MeshTraceMaterial,
    MeshTraceSettings, PanelSampling, PathMode, SphericalGiHandle,
};
use raa_shader::{RAA_DIRECT_ONLY_FEATURE, RAA_INDIRECT_ONLY_FEATURE};
use wasm_bindgen::prelude::*;

use crate::{
    convert::{INVALID_ARGUMENT, api_error, js},
    engine::{Engine, InstallCtx},
    state::{EngineState, MeshSources},
};

struct Confidence(raa_ssr::SsrHandle);

pub(crate) fn install_for_graph(ctx: &mut InstallCtx<'_>) {
    if ctx.features.get::<MeshTraceHandle>().is_none() {
        ctx.features.insert(MeshTraceHandle::default());
    }
}

fn shared_handle(
    st: &mut EngineState,
    pipeline: &mut raa_pipeline::Pipeline,
) -> Result<MeshTraceHandle, RaaError> {
    let handle = st
        .features
        .get::<MeshTraceHandle>()
        .cloned()
        .unwrap_or_default();
    if pipeline.entry_enabled(MESH_TRACE_NAME).is_none() {
        MeshTraceFeature::install_legacy(pipeline, handle.clone())?;
    }
    st.features.insert(handle.clone());
    Ok(handle)
}

pub(crate) fn prepare(st: &EngineState, view: usize) -> Result<(), RaaError> {
    let Some(handle) = st.features.get::<MeshTraceHandle>() else {
        return Ok(());
    };
    let scene = handle.borrow();
    let active = scene.active();
    let settings = scene.settings;
    drop(scene);
    let mut spherical = false;
    if let Some(gi) = st.features.get::<SphericalGiHandle>() {
        let mut gi = gi.borrow_mut();
        if gi.active() {
            spherical = view == 0;
            let sun = st.scene.light;
            let ambient = st.scene.ambient;
            let [dx, dy, dz] = sun.direction.to_array();
            let [r, g, b] = sun.color.map(|c| c * sun.intensity);
            let [ar, ag, ab] = ambient.color.map(|c| c * ambient.intensity);
            let camera = st.scene.camera;
            let forward = (camera.target - camera.eye).normalize_or_zero().to_array();
            // Use the diagonal half-angle so corners of a perspective viewport
            // qualify alongside its horizontal and vertical edges.
            let half_fov_cos = if camera.ortho_height.is_none() {
                let slope = (camera.fov_y_rad * 0.5).tan() * camera.aspect.hypot(1.0);
                1.0 / slope.hypot(1.0)
            } else {
                0.0
            };
            gi.set_view(
                view,
                camera.eye.to_array(),
                forward,
                half_fov_cos,
                [dx, dy, dz, r, g, b, ar, ag, ab, camera.far],
            );
            if spherical {
                gi.set_light_ids(st.lit.local_lights().world_lights().map(|(_, id, _)| id));
            }
        }
    }
    if !active && !spherical {
        return Ok(());
    }
    let lit = st.lit.handle();
    if settings.active() && !settings.reference {
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

    let sources = st.features.get::<MeshSources>();
    let mut parts = BTreeMap::new();
    for (handle, mesh) in st.meshes.iter() {
        let (Some(source), Some(ids)) = (
            sources.and_then(|sources| sources.get(handle)),
            st.lit.submesh_ids(mesh.lit),
        ) else {
            continue;
        };
        for (&id, &(first, count)) in ids.iter().zip(&source.submeshes) {
            let Some(gpu) = lit.mesh_storage(id) else {
                continue;
            };
            let indices = &source.mesh.indices[first as usize..(first + count) as usize];
            parts.insert(id, (gpu.key, source, indices));
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
            let (key, source, indices) = parts.get(&item.mesh)?;
            Some(MeshInput {
                key: *key,
                mesh: item.mesh.0,
                positions: &source.mesh.positions,
                normals: &source.mesh.normals,
                colors: source.colors.as_deref(),
                indices,
                world: item.world,
                material: item.material.0,
            })
        })
        .collect::<Vec<_>>();
    let sky_diffuse = st
        .features
        .get::<raa_ibl::IblHandle>()
        .map_or(1.0, raa_ibl::IblHandle::diffuse);
    handle
        .borrow_mut()
        .update(&inputs, &materials, lit.frame_buffer(), sky_diffuse)
}

fn integer(desc: &JsValue, key: &str, previous: u32, max: u32) -> Result<u32, RaaError> {
    let Some(raw) = js::optional_f64(desc, key)? else {
        return Ok(previous);
    };
    if raw.fract() != 0.0 || !(0.0..=f64::from(max)).contains(&raw) {
        return Err(api_error(
            "trace.invalid_settings",
            format!("{key} must be a whole number from 0 to {max}"),
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
            let previous_depth = match previous.lighting_output {
                LightingOutput::Bounce(depth) => depth,
                _ => 0,
            };
            let depth = integer(
                &options,
                "lightingDepth",
                previous_depth,
                MAX_LIGHTING_DEPTH,
            )?;
            let lighting_output = match js::optional_string(&options, "lightingOutput")?.as_deref()
            {
                None if matches!(previous.lighting_output, LightingOutput::Bounce(_)) => {
                    LightingOutput::Bounce(depth)
                }
                None => previous.lighting_output,
                Some("combined") => LightingOutput::Combined,
                Some("direct") => LightingOutput::Direct,
                Some("indirect") => LightingOutput::Indirect,
                Some("diffuse-indirect") => LightingOutput::IndirectDiffuse,
                Some("rough-specular") => LightingOutput::IndirectSpecular,
                Some("rough-grid") => LightingOutput::RoughGrid,
                Some("bounce") => LightingOutput::Bounce(depth),
                Some(_) => {
                    return Err(api_error(
                        "trace.invalid_settings",
                        "unknown lightingOutput",
                    ));
                }
            };
            let settings = MeshTraceSettings {
                lighting_output,
                reflections: js::optional_bool(&options, "reflections")?
                    .unwrap_or(previous.reflections),
                reference: js::optional_bool(&options, "reference")?.unwrap_or(previous.reference),
                confidence_threshold: confidence_threshold as f32,
                reflection_bounces: integer(
                    &options,
                    "reflectionBounces",
                    previous.reflection_bounces,
                    MAX_BOUNCES,
                )?,
                path_bounces: integer(&options, "pathBounces", previous.path_bounces, MAX_BOUNCES)?,
                diffuse_output: match js::optional_string(&options, "diffuseOutput")?.as_deref() {
                    None => previous.diffuse_output,
                    Some("off") => DiffuseOutput::Off,
                    Some("direct") => DiffuseOutput::Direct,
                    Some("direct-and-indirect") => DiffuseOutput::DirectAndIndirect,
                    Some(_) => {
                        return Err(api_error("trace.invalid_settings", "unknown diffuseOutput"));
                    }
                },
                panel_sampling: match js::optional_string(&options, "panelSampling")?.as_deref() {
                    None => previous.panel_sampling,
                    Some("stable") => PanelSampling::Stable,
                    Some("single") => PanelSampling::Single,
                    Some("independent") => PanelSampling::Independent,
                    Some(_) => {
                        return Err(api_error("trace.invalid_settings", "unknown panelSampling"));
                    }
                },
                path_mode: match js::optional_string(&options, "pathMode")?.as_deref() {
                    None => previous.path_mode,
                    Some("off") => PathMode::Off,
                    Some("short") => PathMode::Short,
                    Some("traced") => PathMode::Traced,
                    Some("learned") => PathMode::Learned,
                    Some(_) => return Err(api_error("trace.invalid_settings", "unknown pathMode")),
                },
            }
            .validate()?;
            if settings.path_mode == PathMode::Learned
                && !existing
                    .as_ref()
                    .is_some_and(|handle| handle.borrow().path_model_ready())
            {
                return Err(api_error(
                    "trace.path_model_missing",
                    "load a specular fit before enabling learned path tracing",
                ));
            }
            let grid = match js::optional_string(&options, "indirectGrid")?.as_deref() {
                None => existing
                    .as_ref()
                    .map_or(MeshIndirectGrid::Auto, |h| h.borrow().indirect_grid),
                Some("auto") => MeshIndirectGrid::Auto,
                Some("off") => MeshIndirectGrid::Off,
                Some("volume") => MeshIndirectGrid::Volume,
                Some("spherical") => MeshIndirectGrid::Spherical,
                Some(_) => return Err(api_error("trace.invalid_settings", "unknown indirectGrid")),
            };
            let hybrid = !settings.reference && settings.active();
            let pipeline = st.pipeline.clone();
            let mut pipeline = pipeline.borrow_mut();
            let handle = shared_handle(st, &mut pipeline)?;
            if hybrid && st.features.get::<Confidence>().is_none() {
                let confidence = raa_ssr::SsrFeature::install_confidence(gpu, &mut pipeline)?;
                st.features.insert(Confidence(confidence));
            }
            if st.features.get::<Confidence>().is_some() {
                pipeline.set_enabled(raa_ssr::CONFIDENCE_NAME, hybrid)?;
            }
            pipeline.set_enabled(MESH_TRACE_NAME, settings.active())?;
            let mut trace = handle.borrow_mut();
            trace.configure(settings);
            trace.indirect_grid = grid;
            let raster = !settings.reference && settings.path_mode == PathMode::Off;
            st.lit.set_frame_feature(
                RAA_DIRECT_ONLY_FEATURE,
                raster && lighting_output == LightingOutput::Direct,
            );
            st.lit.set_frame_feature(
                RAA_INDIRECT_ONLY_FEATURE,
                raster && lighting_output == LightingOutput::Indirect,
            );
            Ok(())
        })
    }

    /// Loads a previously validated 12→8→3 specular fit in exporter tensor order.
    #[wasm_bindgen(js_name = setMeshPathModel)]
    pub fn set_mesh_path_model(&self, weights: &[u8]) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            let handle = if let Some(handle) = st.features.get::<MeshTraceHandle>().cloned() {
                handle
            } else {
                let handle = MeshTraceHandle::default();
                st.features.insert(handle.clone());
                handle
            };
            handle.borrow_mut().set_path_model(weights)
        })
    }

    /// The latest completed GPU counter sample, or null until one is available.
    #[wasm_bindgen(js_name = meshTracingStats)]
    pub fn mesh_tracing_stats(&self, node: Option<String>) -> Result<JsValue, JsValue> {
        self.with_state(|_, st| {
            let Some(handle) = st.features.get::<MeshTraceHandle>() else {
                return Ok(JsValue::NULL);
            };
            let stats = match node {
                Some(node) => handle.borrow().graph_stats(&node),
                None => handle.borrow_mut().stats(),
            };
            let Some(stats) = stats else {
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
                "modelPixels",
                "refreshPixels",
                "diffuseMissingPixels",
                "diffuseRays",
                "specularRays",
                "pathForegroundPixels",
                "modelRejectedPixels",
                "primaryBvhPixels",
                "firstHitRays",
                "firstHitHits",
                "firstHitHash",
                "sphericalDiffuseHits",
                "volumeDiffuseHits",
                "volumeSpecularHits",
                "sphericalSpecularHits",
                "sphericalDiffuseOffscreenInGrid",
                "sphericalDiffuseOffscreenSupported",
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
                ("referenceSpp", f64::from(stats.reference_spp)),
                (
                    "hybridLobeSpp",
                    if !controls.reference && controls.path_mode != PathMode::Off {
                        1.0
                    } else {
                        0.0
                    },
                ),
            ] {
                Reflect::set(&result, &name.into(), &value.into()).expect("fresh object");
            }
            Ok(result.into())
        })
    }
}
