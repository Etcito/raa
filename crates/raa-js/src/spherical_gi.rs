//! Public controls for the optional spherical indirect-light grid.

use js_sys::{Array, Object, Reflect};
use raa_core::RaaError;
use raa_sdf_trace::mesh::{
    MeshTraceHandle, SPHERICAL_GI_NAME, SphericalGiDebug, SphericalGiFeature, SphericalGiHandle,
    SphericalGiSettings,
};
use wasm_bindgen::{JsCast, prelude::*};

use crate::{
    convert::api_error,
    engine::{Engine, InstallCtx},
    state::EngineState,
};

pub(crate) fn install_for_graph(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
    let mesh = ctx
        .features
        .require_mut::<MeshTraceHandle>("mesh trace")?
        .clone();
    let handle = SphericalGiFeature::install(&mut ctx.pipeline.borrow_mut(), mesh)?;
    ctx.features.insert(handle);
    Ok(())
}

const OPTIONS: [&str; 11] = [
    "enabled",
    "sphereSubdivision",
    "exponentialPower",
    "clipStart",
    "zMax",
    "debug",
    "channelMask",
    "maxProbes",
    "captureFaces",
    "captureMs",
    "burstMs",
];

fn invalid(key: &str, reason: &str) -> RaaError {
    api_error("gi.spherical_invalid", format!("{key}: {reason}"))
}

fn read_patch(
    patch: JsValue,
    mut settings: SphericalGiSettings,
) -> Result<SphericalGiSettings, RaaError> {
    if !patch.is_object() || patch.is_null() || Array::is_array(&patch) {
        return Err(invalid("settings", "expected an object"));
    }
    for key in Object::keys(patch.unchecked_ref::<Object>()).iter() {
        let name = key.as_string().expect("Object.keys returns strings");
        if !OPTIONS.contains(&name.as_str()) {
            return Err(invalid(&name, "unknown key"));
        }
        let value =
            Reflect::get(&patch, &key).map_err(|_| invalid(&name, "cannot read setting"))?;
        if value.is_undefined() {
            continue;
        }
        let grid = &mut settings.scheduler.grid;
        let number = || {
            value
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| invalid(&name, "expected a finite number"))
        };
        match name.as_str() {
            "enabled" => {
                settings.enabled = value
                    .as_bool()
                    .ok_or_else(|| invalid(&name, "expected a boolean"))?;
            }
            "sphereSubdivision" => {
                let subdivision = number()?;
                if !(0.0..=6.0).contains(&subdivision) || subdivision.fract() != 0.0 {
                    return Err(invalid(&name, "expected an integer from 0 to 6"));
                }
                grid.angular_resolution = 1 << subdivision as u32;
            }
            "exponentialPower" => {
                let exponent = number()?;
                if !(0.0..=2.0).contains(&exponent) {
                    return Err(invalid(&name, "expected a number from 0 to 2"));
                }
                grid.radial_growth = exponent.exp2() as f32;
            }
            "clipStart" => {
                let clip_start = number()?;
                if clip_start < 0.0 {
                    return Err(invalid(&name, "expected a nonnegative radius"));
                }
                grid.clip_start = clip_start as f32;
            }
            "zMax" => grid.z_max = number()? as f32,
            "channelMask" => {
                let mask = number()?;
                if !(0.0..=7.0).contains(&mask) || mask.fract() != 0.0 {
                    return Err(invalid(&name, "expected an integer from 0 to 7"));
                }
                settings.channels = mask as u32;
            }
            "maxProbes" => {
                let probes = number()?;
                if !(16.0..=4096.0).contains(&probes) || probes.fract() != 0.0 {
                    return Err(invalid(&name, "expected an integer from 16 to 4096"));
                }
                settings.scheduler.max_cells = probes as u32;
            }
            "captureFaces" => {
                let faces = number()?;
                if !(1.0..=1024.0).contains(&faces) || faces.fract() != 0.0 {
                    return Err(invalid(&name, "expected an integer from 1 to 1024"));
                }
                settings.budget.near = faces as u32;
                settings.budget.far = (faces as u32).div_ceil(2);
            }
            "captureMs" | "burstMs" => {
                let ms = number()?;
                if !(0.0..=50.0).contains(&ms) {
                    return Err(invalid(&name, "expected milliseconds from 0 to 50"));
                }
                if name == "captureMs" {
                    settings.capture_ms = ms as f32;
                } else {
                    settings.burst_ms = ms as f32;
                }
            }
            "debug" => {
                settings.debug = match value.as_string().as_deref() {
                    Some("off") => SphericalGiDebug::Off,
                    Some("boundaries") => SphericalGiDebug::Boundaries,
                    Some("occupancy") => SphericalGiDebug::Occupancy,
                    Some("age") => SphericalGiDebug::Age,
                    Some("light-hash") => SphericalGiDebug::LightHash,
                    _ => {
                        return Err(invalid(
                            &name,
                            "expected off, boundaries, occupancy, age or light-hash",
                        ));
                    }
                };
            }
            _ => unreachable!("known setting"),
        }
    }
    let grid = settings.scheduler.grid;
    if !grid.clip_start.is_finite() || grid.clip_start < 0.0 {
        return Err(invalid("clipStart", "expected a finite nonnegative radius"));
    }
    if !grid.z_max.is_finite() || grid.z_max <= grid.clip_start {
        return Err(invalid("zMax", "must be finite and greater than clipStart"));
    }
    settings
        .validate()
        .map_err(|error| invalid("grid", &error.to_string()))
}

fn put(object: &Object, key: &str, value: JsValue) {
    Reflect::set(object, &key.into(), &value).expect("fresh object");
}

fn number_array<T: Into<JsValue>, const N: usize>(values: [T; N]) -> Array {
    let result = Array::new();
    for value in values {
        result.push(&value.into());
    }
    result
}

fn settings_object(settings: SphericalGiSettings) -> JsValue {
    let grid = settings.scheduler.grid;
    let result = Object::new();
    let debug = match settings.debug {
        SphericalGiDebug::Off => "off",
        SphericalGiDebug::Boundaries => "boundaries",
        SphericalGiDebug::Occupancy => "occupancy",
        SphericalGiDebug::Age => "age",
        SphericalGiDebug::LightHash => "light-hash",
    };
    put(&result, "enabled", settings.enabled.into());
    put(
        &result,
        "sphereSubdivision",
        grid.angular_resolution.ilog2().into(),
    );
    put(
        &result,
        "exponentialPower",
        f64::from(grid.radial_growth).log2().into(),
    );
    put(&result, "clipStart", grid.clip_start.into());
    put(&result, "zMax", grid.z_max.into());
    put(&result, "debug", debug.into());
    put(&result, "channelMask", settings.channels.into());
    put(&result, "maxProbes", settings.scheduler.max_cells.into());
    put(&result, "captureFaces", settings.budget.near.into());
    put(&result, "captureMs", settings.capture_ms.into());
    put(&result, "burstMs", settings.burst_ms.into());
    result.into()
}

fn settings(state: &EngineState) -> SphericalGiSettings {
    state
        .features
        .get::<SphericalGiHandle>()
        .map(|handle| handle.borrow().settings)
        .unwrap_or_default()
}

#[wasm_bindgen]
impl Engine {
    /// Merges spherical settings. The pass allocates GPU capture resources on first use.
    #[wasm_bindgen(js_name = setSphericalGi)]
    pub fn set_spherical_gi(&self, options: JsValue) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            let settings = read_patch(options, settings(st))?;
            let handle = st
                .features
                .require_mut::<SphericalGiHandle>("spherical GI")?;
            let mut pipeline = st.pipeline.borrow_mut();
            let enabled = settings.enabled;
            handle.borrow_mut().configure(settings)?;
            pipeline.set_enabled(SPHERICAL_GI_NAME, enabled)?;
            st.lit
                .set_frame_feature(raa_shader::RAA_SPHERICAL_GI_FEATURE, enabled);
            Ok(())
        })
    }

    /// Effective settings for this engine.
    #[wasm_bindgen(js_name = sphericalGi)]
    pub fn spherical_gi(&self) -> Result<JsValue, JsValue> {
        self.with_state(|_, st| Ok(settings_object(settings(st))))
    }

    /// Snapshot of the active spherical grid and its capture work.
    #[wasm_bindgen(js_name = sphericalGiStats)]
    pub fn spherical_gi_stats(&self) -> Result<JsValue, JsValue> {
        self.with_state(|_, st| {
            let Some(handle) = st.features.get::<SphericalGiHandle>() else {
                return Ok(JsValue::NULL);
            };
            let state = handle.borrow();
            let Some(stats) = state.stats() else {
                return Ok(JsValue::NULL);
            };
            let result = Object::new();
            put(&result, "active", state.active().into());
            let scheduler = stats.scheduler;
            for (name, value) in [
                ("cells", scheduler.cells as f64),
                ("nearCells", scheduler.near_cells as f64),
                ("farCells", scheduler.far_cells as f64),
                ("readyCells", scheduler.ready_cells as f64),
                ("readyFaces", scheduler.ready_faces as f64),
                ("inFlightFaces", scheduler.in_flight_faces as f64),
                ("pendingFaces", scheduler.pending_faces as f64),
                ("originEpoch", scheduler.origin_epoch as f64),
                ("reflectedCells", scheduler.reflected_cells as f64),
                (
                    "reflectedReadyCells",
                    scheduler.reflected_ready_cells as f64,
                ),
                ("lastCellQueries", scheduler.last_cell_queries as f64),
                (
                    "lightingMetadataReadyCells",
                    scheduler.lighting_metadata_ready_cells as f64,
                ),
                (
                    "lightInvalidatedCells",
                    scheduler.light_invalidated_cells as f64,
                ),
                (
                    "lightInvalidatedFaces",
                    scheduler.light_invalidated_faces as f64,
                ),
                ("gpuBytes", stats.gpu_bytes as f64),
                ("capturedFaces", stats.captured_faces as f64),
                ("captureFrames", stats.capture_frames as f64),
                ("visibleCells", scheduler.visible_cells as f64),
                (
                    "visiblePublishedCells",
                    scheduler.visible_published_cells as f64,
                ),
                ("faceBudget", stats.face_budget as f64),
                ("captureGpuMs", f64::from(stats.capture_gpu_ms)),
            ] {
                put(&result, name, value.into());
            }
            put(
                &result,
                "lightRevision",
                format!("{:016x}", stats.light_revision).into(),
            );
            put(&result, "origin", number_array(stats.origin).into());
            put(&result, "burst", stats.burst.into());
            put(
                &result,
                "discoveryPending",
                scheduler.discovery_pending.into(),
            );
            let shell = Object::new();
            put(&shell, "inner", stats.last_shell.inner.into());
            put(&shell, "center", stats.last_shell.center.into());
            put(&shell, "outer", stats.last_shell.outer.into());
            put(&result, "lastShell", shell.into());
            Ok(result.into())
        })
    }

    /// Polls one ready probe's GPU texel readback, requesting one if none is pending.
    #[wasm_bindgen(js_name = sphericalGiProbe)]
    pub fn spherical_gi_probe(&self) -> Result<JsValue, JsValue> {
        self.with_state(|_, st| {
            let Some(handle) = st.features.get::<SphericalGiHandle>() else {
                return Ok(JsValue::NULL);
            };
            let Some(probe) = handle.borrow_mut().probe() else {
                return Ok(JsValue::NULL);
            };
            let result = Object::new();
            put(&result, "slot", probe.slot.into());
            put(&result, "frame", (probe.frame as f64).into());
            put(&result, "faceHits", number_array(probe.face_hits).into());
            put(
                &result,
                "faceValidGeometry",
                number_array(probe.face_valid_geometry).into(),
            );
            put(
                &result,
                "faceValidMisses",
                number_array(probe.face_valid_misses).into(),
            );
            let sums = Array::new();
            for rgb in probe.channel_sums {
                sums.push(&number_array(rgb));
            }
            put(&result, "channelSums", sums.into());
            put(
                &result,
                "lightHash",
                format!("{:016x}", probe.light_hash).into(),
            );
            let light_ids = Array::new();
            for id in probe.contributing_light_ids {
                light_ids.push(&format!("{id:016x}").into());
            }
            put(&result, "contributingLightIds", light_ids.into());
            Ok(result.into())
        })
    }
}
