//! Installation and statistics of the DDGI probe volume (`pass.gi.ddgi`). The render graph
//! configures it; there is no standalone setter.

use js_sys::{Array, Object, Reflect, Uint8Array};
use raa_core::RaaError;
use raa_sdf_trace::mesh::{DdgiFeature, DdgiHandle, MeshTraceHandle};
use wasm_bindgen::prelude::*;

use crate::engine::{Engine, InstallCtx};

pub(crate) fn install_for_graph(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
    let mesh = ctx
        .features
        .require_mut::<MeshTraceHandle>("mesh trace")?
        .clone();
    let handle = DdgiFeature::install(&mut ctx.pipeline.borrow_mut(), mesh)?;
    ctx.features.insert(handle);
    Ok(())
}

fn put(object: &Object, key: &str, value: JsValue) {
    Reflect::set(object, &key.into(), &value).expect("fresh object");
}

#[wasm_bindgen]
impl Engine {
    /// Snapshot of the active DDGI volume and its update schedule, or null.
    #[wasm_bindgen(js_name = ddgiGiStats)]
    pub fn ddgi_gi_stats(&self) -> Result<JsValue, JsValue> {
        self.with_state(|_, st| {
            let Some(handle) = st.features.get::<DdgiHandle>() else {
                return Ok(JsValue::NULL);
            };
            let state = handle.borrow();
            let result = Object::new();
            put(&result, "active", state.active().into());
            let s = state.settings;
            let counts = Array::new();
            for n in s.counts {
                counts.push(&n.into());
            }
            put(&result, "counts", counts.into());
            put(&result, "spacing", s.spacing.into());
            put(&result, "cascades", s.cascades.into());
            let cascade_counts = Array::new();
            for n in s.cascade_counts {
                cascade_counts.push(&n.into());
            }
            put(&result, "cascadeCounts", cascade_counts.into());
            put(&result, "far", (s.far as u32).into());
            put(&result, "screenAo", s.screen_ao.into());
            put(&result, "raysPerProbe", s.rays_per_probe.into());
            put(&result, "updateFraction", s.update_fraction.into());
            put(&result, "hysteresis", s.hysteresis.into());
            put(&result, "budgetMs", s.budget_ms.into());
            put(&result, "burstMs", s.burst_ms.into());
            put(&result, "skyTransfer", s.sky_transfer.into());
            put(&result, "surfaceSpacing", s.surface_spacing.into());
            put(&result, "surfaceProbes", s.surface_probes.into());
            put(&result, "invalidation", (s.invalidation as u32).into());
            if let Some(stats) = state.stats() {
                put(&result, "probes", stats.probes.into());
                put(&result, "scheduled", stats.scheduled.into());
                put(&result, "updateFrames", (stats.update_frames as f64).into());
                put(&result, "recycled", (stats.recycled as f64).into());
                let base = Array::new();
                for b in stats.base {
                    base.push(&b.into());
                }
                put(&result, "base", base.into());
                let bases = Array::new();
                for b in stats.cascade_bases.iter().take(stats.cascades as usize) {
                    let base = Array::new();
                    for x in b {
                        base.push(&(*x).into());
                    }
                    bases.push(&base);
                }
                put(&result, "cascadeBases", bases.into());
                put(&result, "gpuBytes", (stats.gpu_bytes as f64).into());
                put(&result, "burst", stats.burst.into());
                put(&result, "visible", stats.visibility.visible.into());
                put(&result, "visibleValid", stats.visibility.valid.into());
                put(
                    &result,
                    "visibleConverged",
                    stats.visibility.converged.into(),
                );
                put(&result, "convergeAge", stats.converge_age.into());
                put(&result, "msPerProbe", stats.ms_per_probe.into());
                put(
                    &result,
                    "updateMs",
                    stats.update_ms.map_or(JsValue::NULL, JsValue::from),
                );
                put(&result, "screenResolve", stats.screen_resolve.into());
                put(&result, "interval", stats.interval.into());
                // Lighting changes: group refreshes traced and probe groups invalidated since
                // the volume started, as [geometry + sky transfer, sun + sky residual, other].
                let triple = |values: [u64; 3]| {
                    let array = Array::new();
                    for value in values {
                        array.push(&(value as f64).into());
                    }
                    JsValue::from(array)
                };
                put(&result, "refreshed", triple(stats.refreshed));
                put(&result, "invalidated", triple(stats.invalidated));
                put(&result, "relights", (stats.relights as f64).into());
                put(&result, "lightEdits", (stats.light_edits as f64).into());
                put(&result, "lastLightProbes", stats.last_light_probes.into());
                put(&result, "lightMasks", stats.light_masks.into());
                put(&result, "hidden", stats.visibility.hidden.into());
                put(
                    &result,
                    "hiddenConverged",
                    stats.visibility.hidden_converged.into(),
                );
                put(&result, "surfacePool", stats.surface_pool.into());
                put(&result, "surfaceLive", stats.surface_live.into());
                put(
                    &result,
                    "surfaceAssigned",
                    (stats.surface_assigned as f64).into(),
                );
                put(
                    &result,
                    "surfaceEvicted",
                    (stats.surface_evicted as f64).into(),
                );
                put(
                    &result,
                    "surfaceRequests",
                    (stats.surface_requests as f64).into(),
                );
                // Frames between two updates of the same probe: [cascade 0 in the frustum,
                // cascade 0 outside, coarser cascades, surface probes] x power-of-two buckets.
                let periods = Array::new();
                for class in stats.periods {
                    let row = Array::new();
                    for count in class {
                        row.push(&(count as f64).into());
                    }
                    periods.push(&row);
                }
                put(&result, "periods", periods.into());
                // Rows (16 bytes) of the probe buffer for ddgiGiDump: records, resolve, table.
                let rows = Array::new();
                for row in stats.rows {
                    rows.push(&row.into());
                }
                put(&result, "rows", rows.into());
                if let Some((min, max)) = stats.scene_bounds {
                    let bounds = Array::new();
                    for x in min.into_iter().chain(max) {
                        bounds.push(&x.into());
                    }
                    put(&result, "sceneBounds", bounds.into());
                }
            }
            Ok(result.into())
        })
    }

    /// Diagnostic readback of `rows` 16-byte rows of the DDGI probe buffer from `first_row`
    /// (see `ddgiGiStats().rows`): probe records (4 rows per slot: coordinate and state, offset
    /// and last back-face fraction / nearest front face, light mask), the screen resolve, the
    /// surface-probe table.
    #[wasm_bindgen(js_name = ddgiGiDump)]
    pub async fn ddgi_gi_dump(&mut self, first_row: u32, rows: u32) -> Result<JsValue, JsValue> {
        let readback = self.with_state(|gpu, st| {
            let source = st
                .features
                .get::<DdgiHandle>()
                .and_then(|handle| handle.borrow().probe_buffer())
                .ok_or_else(|| RaaError::Resource("no active DDGI volume".into()))?;
            let (offset, bytes) = (16 * u64::from(first_row), 16 * u64::from(rows));
            if bytes == 0 || offset + bytes > source.size() {
                return Err(RaaError::Resource("DDGI dump out of range".into()));
            }
            let device = gpu.device();
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gi.ddgi.dump"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gi.ddgi.dump"),
            });
            encoder.copy_buffer_to_buffer(&source, offset, &readback, 0, bytes);
            gpu.queue().submit([encoder.finish()]);
            Ok(readback)
        })?;
        let bytes = raa_web::read_capture_buffer(&readback)
            .await
            .map_err(|error| raa_web::error_to_js(&error))?;
        Ok(Uint8Array::from(bytes.as_slice()).into())
    }

    /// Reads one published image of the latest submitted frame (`producer` is the pipeline
    /// entry, e.g. `mesh.reference#reference` or `forward`; `port` its output) without rendering.
    /// Used for accumulated GI references and comparisons: `{ size, format, data }`.
    #[wasm_bindgen(js_name = captureGraphImage)]
    pub async fn capture_graph_image(
        &mut self,
        producer: String,
        port: String,
    ) -> Result<JsValue, JsValue> {
        let texture = self.with_state(|gpu, st| {
            if !gpu.summary().is_hardware() {
                return Err(RaaError::Unsupported(
                    "graph image capture requires hardware WebGPU".into(),
                ));
            }
            let pipeline = st.pipeline.try_borrow().map_err(|_| {
                RaaError::Resource("graph image capture: the pipeline is busy".into())
            })?;
            let submission = pipeline.scene_submission();
            if submission.record_failed
                || submission.submitted_mask != 1
                || submission.pending_mask != 0
            {
                return Err(RaaError::Resource(
                    "graph image capture requires one submitted primary view".into(),
                ));
            }
            let image = pipeline.resources().get(&producer, &port).ok_or_else(|| {
                RaaError::Resource(format!("no published image {producer}.{port}"))
            })?;
            Ok(image.view.texture().clone())
        })?;
        let runtime = self.capture_runtime()?;
        let bytes = raa_web::read_capture_texture(runtime.gpu(), &texture)
            .await
            .map_err(|error| raa_web::error_to_js(&error))?;
        let result = Object::new();
        let size = Array::new();
        size.push(&texture.width().into());
        size.push(&texture.height().into());
        put(&result, "size", size.into());
        put(&result, "format", format!("{:?}", texture.format()).into());
        put(&result, "data", Uint8Array::from(bytes.as_slice()).into());
        Ok(result.into())
    }
}
