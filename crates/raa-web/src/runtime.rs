//! The browser frame loop (wasm32): [`WebRuntime`] and [`RuntimeOptions`].

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use raa_core::{
    AdapterSummary, CanvasSurface, DeviceRequest, FrameGraph, GpuContext, GpuOptions, RaaError,
};
use wasm_bindgen::JsValue;

use crate::stats::CpuRing;
use crate::{FrameCounters, FrameInput, FrameStats, WebApp};

/// Bit pattern meaning "first frame not done yet" (a NaN, which `performance.now()` never is).
const FIRST_FRAME_UNSET: u64 = u64::MAX;

/// How [`WebRuntime::new`] sets up the canvas and the statistics.
#[derive(Clone, Copy, Debug)]
pub struct RuntimeOptions {
    /// Canvas backing width in pixels, written to the canvas before the surface is created.
    pub width: u32,
    /// Canvas backing height in pixels.
    pub height: u32,
    pub gpu: GpuOptions,
    /// Number of recent frames the CPU-ms median and p95 are computed over.
    pub stats_window: usize,
    /// Cumulative allocation count, set by `alloc-count` builds.
    pub alloc_probe: Option<fn() -> u64>,
}

impl Default for RuntimeOptions {
    fn default() -> Self {
        RuntimeOptions {
            width: 1280,
            height: 720,
            gpu: GpuOptions::default(),
            stats_window: 300,
            alloc_probe: None,
        }
    }
}

/// Owns the GPU context, the canvas surface, the frame graph and the application, and runs
/// one frame per [`WebRuntime::frame`] call.
pub struct WebRuntime {
    gpu: GpuContext,
    surface: CanvasSurface,
    graph: FrameGraph,
    app: Box<dyn WebApp>,
    performance: web_sys::Performance,
    alloc_probe: Option<fn() -> u64>,
    frame: u64,
    last_now_ms: Option<f64>,
    counters: FrameCounters,
    allocs_last_frame: Option<u64>,
    cpu_ms: CpuRing,
    /// Percentile scratch, preallocated to the ring capacity. A `RefCell` so that `stats`
    /// takes `&self`; it is borrowed only inside `stats`, which never re-enters itself.
    scratch: RefCell<Vec<f64>>,
    /// `performance.now()` when frame 0's work is done, as f64 bits (FIRST_FRAME_UNSET until
    /// then). wgpu requires the `on_submitted_work_done` callback to be `Send`, which rules
    /// out `Rc<Cell<Option<f64>>>`; an `Arc<AtomicU64>` is the Send equivalent without unsafe.
    first_frame_bits: Arc<AtomicU64>,
}

impl WebRuntime {
    /// Sizes the canvas to `opts.width` x `opts.height`, creates the GPU context, the surface
    /// and the frame graph, lets the app add its passes, and compiles the graph.
    /// Uses the default [`DeviceRequest`] (no optional features, default limits).
    pub async fn new(
        canvas: web_sys::HtmlCanvasElement,
        app: Box<dyn WebApp>,
        opts: RuntimeOptions,
    ) -> Result<WebRuntime, RaaError> {
        Self::new_with(canvas, app, opts, &DeviceRequest::default()).await
    }

    /// [`WebRuntime::new`] with the device created through
    /// [`GpuContext::for_canvas_with`]: a core adapter, `req.optional_features` intersected
    /// with the adapter's features and, when `req.adapter_limits`, the adapter's limits.
    pub async fn new_with(
        canvas: web_sys::HtmlCanvasElement,
        mut app: Box<dyn WebApp>,
        opts: RuntimeOptions,
        req: &DeviceRequest,
    ) -> Result<WebRuntime, RaaError> {
        let performance = web_sys::window()
            .and_then(|window| window.performance())
            .ok_or_else(|| {
                RaaError::Unsupported("window.performance is unavailable".to_string())
            })?;
        canvas.set_width(opts.width.max(1));
        canvas.set_height(opts.height.max(1));
        let (gpu, surface) = GpuContext::for_canvas_with(canvas, &opts.gpu, req).await?;
        let mut graph = FrameGraph::new(surface.format());
        app.init(
            &gpu,
            &mut graph,
            surface.size(),
            surface.format(),
            surface.is_srgb(),
        )?;
        graph.compile()?;
        Ok(WebRuntime {
            gpu,
            surface,
            graph,
            app,
            performance,
            alloc_probe: opts.alloc_probe,
            frame: 0,
            last_now_ms: None,
            counters: FrameCounters::default(),
            allocs_last_frame: None,
            cpu_ms: CpuRing::new(opts.stats_window),
            scratch: RefCell::new(Vec::with_capacity(opts.stats_window)),
            first_frame_bits: Arc::new(AtomicU64::new(FIRST_FRAME_UNSET)),
        })
    }

    /// One frame: app update, acquire, graph execute, submit, present. Records the CPU ms
    /// (`performance.now()` delta) in the ring and, with an alloc probe, the allocation count
    /// of the frame. Allocates nothing itself after warm-up, except the frame-0 callback.
    pub fn frame(&mut self, now_ms: f64) -> Result<(), RaaError> {
        let allocs_before = self.alloc_probe.map(|probe| probe());
        let t0 = self.performance.now();
        let frame = self.update_and_acquire(now_ms)?;
        let commands = self
            .graph
            .execute(&self.gpu, &frame.view, self.surface.size())?;
        self.submit([commands]);
        frame.present();
        self.cpu_ms.push(self.performance.now() - t0);
        if let (Some(probe), Some(before)) = (self.alloc_probe, allocs_before) {
            self.allocs_last_frame = Some(probe().saturating_sub(before));
        }
        Ok(())
    }

    /// Resizes the surface (zero sizes clamp to 1) and tells the app; graph targets follow
    /// the new size at the next execute.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.surface.resize(&self.gpu, width, height);
        self.app.resize(&self.gpu, self.surface.size());
    }

    /// Frame count, first-frame time, CPU-ms median/p95 over the ring (computed in the
    /// preallocated scratch held in a `RefCell`), last counters and last allocation count.
    pub fn stats(&self) -> FrameStats {
        // Invariant: `scratch` is borrowed nowhere else and `stats` does not re-enter.
        let (cpu_ms_median, cpu_ms_p95) = self.cpu_ms.percentiles(&mut self.scratch.borrow_mut());
        let bits = self.first_frame_bits.load(Ordering::Relaxed);
        FrameStats {
            frames: self.frame,
            first_frame_ms: (bits != FIRST_FRAME_UNSET).then_some(f64::from_bits(bits)),
            cpu_ms_median,
            cpu_ms_p95,
            counters: self.counters,
            allocs_last_frame: self.allocs_last_frame,
        }
    }

    pub fn adapter(&self) -> &AdapterSummary {
        self.gpu.summary()
    }

    pub fn gpu(&self) -> &GpuContext {
        &self.gpu
    }

    pub fn app_mut(&mut self) -> &mut dyn WebApp {
        self.app.as_mut()
    }

    /// Runs `app.update` with the frame clock, then acquires the surface texture.
    fn update_and_acquire(&mut self, now_ms: f64) -> Result<raa_core::SurfaceFrame, RaaError> {
        let dt_ms = self.last_now_ms.map_or(0.0, |last| now_ms - last);
        self.last_now_ms = Some(now_ms);
        let input = FrameInput {
            frame: self.frame,
            now_ms,
            dt_ms,
            size: self.surface.size(),
        };
        self.counters = self.app.update(&self.gpu, &input)?;
        self.surface.acquire(&self.gpu)
    }

    /// Submits the frame's commands; on frame 0 registers the first-frame callback.
    fn submit<I: IntoIterator<Item = wgpu::CommandBuffer>>(&mut self, commands: I) {
        self.gpu.queue().submit(commands);
        if self.frame == 0 {
            let first = Arc::clone(&self.first_frame_bits);
            self.gpu.queue().on_submitted_work_done(move || {
                if let Some(performance) = web_sys::window().and_then(|w| w.performance()) {
                    first.store(performance.now().to_bits(), Ordering::Relaxed);
                }
            });
        }
        self.frame += 1;
    }
}

impl WebRuntime {
    /// Renders one frame with the same steps as [`WebRuntime::frame`], copies the backbuffer
    /// to a MAP_READ buffer before presenting, and returns `raa_core::non_empty` against
    /// CLEAR_COLOR quantised to the canvas encoding. When the surface texture lacks COPY_SRC,
    /// the graph also executes into an offscreen target of the surface format, which is the
    /// one copied. The frame counts in `frames` but not in the CPU-ms ring. Supports 8-bit
    /// RGBA/BGRA canvas formats (`core.unsupported` otherwise); a mapping failure gives
    /// `core.gpu`.
    pub async fn probe_non_empty(&mut self, now_ms: f64) -> Result<bool, RaaError> {
        let format = self.surface.format();
        let swap_red_blue = match format {
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => true,
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => false,
            other => {
                return Err(RaaError::Unsupported(format!(
                    "probe_non_empty supports 8-bit RGBA/BGRA canvas formats, not {other:?}"
                )));
            }
        };
        let (width, height) = self.surface.size();
        let row_bytes = width as usize * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
        let padded_row = row_bytes.div_ceil(align) * align;
        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let frame = self.update_and_acquire(now_ms)?;
        let commands = self
            .graph
            .execute(&self.gpu, &frame.view, (width, height))?;
        let offscreen = if self.surface.supports_copy_src() {
            None
        } else {
            let texture = self.gpu.device().create_texture(&wgpu::TextureDescriptor {
                label: Some("raa-web-probe"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let offscreen_commands = self.graph.execute(&self.gpu, &view, (width, height))?;
            Some((texture, offscreen_commands))
        };
        let buffer = self.gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa-web-probe-readback"),
            size: (padded_row * height as usize) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder =
            self.gpu
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("raa-web-probe-readback"),
                });
        let source = offscreen
            .as_ref()
            .map_or(frame.texture(), |(texture, _)| texture);
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: source,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row as u32),
                    rows_per_image: Some(height),
                },
            },
            extent,
        );
        let copy = encoder.finish();
        let offscreen_commands = offscreen.map(|(_, commands)| commands);
        self.submit(
            std::iter::once(commands)
                .chain(offscreen_commands)
                .chain(std::iter::once(copy)),
        );
        // The copy is submitted, so presenting (and the texture expiring at the await) is safe.
        frame.present();

        map_read(&buffer).await?;
        let mut pixels = Vec::with_capacity(row_bytes * height as usize);
        {
            let mapped = buffer
                .slice(..)
                .get_mapped_range()
                .map_err(|e| RaaError::Gpu(format!("mapped range unavailable: {e}")))?;
            for row in mapped.chunks_exact(padded_row) {
                // Invariant: padded_row >= row_bytes.
                pixels.extend_from_slice(&row[..row_bytes]);
            }
        }
        buffer.unmap();
        if swap_red_blue {
            for px in pixels.chunks_exact_mut(4) {
                px.swap(0, 2);
            }
        }
        let clear = crate::quantise_color(raa_core::CLEAR_COLOR, format.is_srgb());
        Ok(raa_core::non_empty(&pixels, clear))
    }
}

/// Maps `buffer` for reading and waits for it through a JS Promise resolved by the
/// `map_async` callback (a oneshot without extra dependencies).
async fn map_read(buffer: &wgpu::Buffer) -> Result<(), RaaError> {
    let mut resolve_slot = None;
    let promise = js_sys::Promise::new(&mut |resolve, _reject| resolve_slot = Some(resolve));
    let Some(resolve) = resolve_slot else {
        return Err(RaaError::Gpu("Promise executor did not run".to_string()));
    };
    let outcome = Rc::new(Cell::new(None));
    let slot = Rc::clone(&outcome);
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            slot.set(Some(result));
            // Calling a Promise's resolve function does not throw; nothing to report from here.
            let _ = resolve.call0(&JsValue::UNDEFINED);
        });
    wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .map_err(|e| RaaError::Gpu(format!("buffer map promise rejected: {e:?}")))?;
    match outcome.take() {
        Some(Ok(())) => Ok(()),
        Some(Err(e)) => Err(RaaError::Gpu(format!("buffer map failed: {e}"))),
        None => Err(RaaError::Gpu(
            "buffer map callback was never called".to_string(),
        )),
    }
}
