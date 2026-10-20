//! M1 boot demo: the clear colour plus one solid orange triangle, rendered through the
//! RAA frame graph on WebGPU. The wasm-bindgen API lives behind `cfg(target_arch = "wasm32")`,
//! so host builds compile an empty library; the passes also build for host unit tests.

#[cfg(any(target_arch = "wasm32", test))]
mod app {
    use raa_core::{
        BACKBUFFER, CLEAR_COLOR, ClearPass, FrameGraph, GpuContext, PassBuilder, RaaError,
        RecordContext, RenderPass,
    };
    use raa_web::{FrameCounters, FrameInput, WebApp};

    /// Draws one triangle generated in the shader over what earlier passes wrote.
    pub(crate) struct TrianglePass {
        pipeline: wgpu::RenderPipeline,
    }

    impl TrianglePass {
        /// Builds the pipeline once, for the surface format. `SRGB_ENCODE` makes the
        /// fragment shader encode linear to sRGB when the canvas format is not `*Srgb`.
        pub(crate) fn new(
            gpu: &GpuContext,
            surface_format: wgpu::TextureFormat,
            surface_is_srgb: bool,
        ) -> Self {
            let device = gpu.device();
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("triangle"),
                source: wgpu::ShaderSource::Wgsl(include_str!("triangle.wgsl").into()),
            });
            let constants = [("SRGB_ENCODE", if surface_is_srgb { 0.0 } else { 1.0 })];
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("triangle"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &constants,
                        zero_initialize_workgroup_memory: false,
                    },
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
            TrianglePass { pipeline }
        }
    }

    impl RenderPass for TrianglePass {
        fn name(&self) -> &str {
            "triangle"
        }

        fn declare(&self, builder: &mut PassBuilder) {
            builder.write(BACKBUFFER);
        }

        fn record(&mut self, ctx: &mut RecordContext<'_>) -> Result<(), RaaError> {
            // The clear pass writes the backbuffer first, so this is a Load in practice.
            let load = if ctx.is_first_write(BACKBUFFER) {
                wgpu::LoadOp::Clear(CLEAR_COLOR)
            } else {
                wgpu::LoadOp::Load
            };
            let view = ctx.view(BACKBUFFER)?;
            let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("triangle"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.draw(0..3, 0..1);
            Ok(())
        }
    }

    /// The demo application: a `clear` pass followed by the `triangle` pass.
    pub(crate) struct BootApp;

    impl WebApp for BootApp {
        fn init(
            &mut self,
            gpu: &GpuContext,
            graph: &mut FrameGraph,
            _size: (u32, u32),
            surface_format: wgpu::TextureFormat,
            surface_is_srgb: bool,
        ) -> Result<(), RaaError> {
            graph.add_pass(Box::new(ClearPass::new("clear", BACKBUFFER, CLEAR_COLOR)));
            let triangle = TrianglePass::new(gpu, surface_format, surface_is_srgb);
            graph.add_pass(Box::new(triangle));
            Ok(())
        }

        fn update(
            &mut self,
            _gpu: &GpuContext,
            _input: &FrameInput,
        ) -> Result<FrameCounters, RaaError> {
            Ok(FrameCounters {
                instances: 1,
                draw_calls: 1,
                triangles: 1,
            })
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod web {
    use raa_web::{RuntimeOptions, WebRuntime, error_to_js, stats_to_js};
    use wasm_bindgen::prelude::*;

    use crate::app::BootApp;

    /// Reads `options[key]` as a positive integer size, if present and valid.
    fn size_option(options: &JsValue, key: &str) -> Option<u32> {
        if !options.is_object() {
            return None;
        }
        let value = js_sys::Reflect::get(options, &JsValue::from_str(key))
            .ok()?
            .as_f64()?;
        // Rejects NaN, non-positive and out-of-range values before the cast.
        (1.0..=f64::from(u32::MAX))
            .contains(&value)
            .then_some(value as u32)
    }

    /// The running demo, driven from JS.
    #[wasm_bindgen]
    pub struct Demo {
        runtime: WebRuntime,
    }

    /// Starts the demo on `canvas`. `options` may carry `width` and `height` (canvas
    /// backing size in pixels); the canvas's current size is used otherwise.
    #[wasm_bindgen]
    pub async fn start(
        canvas: web_sys::HtmlCanvasElement,
        options: JsValue,
    ) -> Result<Demo, JsValue> {
        let opts = RuntimeOptions {
            width: size_option(&options, "width").unwrap_or(canvas.width()),
            height: size_option(&options, "height").unwrap_or(canvas.height()),
            ..RuntimeOptions::default()
        };
        let runtime = WebRuntime::new(canvas, Box::new(BootApp), opts)
            .await
            .map_err(|e| error_to_js(&e))?;
        Ok(Demo { runtime })
    }

    #[wasm_bindgen]
    impl Demo {
        pub fn frame(&mut self, now_ms: f64) -> Result<(), JsValue> {
            self.runtime.frame(now_ms).map_err(|e| error_to_js(&e))
        }

        pub fn resize(&mut self, width: u32, height: u32) {
            self.runtime.resize(width, height);
        }

        pub fn stats(&self) -> JsValue {
            stats_to_js(&self.runtime.stats(), self.runtime.adapter())
        }

        pub async fn probe_non_empty(&mut self, now_ms: f64) -> Result<bool, JsValue> {
            self.runtime
                .probe_non_empty(now_ms)
                .await
                .map_err(|e| error_to_js(&e))
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub use web::{Demo, start};

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use raa_core::testing::{skip, software_gpu};
    use raa_core::{BACKBUFFER, FrameGraph, GpuContext, OffscreenTarget};
    use raa_web::{FrameInput, WebApp};

    use crate::app::BootApp;

    const SIZE: u32 = 64;

    /// Renders one frame of the boot app into an offscreen target and reads it back.
    fn render(gpu: &GpuContext, format: wgpu::TextureFormat, is_srgb: bool) -> Vec<u8> {
        let target = OffscreenTarget::new(gpu, SIZE, SIZE, format);
        let mut graph = FrameGraph::new(format);
        let mut app = BootApp;
        app.init(gpu, &mut graph, (SIZE, SIZE), format, is_srgb)
            .expect("app init");
        graph.compile().expect("graph compiles");
        assert_eq!(graph.pass_names(), ["clear", "triangle"]);
        assert_eq!(graph.first_write_of("triangle", BACKBUFFER), Some(false));
        let commands = graph
            .execute(gpu, target.view(), (SIZE, SIZE))
            .expect("graph executes");
        gpu.queue().submit([commands]);
        let pixels = target.read_rgba8(gpu).expect("readback");
        // Leaked like the context (see the end of the test): dropping a pipeline on Mesa's
        // D3D12 GL adapter (WSL) also leaves the driver to crash at process exit.
        std::mem::forget(graph);
        std::mem::forget(target);
        pixels
    }

    fn pixel(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
        let i = ((y * SIZE + x) * 4) as usize;
        [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
    }

    fn assert_close(actual: [u8; 4], expected: [u8; 4], what: &str) {
        for c in 0..4 {
            assert!(
                actual[c].abs_diff(expected[c]) <= 2,
                "{what}: {actual:?}, expected {expected:?} (±2)"
            );
        }
    }

    #[test]
    fn one_triangle_is_counted_drawn_orange_and_centred_on_the_clear_colour() {
        let Some(gpu) = software_gpu() else {
            skip(
                "one_triangle_is_counted_drawn_orange_and_centred_on_the_clear_colour",
                "software_gpu returned None; see the software_gpu line above",
            );
            return;
        };
        let input = FrameInput::default();
        let counters = BootApp.update(&gpu, &input).expect("update");
        assert_eq!(counters.instances, 1);
        assert_eq!(counters.draw_calls, 1);
        assert_eq!(counters.triangles, 1);

        // Linear (1.0, 0.35, 0.05) encoded to sRGB: by the shader on a Unorm target
        // (SRGB_ENCODE on), by the hardware on an Srgb target.
        let orange = [255, 160, 63, 255];
        let clear_linear = [13, 15, 20, 255];
        let clear_srgb = [63, 69, 80, 255];
        for (format, is_srgb, clear) in [
            (wgpu::TextureFormat::Rgba8Unorm, false, clear_linear),
            (wgpu::TextureFormat::Rgba8UnormSrgb, true, clear_srgb),
        ] {
            let pixels = render(&gpu, format, is_srgb);
            let (mid, top, bottom) = (SIZE / 2, SIZE / 4, SIZE * 3 / 4 - 1);
            // Inside: the centre and just inside the apex (clip y 0.5 is row SIZE / 4) and
            // the two base corners. Outside: the image corners and beside the apex.
            let inside = [
                (mid, mid),
                (mid, top + 2),
                (top + 2, bottom),
                (bottom - 2, bottom),
            ];
            let outside = [
                (0, 0),
                (SIZE - 1, 0),
                (0, SIZE - 1),
                (SIZE - 1, SIZE - 1),
                (top, top + 2),
            ];
            for (x, y) in inside {
                let what = format!("{format:?} inside ({x}, {y})");
                assert_close(pixel(&pixels, x, y), orange, &what);
            }
            for (x, y) in outside {
                let what = format!("{format:?} outside ({x}, {y})");
                assert_close(pixel(&pixels, x, y), clear, &what);
            }
        }
        let summary = gpu.summary();
        let kind = if summary.is_hardware() {
            "hardware"
        } else {
            "software"
        };
        eprintln!(
            "m1-boot triangle test: adapter `{}` ({}, {kind})",
            summary.device, summary.backend
        );
        // Tearing down a wgpu instance that enumerated Mesa's D3D12 GL adapter (WSL) segfaults
        // inside the driver (see raa-core's gpu_readback test). Leak the context at the end so
        // the process exits cleanly; all GPU work has completed by now.
        std::mem::forget(gpu);
    }
}
