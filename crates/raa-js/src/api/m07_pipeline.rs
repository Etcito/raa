//! m07 pipeline API: the pass list and its configuration (`passes`, `setPassEnabled`,
//! `setPassOrder`), tone mapping (`setToneMapping`), the display-referred background
//! (`setBackground`) and the device capabilities (`capabilities`).
//!
//! S26: the default full-canvas view retains the direct pipeline path. Split views
//! get pixel-sized target sets and independent post passes. Camera uploads happen
//! between ordered view submissions; output composition preserves earlier views.
//!
//! S25: the `overlay` pass (order 220, a display overlay after `tonemap` and `editor`)
//! draws the m21 overlay layers 0..7 in order, per viewport with that viewport's camera
//! (the lit frame uniform its view uploaded), into that view's final colour target only:
//! the HDR chain, lighting and history buffers never see them. The scene depth is bound
//! read-only for layers with `depthTest`. No overlay data means no render pass.
//!
//! Pipeline refusals (`pipeline.*`) and tone mapping refusals (`post.invalid`) are thrown
//! with their own code; malformed JS arguments throw `api.invalid_argument`.

use js_sys::{Array, Object, Reflect};
use raa_core::{GpuContext, RaaError};
use raa_lit::LitHandle;
use raa_pipeline::{FeaturePass, PassInfo, PassRole, Pipeline, RecordCtx};
use raa_tonemap::{Look, ToneMapping, TonemapControl};
use raa_web::FrameInput;
use wasm_bindgen::prelude::*;

use super::m21_views::{OVERLAY_SEGMENT_FLOATS, OVERLAY_WGSL, OverlayBuffers};
use crate::convert::{INVALID_ARGUMENT, api_error, js, tone_mapper};
use crate::engine::{Engine, InstallCtx};
use crate::state::EngineState;

/// `InstallCtx` does not carry the tonemap control (engine.rs keeps it in
/// `EngineState::tonemap`), so it is stored in the feature map by [`frame`]. Adds the
/// `overlay` pass and stores the overlay buffers it shares with m21_views.
pub(crate) fn install(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
    let buffers = OverlayBuffers::default();
    ctx.features.insert(buffers.clone());
    let pass = OverlayPass {
        lit: ctx.lit.handle(),
        buffers,
        pipelines: None,
    };
    ctx.pipeline
        .try_borrow_mut()
        .map_err(|_| api_error("pipeline.busy", "the pipeline is borrowed during install"))?
        .add(Box::new(pass))
}

/// Order of the `overlay` pass: after `tonemap` (200) and `editor` (210).
const OVERLAY_ORDER: u32 = 220;
const OVERLAY_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const OVERLAY_ATTRIBUTES: [wgpu::VertexAttribute; 4] =
    wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32x4, 3 => Float32x4];

/// Draws the overlay layers after tone mapping (see the module doc).
struct OverlayPass {
    lit: LitHandle,
    buffers: OverlayBuffers,
    /// Built for the output format: [always on top, depth tested].
    pipelines: Option<(wgpu::TextureFormat, [wgpu::RenderPipeline; 2])>,
}

impl OverlayPass {
    fn build(&self, gpu: &GpuContext, format: wgpu::TextureFormat) -> [wgpu::RenderPipeline; 2] {
        let d = gpu.device();
        let frame_layout = self.lit.frame_layout();
        let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raa.overlay"),
            bind_group_layouts: &[Some(&frame_layout)],
            immediate_size: 0,
        });
        let shader = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa.overlay"),
            source: wgpu::ShaderSource::Wgsl(OVERLAY_WGSL.into()),
        });
        let buffers = [Some(wgpu::VertexBufferLayout {
            array_stride: (OVERLAY_SEGMENT_FLOATS * 4) as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &OVERLAY_ATTRIBUTES,
        })];
        [false, true].map(|tested| {
            d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(if tested {
                    "raa.overlay.tested"
                } else {
                    "raa.overlay.top"
                }),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_line"),
                    compilation_options: Default::default(),
                    buffers: &buffers,
                },
                primitive: wgpu::PrimitiveState::default(),
                // The scene depth is never written; untested layers pass always (on top).
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: OVERLAY_DEPTH_FORMAT,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(if tested {
                        wgpu::CompareFunction::LessEqual
                    } else {
                        wgpu::CompareFunction::Always
                    }),
                    stencil: wgpu::StencilState::default(),
                    // Pulls tested lines lying on a surface in front of it.
                    bias: wgpu::DepthBiasState {
                        constant: if tested { -16 } else { 0 },
                        slope_scale: if tested { -1.0 } else { 0.0 },
                        clamp: 0.0,
                    },
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_line"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        })
    }
}

impl FeaturePass for OverlayPass {
    fn name(&self) -> &str {
        "overlay"
    }

    fn order(&self) -> u32 {
        OVERLAY_ORDER
    }

    fn output_overlay(&self) -> bool {
        true
    }

    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let layers = self.buffers.0.borrow();
        if layers.iter().all(|l| l.segments == 0 || l.buffer.is_none()) {
            return Ok(());
        }
        let format = ctx.backbuffer_format;
        if self.pipelines.as_ref().is_none_or(|(f, _)| *f != format) {
            self.pipelines = Some((format, self.build(ctx.gpu, format)));
        }
        let Some((_, pipelines)) = &self.pipelines else {
            return Ok(());
        };
        let frame = self.lit.frame_bind_group();
        // `backbuffer` is the canvas (single view) or this view's output target (split
        // views, composited at its rect afterwards).
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("raa.overlay"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.backbuffer,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: ctx.targets.depth(),
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_bind_group(0, &frame, &[]);
        for layer in layers.iter() {
            let Some(buffer) = layer.buffer.as_ref().filter(|_| layer.segments > 0) else {
                continue;
            };
            pass.set_pipeline(&pipelines[usize::from(layer.depth_test)]);
            pass.set_vertex_buffer(0, buffer.slice(..));
            pass.draw(0..6, 0..layer.segments);
        }
        Ok(())
    }
}

/// Stores the tonemap control once and reconciles viewport targets/post-pass banks.
/// The default full-canvas path performs no recurring allocations here.
pub(crate) fn frame(
    st: &mut EngineState,
    gpu: &GpuContext,
    input: &FrameInput,
) -> Result<(), RaaError> {
    if st.features.get::<TonemapControl>().is_none() {
        st.features.insert(st.tonemap.clone());
    }
    configure_views(st, gpu, input)?;
    Ok(())
}

fn invalid(message: String) -> RaaError {
    api_error(INVALID_ARGUMENT, message)
}

/// Runs `f` on the mutably borrowed pipeline (`pipeline.busy` while a frame executes).
fn with_pipeline<R>(
    st: &EngineState,
    f: impl FnOnce(&mut Pipeline) -> Result<R, RaaError>,
) -> Result<R, RaaError> {
    let mut p = st.pipeline.try_borrow_mut().map_err(|_| {
        api_error(
            "pipeline.busy",
            "the pipeline is borrowed (called while a frame executes)",
        )
    })?;
    f(&mut p)
}

/// The JS name of an AgX look: 'none', 'punchy' or 'golden'.
fn look(name: &str) -> Option<Look> {
    match name {
        "none" => Some(Look::None),
        "punchy" => Some(Look::Punchy),
        "golden" => Some(Look::Golden),
        _ => None,
    }
}

/// Sets `key` on a fresh plain object; this cannot throw, so the result is ignored.
fn set(obj: &Object, key: &str, value: &JsValue) {
    let _ = Reflect::set(obj, &JsValue::from_str(key), value);
}

fn pass_info_to_js(p: &PassInfo) -> JsValue {
    let o = Object::new();
    let role = match p.role {
        PassRole::Pass => "pass",
        PassRole::Contributor => "contributor",
    };
    set(&o, "name", &JsValue::from_str(&p.name));
    set(&o, "role", &JsValue::from_str(role));
    set(&o, "enabled", &JsValue::from_bool(p.enabled));
    set(&o, "order", &JsValue::from(p.order));
    set(&o, "required", &JsValue::from_bool(p.required));
    o.into()
}

/// The enabled WebGPU features of `features`, as their WebGPU names (`float32-filterable`).
fn feature_names(features: wgpu::Features) -> Array {
    wgpu::Features::from(features.features_webgpu)
        .iter()
        .filter_map(|f| f.as_str())
        .map(JsValue::from_str)
        .collect()
}

#[wasm_bindgen(js_class = Engine)]
impl Engine {
    /// Every pass and forward contributor in execution order:
    /// `{name, role: 'pass' | 'contributor', enabled, order, required}`.
    pub fn passes(&self) -> Result<Array, JsValue> {
        self.with_state(|_, st| {
            let list = with_pipeline(st, |p| Ok(p.passes()))?;
            Ok(list.iter().map(pass_info_to_js).collect())
        })
    }

    /// Enables or disables a pass or contributor (`pipeline.unknown_pass`,
    /// `pipeline.required_pass`).
    #[wasm_bindgen(js_name = setPassEnabled)]
    pub fn set_pass_enabled(&mut self, name: &str, enabled: bool) -> Result<(), JsValue> {
        self.with_state(|_, st| with_pipeline(st, |p| p.set_enabled(name, enabled)))
    }

    /// Sets the execution order of the passes: every pass-role name exactly once
    /// (`pipeline.order_incomplete`, `pipeline.unknown_pass`, `pipeline.required_order`).
    #[wasm_bindgen(js_name = setPassOrder)]
    pub fn set_pass_order(&mut self, names: Vec<String>) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            let names: Vec<&str> = names.iter().map(String::as_str).collect();
            with_pipeline(st, |p| p.set_order(&names))
        })
    }

    /// Merges `{mapper?: 'agx' | 'aces' | 'linear', exposureEv?: number (finite),
    /// look?: 'none' | 'punchy' | 'golden', dither?: boolean}` over the current tone
    /// mapping. Unknown names or a non-finite ev throw `api.invalid_argument`; nothing
    /// changes on a refusal.
    #[wasm_bindgen(js_name = setToneMapping)]
    pub fn set_tone_mapping(&mut self, settings: JsValue) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            let mut t: ToneMapping = st.tonemap.get();
            if let Some(name) = js::optional_string(&settings, "mapper")? {
                t.mapper = tone_mapper(&name).ok_or_else(|| {
                    invalid(format!(
                        "'mapper' must be 'agx', 'aces' or 'linear', got '{name}'"
                    ))
                })?;
            }
            if let Some(ev) = js::optional_f64(&settings, "exposureEv")? {
                let ev = ev as f32;
                if !ev.is_finite() {
                    return Err(invalid(format!(
                        "'exposureEv' must be finite as f32, got {ev}"
                    )));
                }
                t.exposure_ev = ev;
            }
            if let Some(name) = js::optional_string(&settings, "look")? {
                t.look = look(&name).ok_or_else(|| {
                    invalid(format!(
                        "'look' must be 'none', 'punchy' or 'golden', got '{name}'"
                    ))
                })?;
            }
            if let Some(d) = js::optional_bool(&settings, "dither")? {
                t.dither = d;
            }
            st.tonemap.set(t)
        })
    }

    /// Sets the display-referred linear background colour (S12), finite and >= 0.
    #[wasm_bindgen(js_name = setBackground)]
    pub fn set_background(&mut self, color: &[f32]) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            let rgb: [f32; 3] = color.try_into().map_err(|_| {
                invalid(format!(
                    "'color' must have 3 components, got {}",
                    color.len()
                ))
            })?;
            if !rgb.iter().all(|c| c.is_finite() && *c >= 0.0) {
                return Err(invalid(format!(
                    "'color' must hold finite numbers >= 0, got {rgb:?}"
                )));
            }
            st.lit.set_background(rgb);
            Ok(())
        })
    }

    /// `{coreWebGpu: true, features: string[] (enabled device features, WebGPU names),
    /// msaa, hdrFormat: 'rgba16float', maxStorageBufferBindingSize}`.
    pub fn capabilities(&self) -> Result<JsValue, JsValue> {
        self.with_state(|gpu, st| {
            let o = Object::new();
            set(&o, "coreWebGpu", &JsValue::TRUE);
            set(&o, "features", &feature_names(gpu.features()));
            set(&o, "msaa", &JsValue::from(st.lit.handle().msaa()));
            set(&o, "hdrFormat", &JsValue::from_str("rgba16float"));
            let max = gpu.limits().max_storage_buffer_binding_size;
            set(
                &o,
                "maxStorageBufferBindingSize",
                &JsValue::from(max as f64),
            );
            Ok(o.into())
        })
    }
}

/// Each post pass owns private scratch textures and (DOF) autofocus history.
struct ViewPost {
    ssr: raa_ssr::SsrHandle,
    dof: raa_dof::DofHandle,
    shafts: raa_shafts::ShaftsHandle,
    bloom: raa_bloom::BloomHandle,
    grading: raa_grading::GradingHandle,
}
#[derive(Default)]
struct ViewPosts(Vec<ViewPost>);

fn configure_views(
    st: &mut EngineState,
    gpu: &GpuContext,
    input: &FrameInput,
) -> Result<(), RaaError> {
    use super::m21_views::{ViewRect, ViewsState};
    let Some(views) = st.features.get::<ViewsState>() else {
        return Ok(());
    };
    if views.viewports() == [ViewRect::FULL] {
        st.pipeline.borrow_mut().set_view_rects(&[]);
        if let Some(posts) = st.features.get_mut::<ViewPosts>() {
            posts.0.clear();
        }
        return Ok(());
    }
    let rects: Vec<_> = views
        .viewports()
        .iter()
        .map(|r| {
            let x = (r.x * input.size.0 as f64).round() as u32;
            let y = (r.y * input.size.1 as f64).round() as u32;
            let right = ((r.x + r.width) * input.size.0 as f64).round() as u32;
            let bottom = ((r.y + r.height) * input.size.1 as f64).round() as u32;
            [x, y, right.saturating_sub(x), bottom.saturating_sub(y)]
        })
        .collect();
    let mut p = st.pipeline.borrow_mut();
    p.set_view_rects(&rects);
    if st.features.get::<ViewPosts>().is_none() {
        st.features.insert(ViewPosts::default());
    }
    let posts = st.features.get_mut::<ViewPosts>().expect("inserted above");
    posts.0.truncate(rects.len());
    while posts.0.len() < rects.len() {
        let temp = std::rc::Rc::new(std::cell::RefCell::new(Pipeline::new(
            gpu,
            p.surface_format(),
            p.surface_is_srgb(),
        )));
        let post = ViewPost {
            ssr: raa_ssr::SsrFeature::install(gpu, &temp)?,
            dof: raa_dof::DofFeature::install(gpu, &temp)?,
            shafts: raa_shafts::ShaftsFeature::install(gpu, &temp)?,
            bloom: raa_bloom::BloomFeature::install(gpu, &temp)?,
            grading: raa_grading::GradingFeature::install(gpu, &temp)?,
        };
        p.set_view_passes(posts.0.len(), &mut temp.borrow_mut());
        posts.0.push(post);
    }
    Ok(())
}

/// Runs only camera-dependent updates. Simulation and animation still advance once.
pub(crate) fn prepare_view(
    st: &mut EngineState,
    gpu: &GpuContext,
    index: usize,
    size: (u32, u32),
) -> Result<(), RaaError> {
    use super::m21_views::ViewsState;
    let original = st.scene.camera;
    if let Some(camera) = st
        .features
        .get::<ViewsState>()
        .and_then(|v| v.camera(index))
        .copied()
    {
        let cam = &mut st.scene.camera;
        cam.eye = raa_math::Vec3::from_array(camera.eye);
        cam.target = raa_math::Vec3::from_array(camera.target);
        cam.up = raa_math::Vec3::Y;
        cam.fov_y_rad = camera.fov_y_deg.to_radians();
        cam.near = camera.near;
        cam.far = camera.far;
        cam.ortho_height = None;
    }
    st.scene.camera.set_viewport(size.0, size.1);
    let result = (|| {
        let input = FrameInput {
            size,
            ..FrameInput::default()
        };
        super::m09_shadows_ao::frame(st, gpu, &input)?;
        super::m14_camera_effects::frame(st, gpu, &input)?;
        if let Some(post) = st.features.get::<ViewPosts>().and_then(|p| p.0.get(index)) {
            let view = st.scene.camera.view();
            let proj = st.scene.camera.projection();
            if let Some(h) = st.features.get::<raa_ssr::SsrHandle>() {
                post.ssr.set(h.get())?;
            }
            if let Some(h) = st.features.get::<raa_dof::DofHandle>() {
                post.dof.set(h.get())?;
            }
            if let Some(h) = st.features.get::<raa_shafts::ShaftsHandle>() {
                post.shafts.set(h.get())?;
            }
            if let Some(h) = st.features.get::<raa_bloom::BloomHandle>() {
                post.bloom.set(h.get())?;
            }
            if let Some(h) = st.features.get::<raa_grading::GradingHandle>() {
                post.grading.set(h.get())?;
            }
            post.ssr.set_camera(&view, &proj);
            post.dof.set_camera(&view, &proj);
            post.shafts.set_camera(&view, &proj);
            let light = st.scene.light;
            post.shafts
                .set_sun(light.direction, light.color.map(|c| c * light.intensity));
        }
        st.lit
            .prepare(gpu, &mut st.scene, &mut st.list, &mut st.tags)?;
        Ok(())
    })();
    st.scene.camera = original;
    result
}

impl EngineState {
    pub(crate) fn prepare_pipeline_view(
        &mut self,
        gpu: &GpuContext,
        index: usize,
        size: (u32, u32),
    ) -> Result<(), RaaError> {
        prepare_view(self, gpu, index, size)
    }
}
