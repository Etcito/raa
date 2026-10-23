//! `TONEMAP_WGSL` validates (CPU, Q11), and a GPU run of Pipeline + TonemapPass on an
//! offscreen Rgba8Unorm target: the empty frame equals the background colour, and the shader
//! agrees with the CPU curves. GPU parts skip with the SKIP line without a software adapter.

#![cfg(not(target_arch = "wasm32"))]

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::testing::{skip, software_gpu};
use raa_core::{CLEAR_COLOR, FrameGraph, GpuContext, OffscreenTarget, RaaError, non_empty};
use raa_pipeline::{FeaturePass, Pipeline, RecordCtx};
use raa_shader::Composer;
use raa_tonemap::curves::{srgb_oetf, tonemap};
use raa_tonemap::{Look, TONEMAP_WGSL, ToneMapper, ToneMapping, TonemapPass};

const SIZE: u32 = 32;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

#[test]
fn tonemap_wgsl_validates() {
    Composer::validate(TONEMAP_WGSL).expect("TONEMAP_WGSL validates");
}

/// Stands in for prepass + forward: clears depth and writes a flat HDR colour to the chain.
struct SetupPass {
    depth: f32,
    hdr: [f64; 3],
}

impl FeaturePass for SetupPass {
    fn name(&self) -> &str {
        "test.setup"
    }

    fn order(&self) -> u32 {
        30
    }

    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let hdr = ctx.targets.hdr();
        {
            let _pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("test.setup"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: hdr.dst(),
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: self.hdr[0],
                            g: self.hdr[1],
                            b: self.hdr[2],
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: ctx.targets.depth(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.depth),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        hdr.swap();
        Ok(())
    }
}

struct Run {
    depth: f32,
    hdr: [f64; 3],
    settings: ToneMapping,
    sky_background: bool,
}

fn render(gpu: &GpuContext, run: &Run) -> Vec<u8> {
    let target = OffscreenTarget::new(gpu, SIZE, SIZE, FORMAT);
    let mut pipeline = Pipeline::new(gpu, FORMAT, false);
    let (pass, control) = TonemapPass::new(gpu, &pipeline);
    control.set(run.settings).expect("valid settings");
    pipeline
        .add(Box::new(SetupPass {
            depth: run.depth,
            hdr: run.hdr,
        }))
        .expect("setup pass added");
    pipeline.add(Box::new(pass)).expect("tonemap pass added");
    if run.sky_background {
        pipeline.lighting_mut().env_mut().sky_params[2] = 1.0;
    }
    let names: Vec<_> = pipeline.passes().into_iter().map(|p| p.name).collect();
    assert_eq!(names, ["test.setup", "tonemap"]);

    let pipeline = Rc::new(RefCell::new(pipeline));
    let mut graph = FrameGraph::new(FORMAT);
    Pipeline::install(&pipeline, &mut graph);
    graph.compile().expect("graph compiles");
    let commands = graph
        .execute(gpu, target.view(), (SIZE, SIZE))
        .expect("graph executes");
    gpu.queue().submit([commands]);
    target.read_rgba8(gpu).expect("readback")
}

fn unorm8(v: f64) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn expected(run: &Run) -> [u8; 4] {
    let hdr = [run.hdr[0] as f32, run.hdr[1] as f32, run.hdr[2] as f32];
    let c = tonemap(&run.settings, hdr);
    [
        unorm8(f64::from(srgb_oetf(c[0]))),
        unorm8(f64::from(srgb_oetf(c[1]))),
        unorm8(f64::from(srgb_oetf(c[2]))),
        255,
    ]
}

fn assert_all_pixels(pixels: &[u8], want: [u8; 4], tolerance: u8, what: &str) {
    assert_eq!(pixels.len(), (SIZE * SIZE * 4) as usize);
    for (i, px) in pixels.chunks_exact(4).enumerate() {
        for c in 0..4 {
            assert!(
                px[c].abs_diff(want[c]) <= tolerance,
                "{what}: pixel {i} is {px:?}, expected {want:?} (±{tolerance})"
            );
        }
    }
}

#[test]
fn empty_scene_leaves_every_pixel_at_the_background_colour() {
    let Some(gpu) = software_gpu() else {
        skip(
            "empty_scene_leaves_every_pixel_at_the_background_colour",
            "software_gpu returned None; see the software_gpu line above",
        );
        return;
    };
    // default settings: AgX, exposure 0, dither on; a bright HDR value must not leak in
    // where depth == 1
    let run = Run {
        depth: 1.0,
        hdr: [4.0, 4.0, 4.0],
        settings: ToneMapping::default(),
        sky_background: false,
    };
    let pixels = render(&gpu, &run);
    let want = [
        unorm8(CLEAR_COLOR.r),
        unorm8(CLEAR_COLOR.g),
        unorm8(CLEAR_COLOR.b),
        unorm8(CLEAR_COLOR.a),
    ];
    assert_all_pixels(&pixels, want, 1, "background");
    assert!(!non_empty(&pixels, want));
}

#[test]
fn shader_matches_the_cpu_curves() {
    let Some(gpu) = software_gpu() else {
        skip(
            "shader_matches_the_cpu_curves",
            "software_gpu returned None; see the software_gpu line above",
        );
        return;
    };
    let cases = [
        (ToneMapper::Agx, Look::None),
        (ToneMapper::Agx, Look::Punchy),
        (ToneMapper::Agx, Look::Golden),
        (ToneMapper::Aces, Look::None),
        (ToneMapper::Linear, Look::None),
    ];
    for (mapper, look) in cases {
        let run = Run {
            depth: 0.5,
            hdr: [2.0, 0.5, 0.1],
            settings: ToneMapping {
                mapper,
                exposure_ev: 0.5,
                look,
                dither: false,
            },
            sky_background: false,
        };
        let want = expected(&run);
        assert_all_pixels(
            &render(&gpu, &run),
            want,
            2,
            &format!("{mapper:?}/{look:?}"),
        );
    }
}

#[test]
fn dither_moves_pixels_by_at_most_one_level() {
    let Some(gpu) = software_gpu() else {
        skip(
            "dither_moves_pixels_by_at_most_one_level",
            "software_gpu returned None; see the software_gpu line above",
        );
        return;
    };
    let run = Run {
        depth: 0.5,
        hdr: [0.18, 0.18, 0.18],
        settings: ToneMapping::default(),
        sky_background: false,
    };
    let undithered = Run {
        settings: ToneMapping {
            dither: false,
            ..run.settings
        },
        ..run
    };
    assert_all_pixels(
        &render(&gpu, &undithered),
        expected(&undithered),
        2,
        "plain",
    );
    assert_all_pixels(&render(&gpu, &run), expected(&undithered), 3, "dithered");
}

#[test]
fn sky_background_is_tone_mapped_where_depth_is_one() {
    let Some(gpu) = software_gpu() else {
        skip(
            "sky_background_is_tone_mapped_where_depth_is_one",
            "software_gpu returned None; see the software_gpu line above",
        );
        return;
    };
    let run = Run {
        depth: 1.0,
        hdr: [0.18, 0.18, 0.18],
        settings: ToneMapping {
            dither: false,
            ..ToneMapping::default()
        },
        sky_background: true,
    };
    assert_all_pixels(&render(&gpu, &run), expected(&run), 2, "sky mode");
}
