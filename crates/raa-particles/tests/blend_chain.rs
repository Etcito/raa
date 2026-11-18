//! `particles.blend` on a bound HDR chain (rg-hdr-chain-binding): two frames without a
//! particle execute Ok with the chain bound to two pool-like textures around the pass.

#[cfg(not(target_arch = "wasm32"))]
mod software_gpu_chain {
    use std::cell::RefCell;
    use std::rc::Rc;

    use raa_core::testing::{skip, software_gpu};
    use raa_core::{FrameGraph, GpuContext, OffscreenTarget, RaaError, wgpu};
    use raa_lit::{LitOptions, LitRenderer};
    use raa_particles::ParticleFeature;
    use raa_pipeline::{GraphFrame, Pipeline, RecordCtx};

    const SIZE: (u32, u32) = (64, 64);
    const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

    fn texture(gpu: &GpuContext) -> wgpu::Texture {
        gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("blend_chain.pool"),
            size: wgpu::Extent3d {
                width: SIZE.0,
                height: SIZE.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    /// Binds the chain around `particles.blend` only.
    struct Wiring {
        src: wgpu::TextureView,
        dst: wgpu::TextureView,
        bound: Rc<RefCell<Vec<bool>>>,
    }

    impl GraphFrame for Wiring {
        fn before_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
            if pass == "particles.blend" {
                ctx.targets.hdr().bind(&self.src, &self.dst);
            }
            Ok(())
        }

        fn after_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
            if pass == "particles.blend" {
                let hdr = ctx.targets.hdr();
                self.bound.borrow_mut().push(hdr.is_bound());
                hdr.unbind();
            }
            Ok(())
        }
    }

    #[test]
    fn blend_without_particles_executes_on_a_bound_chain() {
        let test = "blend_without_particles_executes_on_a_bound_chain";
        let Some(gpu) = software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
        let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, SURFACE, false)));
        let mut lit = LitRenderer::new(
            &gpu,
            &pipeline,
            LitOptions {
                max_instances: 16,
                max_materials: 16,
                msaa: 1,
            },
        )
        .expect("lit renderer");
        ParticleFeature::install(&gpu, &pipeline, &mut lit).expect("install");
        let (a, b) = (texture(&gpu), texture(&gpu));
        let bound = Rc::new(RefCell::new(Vec::new()));
        pipeline
            .borrow_mut()
            .add_graph_frame(
                "test",
                Box::new(Wiring {
                    src: a.create_view(&wgpu::TextureViewDescriptor::default()),
                    dst: b.create_view(&wgpu::TextureViewDescriptor::default()),
                    bound: Rc::clone(&bound),
                }),
            )
            .expect("graph frame");
        let mut graph = FrameGraph::new(SURFACE);
        Pipeline::install(&pipeline, &mut graph);
        graph.compile().expect("compile");
        let target = OffscreenTarget::new(&gpu, SIZE.0, SIZE.1, SURFACE);
        for _ in 0..2 {
            let commands = graph
                .execute(&gpu, target.view(), target.size())
                .expect("execute");
            gpu.queue().submit([commands]);
            gpu.device()
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("device poll");
        }
        assert_eq!(*bound.borrow(), vec![true, true], "bound in both frames");
        assert!(
            pollster_none(errors.pop()),
            "no validation error from the bound frames"
        );
    }

    fn pollster_none(fut: impl std::future::Future<Output = Option<wgpu::Error>>) -> bool {
        raa_core::testing::block_on(fut).is_none()
    }
}
