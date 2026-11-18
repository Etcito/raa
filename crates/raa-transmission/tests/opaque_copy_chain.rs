//! `opaque.copy` with a bound HDR chain (contract rg-hdr-chain-binding): the pass accepts
//! a bound source that is not one of its A/B textures and copies it into the destination.
//! Native only; the rig follows `raa-taa`'s `tests/frame.rs` (no scene: nothing draws).

#[cfg(not(target_arch = "wasm32"))]
mod software_gpu_chain {
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    use raa_core::testing::{skip, software_gpu};
    use raa_core::{FrameGraph, GpuContext, OffscreenTarget, RaaError, wgpu};
    use raa_lit::{LitOptions, LitRenderer};
    use raa_pipeline::{GraphFrame, Pipeline, RecordCtx};
    use raa_transmission::TransmissionFeature;

    const SIZE: (u32, u32) = (64, 64);
    const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
    const POOL: wgpu::TextureUsages = wgpu::TextureUsages::RENDER_ATTACHMENT
        .union(wgpu::TextureUsages::TEXTURE_BINDING)
        .union(wgpu::TextureUsages::STORAGE_BINDING)
        .union(wgpu::TextureUsages::COPY_SRC)
        .union(wgpu::TextureUsages::COPY_DST);

    /// Binds the two test views around `opaque.copy` and counts what the pass saw.
    struct Binder {
        views: [wgpu::TextureView; 2],
        bound_seen: Rc<Cell<u32>>,
    }

    impl GraphFrame for Binder {
        fn before_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
            if pass == "opaque.copy" {
                ctx.targets.hdr().bind(&self.views[0], &self.views[1]);
                self.bound_seen
                    .set(self.bound_seen.get() + u32::from(ctx.targets.hdr().is_bound()));
            }
            Ok(())
        }

        fn after_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
            if pass == "opaque.copy" {
                ctx.targets.hdr().unbind();
            }
            Ok(())
        }
    }

    fn pool_view(gpu: &GpuContext) -> wgpu::TextureView {
        gpu.device()
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("opaque_copy_chain.pool"),
                size: wgpu::Extent3d {
                    width: SIZE.0,
                    height: SIZE.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: POOL,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default())
    }

    #[test]
    fn opaque_copy_executes_with_a_bound_chain() {
        let Some(gpu) = software_gpu() else {
            skip("opaque_copy_executes_with_a_bound_chain", "no adapter");
            return;
        };
        let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, SURFACE, false)));
        let mut lit =
            LitRenderer::new(&gpu, &pipeline, LitOptions::default()).expect("lit renderer");
        TransmissionFeature::install(&gpu, &pipeline, &mut lit).expect("install");
        let bound_seen = Rc::new(Cell::new(0));
        {
            let mut p = pipeline.borrow_mut();
            p.set_enabled("opaque.copy", true).expect("entry");
            let binder = Binder {
                views: [pool_view(&gpu), pool_view(&gpu)],
                bound_seen: bound_seen.clone(),
            };
            p.add_graph_frame("test.chain", Box::new(binder))
                .expect("graph frame");
        }
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
        assert_eq!(bound_seen.get(), 2, "the pass ran bound in both frames");
    }
}
