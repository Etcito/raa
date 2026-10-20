//! Canvas surface (wasm32): [`CanvasSurface`] and [`SurfaceFrame`].

use crate::{GpuContext, RaaError};

/// A configured WebGPU canvas surface.
#[derive(Debug)]
pub struct CanvasSurface {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    /// False when the adapter does not allow COPY_SRC on the surface texture.
    copy_src: bool,
}

impl CanvasSurface {
    pub(crate) fn new(
        gpu: &GpuContext,
        surface: wgpu::Surface<'static>,
        width: u32,
        height: u32,
    ) -> Result<CanvasSurface, RaaError> {
        let caps = surface.get_capabilities(gpu.adapter());
        let Some(&format) = caps.formats.first() else {
            return Err(RaaError::Surface(
                "the surface is incompatible with the adapter (no formats)".to_string(),
            ));
        };
        let alpha_mode = caps
            .alpha_modes
            .first()
            .copied()
            .unwrap_or(wgpu::CompositeAlphaMode::Auto);
        let copy_src = caps.usages.contains(wgpu::TextureUsages::COPY_SRC);
        let usage = if copy_src {
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC
        } else {
            wgpu::TextureUsages::RENDER_ATTACHMENT
        };
        let config = wgpu::SurfaceConfiguration {
            usage,
            format,
            color_space: wgpu::SurfaceColorSpace::default(),
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode,
            view_formats: Vec::new(),
        };
        surface.configure(gpu.device(), &config);
        Ok(CanvasSurface {
            surface,
            config,
            copy_src,
        })
    }

    /// The preferred (first capability) format the surface is configured with.
    pub fn format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    pub fn is_srgb(&self) -> bool {
        self.config.format.is_srgb()
    }

    /// True when the surface texture also has COPY_SRC usage.
    pub fn supports_copy_src(&self) -> bool {
        self.copy_src
    }

    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Reconfigures for a new size; zero sizes clamp to 1.
    pub fn resize(&mut self, gpu: &GpuContext, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(gpu.device(), &self.config);
    }

    /// Acquires the next frame. On Outdated or Lost, reconfigures and retries once.
    pub fn acquire(&mut self, gpu: &GpuContext) -> Result<SurfaceFrame, RaaError> {
        let texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(gpu.device(), &self.config);
                match self.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(t)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
                    other => return Err(acquire_error(&other, true)),
                }
            }
            other => return Err(acquire_error(&other, false)),
        };
        let view = texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        Ok(SurfaceFrame {
            view,
            texture,
            queue: gpu.queue().clone(),
        })
    }
}

fn acquire_error(status: &wgpu::CurrentSurfaceTexture, retried: bool) -> RaaError {
    let what = match status {
        wgpu::CurrentSurfaceTexture::Success(_) | wgpu::CurrentSurfaceTexture::Suboptimal(_) => {
            "success"
        }
        wgpu::CurrentSurfaceTexture::Timeout => "timeout",
        wgpu::CurrentSurfaceTexture::Occluded => "occluded",
        wgpu::CurrentSurfaceTexture::Outdated => "outdated",
        wgpu::CurrentSurfaceTexture::Lost => "lost",
        wgpu::CurrentSurfaceTexture::Validation => "validation error",
    };
    let suffix = if retried { " after reconfiguring" } else { "" };
    RaaError::Surface(format!(
        "could not acquire the surface texture: {what}{suffix}"
    ))
}

/// One acquired surface texture and its default view.
#[derive(Debug)]
pub struct SurfaceFrame {
    pub view: wgpu::TextureView,
    texture: wgpu::SurfaceTexture,
    /// wgpu 30 presents through the queue, so the frame keeps a (cheap) queue clone.
    queue: wgpu::Queue,
}

impl SurfaceFrame {
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture.texture
    }

    /// Presents the frame (wgpu 30 equivalent of `SurfaceTexture::present`).
    pub fn present(self) {
        self.queue.present(self.texture);
    }
}
