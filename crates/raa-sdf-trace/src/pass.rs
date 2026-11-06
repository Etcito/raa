//! Trace scene state and the section 67 render targets.

use raa_core::GpuContext;
use raa_core::wgpu;

use crate::bvh::{self, Bvh};
use crate::program::{INSTANCE_COUNT, INSTANCE_WORDS, Instance, TraceError, TraceProgram, parse};

pub const MAX_TARGET: u32 = 1024;
pub const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// `ceil(w*f)` x `ceil(h*f)` with `f = min(1, MAX_TARGET / max(w, h))`.
pub fn target_size(width: u32, height: u32) -> (u32, u32) {
    let longest = width.max(height);
    if longest == 0 {
        return (1, 1);
    }
    if longest <= MAX_TARGET {
        return (width.max(1), height.max(1));
    }
    let scale = f64::from(MAX_TARGET) / f64::from(longest);
    (
        ((f64::from(width) * scale).ceil() as u32).max(1),
        ((f64::from(height) * scale).ceil() as u32).max(1),
    )
}

pub struct TraceTargets {
    pub color: wgpu::Texture,
    pub color_view: wgpu::TextureView,
    pub depth: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
    pub size: (u32, u32),
}

impl TraceTargets {
    pub fn new(gpu: &GpuContext, width: u32, height: u32) -> TraceTargets {
        let size = target_size(width, height);
        let make = |label: &str, format: wgpu::TextureFormat| {
            let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size.0,
                    height: size.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    .union(wgpu::TextureUsages::TEXTURE_BINDING)
                    .union(wgpu::TextureUsages::COPY_SRC),
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view)
        };
        let (color, color_view) = make("raa-sdf-trace.color", COLOR_FORMAT);
        let (depth, depth_view) = make("raa-sdf-trace.depth", DEPTH_FORMAT);
        TraceTargets {
            color,
            color_view,
            depth,
            depth_view,
            size,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TraceStats {
    pub sources: usize,
    pub instances: usize,
    pub bvh_nodes: usize,
    pub width: u32,
    pub height: u32,
    pub placeholders: usize,
}

#[derive(Clone, Debug, Default)]
pub struct TraceScene {
    pub program: Option<TraceProgram>,
    pub bvh: Bvh,
}

impl TraceScene {
    pub fn set_program(&mut self, bytes: &[u8]) -> Result<(), TraceError> {
        let program = parse(bytes)?;
        self.bvh = bvh::build(&program);
        self.program = Some(program);
        Ok(())
    }

    pub fn update_instances(&mut self, words: &[f32]) -> Result<(), TraceError> {
        let Some(current) = self.program.as_ref() else {
            return Err(refusal(
                INSTANCE_COUNT,
                "instances",
                "the trace scene has no program",
            ));
        };
        if !words.len().is_multiple_of(INSTANCE_WORDS)
            || words.len() / INSTANCE_WORDS != current.instances.len()
        {
            return Err(refusal(
                INSTANCE_COUNT,
                "instances",
                format!(
                    "{} words is not {} records of {INSTANCE_WORDS} words",
                    words.len(),
                    current.instances.len()
                ),
            ));
        }
        let mut next = current.clone();
        for (index, record) in words.chunks_exact(INSTANCE_WORDS).enumerate() {
            let raw: Vec<u32> = record.iter().map(|value| value.to_bits()).collect();
            let instance = read_instance(&raw, current.instances[index].bend);
            next.instances[index] = instance;
        }
        self.bvh = bvh::build(&next);
        self.program = Some(next);
        Ok(())
    }

    pub fn clear(&mut self) {
        self.program = None;
        self.bvh = Bvh::default();
    }

    pub fn stats(&self, size: (u32, u32)) -> TraceStats {
        let program = self.program.as_ref();
        TraceStats {
            sources: program.map_or(0, |p| p.sources.len()),
            instances: program.map_or(0, |p| p.instances.len()),
            bvh_nodes: self.bvh.nodes.len(),
            width: size.0,
            height: size.1,
            placeholders: program
                .map_or(0, |p| p.sources.iter().filter(|s| s.placeholder()).count()),
        }
    }
}

fn refusal(code: &'static str, path: &str, message: impl Into<String>) -> TraceError {
    TraceError {
        code,
        path: path.to_string(),
        message: message.into(),
    }
}

fn affine(words: &[u32], base: usize) -> [[f32; 4]; 3] {
    core::array::from_fn(|row| {
        core::array::from_fn(|column| f32::from_bits(words[base + row * 4 + column]))
    })
}

fn read_instance(words: &[u32], bend: Option<[f64; 8]>) -> Instance {
    Instance {
        world_from_local: affine(words, 0),
        local_from_world: affine(words, 12),
        dist_scale: f32::from_bits(words[24]),
        source: words[25],
        material: words[26] as i32,
        key: words[27],
        deform_kind: words[28],
        bend_angle: f32::from_bits(words[29]),
        bend_length: f32::from_bits(words[30]),
        axes: words[31],
        bend,
    }
}
