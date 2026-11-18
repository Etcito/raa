//! [`Lazy`]: a GPU object (a pipeline, usually) built the first time a frame needs it.
//!
//! Why: a browser keeps a bounded cache of compiled WebGPU pipelines (Chrome: 6 MB per
//! profile, least recently used first). An engine that compiles every pass of every feature
//! at init (disabled post effects, the GI modes that are not selected, particles in a scene
//! without emitters) fills that cache with pipelines no frame binds and evicts the ones the
//! frame does use, which are then compiled again at every page load (2026-11-18: 40 of the
//! 125 pipelines of a 4-cube SDF Modeler scene were never bound, ~1.5 MB of compiled cache).
//! A feature keeps the cheap objects (layouts, buffers) eager and wraps each pipeline in a
//! `Lazy`, so a pass that never records never compiles.

use std::cell::OnceCell;
use std::fmt;

/// A value built by `build` on the first [`Lazy::get`] and kept after that.
pub struct Lazy<T> {
    cell: OnceCell<T>,
    build: Box<dyn Fn() -> T>,
}

impl<T> Lazy<T> {
    /// A value `build` makes on first use. `build` owns what it needs (the device and the
    /// layouts and modules it reads are cheap clones in wgpu).
    pub fn new(build: impl Fn() -> T + 'static) -> Self {
        Self {
            cell: OnceCell::new(),
            build: Box::new(build),
        }
    }

    /// The value, built now on the first call.
    pub fn get(&self) -> &T {
        self.cell.get_or_init(|| (self.build)())
    }

    /// True once the value was built.
    pub fn is_built(&self) -> bool {
        self.cell.get().is_some()
    }
}

/// A compute pipeline of `module`'s `entry` over the bind group layouts `groups` (a pipeline
/// layout of its own) with the override `constants`, built on its first use.
pub fn lazy_compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    module: &wgpu::ShaderModule,
    entry: &str,
    groups: &[Option<&wgpu::BindGroupLayout>],
    constants: &[(&'static str, f64)],
) -> Lazy<wgpu::ComputePipeline> {
    let (device, module) = (device.clone(), module.clone());
    let (label, entry) = (label.to_owned(), entry.to_owned());
    let groups: Vec<Option<wgpu::BindGroupLayout>> = groups.iter().map(|g| g.cloned()).collect();
    let constants = constants.to_vec();
    Lazy::new(move || {
        let refs: Vec<Option<&wgpu::BindGroupLayout>> = groups.iter().map(Option::as_ref).collect();
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&label),
            bind_group_layouts: &refs,
            immediate_size: 0,
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&label),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(&entry),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            },
            cache: None,
        })
    })
}

impl<T> fmt::Debug for Lazy<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Lazy")
            .field("built", &self.is_built())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::Lazy;
    use std::cell::Cell;
    use std::rc::Rc;

    #[test]
    fn builds_once_on_first_get() {
        let calls = Rc::new(Cell::new(0));
        let counter = calls.clone();
        let lazy = Lazy::new(move || {
            counter.set(counter.get() + 1);
            7
        });
        assert!(!lazy.is_built());
        assert_eq!(calls.get(), 0);
        assert_eq!(*lazy.get(), 7);
        assert_eq!(*lazy.get(), 7);
        assert_eq!(calls.get(), 1);
        assert!(lazy.is_built());
    }
}
