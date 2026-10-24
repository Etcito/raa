//! `ParticleFeature::install` and the [`ParticleHandle`]: the shared emitter state
//! (state.rs) seen by the `particles.sim` pass (sim.rs, order 50), the `particles.opaque`
//! forward contributor (opaque.rs, order 600) and the `particles.blend` pass (blend.rs,
//! order 85).
//!
//! The install is atomic like `raa_transmission::TransmissionFeature::install`: the
//! pipeline borrow and the three entry names are checked first, then every object that can
//! refuse (the sim pass, the opaque module with the snapshot of the slot winners, the blend
//! pipelines) is built before the first entry is added, so a refusal leaves the pipeline
//! and the lit renderer as they were.

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::{GpuContext, RaaError};
use raa_lit::LitRenderer;
use raa_pipeline::Pipeline;

use crate::blend::{self, BlendPass};
use crate::desc::EmitterDesc;
use crate::opaque::{self, OpaqueContributor};
use crate::sim::{self, SimPass};
use crate::state::{EmitterTable, SharedEmitters};

fn refusal(code: &'static str, message: &str) -> RaaError {
    RaaError::FrameGraph {
        code,
        message: message.to_string(),
    }
}

/// Same refusal as raa-lit's for a pipeline borrowed while a frame executes.
fn pipeline_busy() -> RaaError {
    refusal(
        "pipeline.busy",
        "the pipeline is borrowed (called while a frame executes)",
    )
}

/// The emitter state is borrowed (called while a frame executes).
fn state_busy() -> RaaError {
    refusal(
        "pipeline.busy",
        "the particle state is borrowed (called while a frame executes)",
    )
}

fn duplicate(name: &str) -> RaaError {
    refusal(
        "pipeline.duplicate_pass",
        &format!("the entry `{name}` is already installed"),
    )
}

/// Installs GPU particles on a pipeline and a lit renderer.
pub struct ParticleFeature;

impl ParticleFeature {
    /// Creates the shared emitter state, adds the FeaturePass `particles.sim` (order 50),
    /// the forward contributor `particles.opaque` (order 600, through
    /// `lit.add_contributor`) and the FeaturePass `particles.blend` (order 85).
    ///
    /// Refusals (nothing installed): `pipeline.busy` when the pipeline is borrowed,
    /// `pipeline.duplicate_pass` when any of the three entries already exists, and the
    /// shader refusal of the draw modules with its code (`shader.slot_missing` when a slot
    /// has no provider).
    pub fn install(
        gpu: &GpuContext,
        pipeline: &Rc<RefCell<Pipeline>>,
        lit: &mut LitRenderer,
    ) -> Result<ParticleHandle, RaaError> {
        {
            let p = pipeline.try_borrow().map_err(|_| pipeline_busy())?;
            let entries = p.passes();
            for name in [sim::NAME, opaque::NAME, blend::NAME] {
                if entries.iter().any(|info| info.name == name) {
                    return Err(duplicate(name));
                }
            }
        }
        let emitters = EmitterTable::shared();
        let sim = SimPass::new(gpu, Rc::clone(&emitters))?;
        let buffers = sim.buffers();
        let opaque = OpaqueContributor::new(gpu, lit, Rc::clone(&emitters), Rc::clone(&buffers))?;
        let blend = {
            let p = pipeline.try_borrow().map_err(|_| pipeline_busy())?;
            BlendPass::new(gpu, lit, p.lighting(), Rc::clone(&emitters), buffers)?
        };
        pipeline
            .try_borrow_mut()
            .map_err(|_| pipeline_busy())?
            .add(Box::new(sim))?;
        lit.add_contributor(Box::new(opaque))?;
        pipeline
            .try_borrow_mut()
            .map_err(|_| pipeline_busy())?
            .add(Box::new(blend))?;
        Ok(ParticleHandle { state: emitters })
    }
}

/// Access to the installed particle emitters (clones share one state).
#[derive(Clone)]
pub struct ParticleHandle {
    state: SharedEmitters,
}

impl ParticleHandle {
    /// Validates and adds `d` under the lowest free id (dense ids, holes reused).
    /// Refusals: `particles.invalid` (validation first, naming the field),
    /// `particles.capacity` (more than 64 emitters or more than 262144 particles) and
    /// `pipeline.busy` on a re-entrant call.
    pub fn add(&self, d: EmitterDesc) -> Result<u32, RaaError> {
        d.validate()?;
        self.state
            .try_borrow_mut()
            .map_err(|_| state_busy())?
            .add(d)
    }

    /// Validates and replaces the description of `id`, reallocating its slot range when
    /// `max_particles` changes. Refusals: `particles.invalid` (validation first),
    /// `particles.unknown_emitter`, `particles.capacity`, `pipeline.busy`.
    pub fn set(&self, id: u32, d: EmitterDesc) -> Result<(), RaaError> {
        d.validate()?;
        self.state
            .try_borrow_mut()
            .map_err(|_| state_busy())?
            .set(id, d)
    }

    /// Removes `id` and frees its slot range (the id is reused next). Refusals:
    /// `particles.unknown_emitter`, `pipeline.busy`.
    pub fn remove(&self, id: u32) -> Result<(), RaaError> {
        self.state
            .try_borrow_mut()
            .map_err(|_| state_busy())?
            .remove(id)
    }

    /// Adds `count` particles at the next frame's first fixed step. Refusals:
    /// `particles.unknown_emitter`, `pipeline.busy`.
    pub fn burst(&self, id: u32, count: u32) -> Result<(), RaaError> {
        self.state
            .try_borrow_mut()
            .map_err(|_| state_busy())?
            .burst(id, count)
    }

    /// Particle slots in use: the sum over emitters of
    /// `min(max_particles, particles spawned so far)`, without a GPU readback. Returns 0
    /// on a re-entrant call.
    pub fn count(&self) -> u32 {
        self.state.try_borrow().map(|s| s.count()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::desc::ParticleKind;

    fn handle() -> ParticleHandle {
        ParticleHandle {
            state: EmitterTable::shared(),
        }
    }

    fn invalid() -> EmitterDesc {
        let mut d = EmitterDesc::for_kind(ParticleKind::Splash);
        d.spread_rad = -0.5;
        d
    }

    #[test]
    fn ids_are_dense_and_the_lowest_free_is_reused() {
        let h = handle();
        let d = EmitterDesc::for_kind(ParticleKind::Snow);
        assert_eq!(h.add(d.clone()).unwrap(), 0);
        assert_eq!(h.add(d.clone()).unwrap(), 1);
        assert_eq!(h.add(d.clone()).unwrap(), 2);
        h.remove(1).unwrap();
        assert_eq!(h.add(d).unwrap(), 1);
    }

    #[test]
    fn invalid_descriptions_are_refused_before_anything_else() {
        let h = handle();
        let err = h.add(invalid()).err().unwrap();
        assert_eq!(err.code(), "particles.invalid");
        assert!(!err.to_string().is_empty());
        let err = h.set(99, invalid()).err().unwrap();
        assert_eq!(
            err.code(),
            "particles.invalid",
            "validate before unknown id"
        );
    }

    #[test]
    fn unknown_emitters_are_refused() {
        let h = handle();
        let d = EmitterDesc::for_kind(ParticleKind::Confetti);
        assert_eq!(
            h.set(0, d).err().unwrap().code(),
            "particles.unknown_emitter"
        );
        assert_eq!(
            h.remove(0).err().unwrap().code(),
            "particles.unknown_emitter"
        );
        assert_eq!(
            h.burst(0, 4).err().unwrap().code(),
            "particles.unknown_emitter"
        );
    }

    #[test]
    fn sixty_four_emitters_are_the_limit() {
        let h = handle();
        let mut d = EmitterDesc::for_kind(ParticleKind::Sparkles);
        d.max_particles = 1;
        d.rate_per_s = 0.0;
        for id in 0..64 {
            assert_eq!(h.add(d.clone()).unwrap(), id);
        }
        let err = h.add(d).err().unwrap();
        assert_eq!(err.code(), "particles.capacity");
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn more_than_262144_particles_are_refused() {
        let h = handle();
        let mut d = EmitterDesc::for_kind(ParticleKind::Splash);
        d.max_particles = 65_536;
        d.rate_per_s = 0.0;
        for _ in 0..4 {
            h.add(d.clone()).unwrap();
        }
        let err = h.add(d).err().unwrap();
        assert_eq!(err.code(), "particles.capacity");
    }

    #[test]
    fn clones_share_one_state() {
        let h = handle();
        let clone = h.clone();
        let id = clone
            .add(EmitterDesc::for_kind(ParticleKind::Splash))
            .unwrap();
        assert_eq!(id, 0);
        h.remove(id).unwrap();
        assert!(h.count() == 0 && clone.count() == 0);
        assert_eq!(
            clone.remove(id).err().unwrap().code(),
            "particles.unknown_emitter"
        );
    }

    #[test]
    fn burst_and_count_delegate_to_the_simulated_state() {
        let h = handle();
        let mut d = EmitterDesc::for_kind(ParticleKind::Snow);
        d.rate_per_s = 0.0;
        d.max_particles = 16;
        let id = h.add(d).unwrap();
        h.burst(id, 5).unwrap();
        assert_eq!(h.count(), 0, "nothing spawned before a step");
        h.state.borrow_mut().advance(0.1);
        assert_eq!(h.count(), 5);
    }
}

// GPU tests: `raa_core::testing` is native-only.
#[cfg(all(test, not(target_arch = "wasm32")))]
mod gpu_tests {
    use super::*;
    use crate::desc::ParticleKind;
    use raa_core::testing::{skip, software_gpu};
    use raa_core::wgpu;
    use raa_lit::LitOptions;
    use raa_pipeline::PassRole;

    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

    fn setup(gpu: &GpuContext) -> (Rc<RefCell<Pipeline>>, LitRenderer) {
        let pipeline = Rc::new(RefCell::new(Pipeline::new(gpu, FORMAT, false)));
        let lit = LitRenderer::new(gpu, &pipeline, LitOptions::default()).unwrap();
        (pipeline, lit)
    }

    const KINDS: [ParticleKind; 4] = [
        ParticleKind::Splash,
        ParticleKind::Snow,
        ParticleKind::Confetti,
        ParticleKind::Sparkles,
    ];

    #[test]
    fn install_lists_the_three_entries_and_a_second_refuses() {
        let Some(gpu) = software_gpu() else {
            skip(
                "install_lists_the_three_entries_and_a_second_refuses",
                "software_gpu returned None; see the software_gpu line above",
            );
            return;
        };
        let (pipeline, mut lit) = setup(&gpu);
        let handle = ParticleFeature::install(&gpu, &pipeline, &mut lit).unwrap();
        for (expected, kind) in KINDS.into_iter().enumerate() {
            let id = handle.add(EmitterDesc::for_kind(kind)).unwrap();
            assert_eq!(id, expected as u32, "{kind:?}");
        }
        let passes = pipeline.borrow().passes();
        let sim = passes
            .iter()
            .find(|i| i.name == sim::NAME)
            .expect("particles.sim");
        assert_eq!(
            (sim.role, sim.order, sim.enabled, sim.required),
            (PassRole::Pass, sim::ORDER, true, sim::REQUIRED)
        );
        let opaque = passes
            .iter()
            .find(|i| i.name == opaque::NAME)
            .expect("particles.opaque");
        assert_eq!(
            (opaque.role, opaque.order, opaque.enabled),
            (PassRole::Contributor, opaque::ORDER, true)
        );
        let blend = passes
            .iter()
            .find(|i| i.name == blend::NAME)
            .expect("particles.blend");
        assert_eq!(
            (blend.role, blend.order, blend.enabled, blend.required),
            (PassRole::Pass, blend::ORDER, true, blend::REQUIRED)
        );
        let err = ParticleFeature::install(&gpu, &pipeline, &mut lit)
            .err()
            .expect("second install");
        assert_eq!(err.code(), "pipeline.duplicate_pass");
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn install_while_pipeline_borrowed_refuses_busy() {
        let Some(gpu) = software_gpu() else {
            skip(
                "install_while_pipeline_borrowed_refuses_busy",
                "software_gpu returned None; see the software_gpu line above",
            );
            return;
        };
        let (pipeline, mut lit) = setup(&gpu);
        let guard = pipeline.borrow();
        let err = ParticleFeature::install(&gpu, &pipeline, &mut lit)
            .err()
            .expect("borrowed pipeline");
        drop(guard);
        assert_eq!(err.code(), "pipeline.busy");
        assert!(
            pipeline
                .borrow()
                .passes()
                .iter()
                .all(|i| !i.name.starts_with("particles."))
        );
    }
}
