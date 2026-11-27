//! The 30 Hz half-resolution lighting cadence of forward lightingOutput "layers-interlaced".
//!
//! raa-lit writes one [`LightingInterlace`] per scene frame into the
//! [`LightingResources`](crate::LightingResources) (`set_interlace`) before the pipeline
//! prepares, and the passes that only feed the lighting signals read it in their record
//! (`LightingResources::interlace`): on a G frame they record nothing and leave their (bound,
//! published) outputs as they were.

/// Phase (px, py) of lighting frame `i = n / 2`, indexed by `i % 4`: the pixel of each 2x2
/// block whose signals an R frame evaluates.
pub const INTERLACE_PHASES: [[u32; 2]; 4] = [[0, 0], [1, 1], [1, 0], [0, 1]];

/// The interlace state of one frame. `Default` is the mode off (nothing skipped).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LightingInterlace {
    /// The forward pass is in lightingOutput "layers-interlaced" (msaa 1).
    pub enabled: bool,
    /// The real-frame counter n of the frame (the lit scene frame index, which advances only
    /// on submitted frames).
    pub frame: u64,
    /// G frame: `shadow.csm` renders its hero layers only (no cascade).
    pub skip_cascades: bool,
    /// G frame: the primary `shadow.contact` records nothing.
    pub skip_contact: bool,
    /// G frame: the primary `ao.gtao` records nothing.
    pub skip_ao: bool,
    /// G frame: `gi.ddgi` skips its screen resolve (the probe update still runs).
    pub skip_gi_resolve: bool,
}

impl LightingInterlace {
    /// The state of frame `n` in the given mode, nothing skipped yet (raa-lit sets the
    /// `skip_*` flags of a G frame from the enabled consumers).
    pub fn new(enabled: bool, frame: u64) -> LightingInterlace {
        LightingInterlace {
            enabled,
            frame,
            ..LightingInterlace::default()
        }
    }

    /// R frame (n even): the forward evaluates the signals at the phase pixels.
    pub fn real(&self) -> bool {
        self.frame & 1 == 0
    }

    /// G frame of the mode: no signal is evaluated.
    pub fn generated(&self) -> bool {
        self.enabled && !self.real()
    }

    /// The phase (px, py) of lighting frame n / 2.
    pub fn phase(&self) -> [u32; 2] {
        INTERLACE_PHASES[((self.frame / 2) % 4) as usize]
    }

    /// The `FrameGpu.temporal.w` word (`RgFrame.interlace` in compute nodes): 0 with the mode
    /// off, else bit 0 = 1, bit 1 = R frame, bit 2 = px, bit 3 = py, bits 4.. = (n / 2) % 4.
    pub fn word(&self) -> u32 {
        if !self.enabled {
            return 0;
        }
        let [px, py] = self.phase();
        1 | (u32::from(self.real()) << 1)
            | (px << 2)
            | (py << 3)
            | ((((self.frame / 2) % 4) as u32) << 4)
    }

    /// The pipeline entries this state skips, in pipeline order.
    pub fn skipped(&self) -> impl Iterator<Item = &'static str> {
        [
            (self.skip_cascades, "shadow.csm"),
            (self.skip_ao, "ao.gtao"),
            (self.skip_contact, "shadow.contact"),
            (self.skip_gi_resolve, "gi.ddgi"),
        ]
        .into_iter()
        .filter_map(|(skip, name)| skip.then_some(name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phases_cycle_over_eight_frames() {
        let phases: Vec<_> = (0..10)
            .map(|n| LightingInterlace::new(true, n).phase())
            .collect();
        assert_eq!(
            phases,
            [
                [0, 0],
                [0, 0],
                [1, 1],
                [1, 1],
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
                [0, 0],
                [0, 0]
            ]
        );
        assert!(LightingInterlace::new(true, 4).real());
        assert!(LightingInterlace::new(true, 5).generated());
        assert!(!LightingInterlace::new(false, 5).generated());
    }

    #[test]
    fn word_packs_mode_class_and_phase() {
        assert_eq!(LightingInterlace::new(false, 3).word(), 0);
        assert_eq!(LightingInterlace::new(true, 0).word(), 0b11);
        // n = 2: R, i = 1, phase (1, 1).
        assert_eq!(
            LightingInterlace::new(true, 2).word(),
            1 | 2 | 4 | 8 | (1 << 4)
        );
        // n = 5: G, i = 2, phase (1, 0).
        assert_eq!(LightingInterlace::new(true, 5).word(), 1 | 4 | (2 << 4));
    }
}
