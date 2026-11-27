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

/// Where and how often the forward evaluates the sun visibility s in "layers-interlaced".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ShadowRate {
    /// At the phase pixel of an R frame, like AO and irradiance (1/2 resolution, 30 Hz).
    #[default]
    Phase,
    /// At every pixel of an R frame (full resolution, 30 Hz). Word bit 6.
    Real,
    /// At every pixel of every frame (full resolution, 60 Hz). A G frame samples the cascade
    /// maps its R frame rendered (raa-shadow keeps their matrices) and the contact shadow of the
    /// frame (`shadow.contact` runs on G frames). Word bits 6 and 7.
    Every,
}

/// Where and how often the forward evaluates the probe irradiance E (with its weight w).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IrradianceRate {
    /// At the phase pixel of an R frame (1/2 resolution, 30 Hz).
    #[default]
    Phase30,
    /// At the phase pixel of every frame (1/2 resolution, 60 Hz interlaced): a G frame samples
    /// [`LightingInterlace::phase`], the R phase two lighting frames ahead, so any four
    /// consecutive frames starting on an R frame visit the four pixels of a block. The DDGI
    /// screen resolve runs on G frames. Word bit 8.
    Phase60,
    /// At every pixel of every frame (full resolution, 60 Hz): the DDGI screen resolve runs at
    /// scale 1 on every frame. Word bit 9.
    Full,
}

/// The signal rates of forward "layers-interlaced", set on the lit renderer by
/// pass.lighting.reconstruct (`LitRenderer::set_interlace_signals`). AO stays at the phase
/// pixel of R frames in every combination.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InterlaceSignals {
    pub shadow: ShadowRate,
    pub irradiance: IrradianceRate,
}

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
    /// The signal rates of the mode (default with the mode off).
    pub signals: InterlaceSignals,
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

    /// [`new`](Self::new) with the given signal rates (ignored with the mode off).
    pub fn with_signals(enabled: bool, frame: u64, signals: InterlaceSignals) -> LightingInterlace {
        LightingInterlace {
            signals: if enabled { signals } else { InterlaceSignals::default() },
            ..LightingInterlace::new(enabled, frame)
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

    /// The phase (px, py) the frame samples: lighting frame i = n / 2 uses phase i % 4; a G
    /// frame with [`IrradianceRate::Phase60`] samples E at phase (i + 2) % 4, so the sequence
    /// over n is (0,0), (1,0), (1,1), (0,1), (1,0), (0,0), (0,1), (1,1).
    pub fn phase(&self) -> [u32; 2] {
        let i = self.frame / 2 + if self.irradiance_sampled_on_g() { 2 } else { 0 };
        INTERLACE_PHASES[(i % 4) as usize]
    }

    /// A G frame of the mode evaluates E at its phase pixels ([`IrradianceRate::Phase60`]).
    fn irradiance_sampled_on_g(&self) -> bool {
        self.generated() && self.signals.irradiance == IrradianceRate::Phase60
    }

    /// The frame evaluates s at every pixel.
    pub fn shadow_everywhere(&self) -> bool {
        self.enabled
            && match self.signals.shadow {
                ShadowRate::Phase => false,
                ShadowRate::Real => self.real(),
                ShadowRate::Every => true,
            }
    }

    /// The frame evaluates E: on R frames always, on G frames with Phase60 / Full.
    pub fn irradiance_sampled(&self) -> bool {
        self.enabled && (self.real() || self.signals.irradiance != IrradianceRate::Phase30)
    }

    /// The `FrameGpu.temporal.w` word (`RgFrame.interlace` in compute nodes): 0 with the mode
    /// off, else bit 0 = 1, bit 1 = R frame, bit 2 = px, bit 3 = py, bits 4..5 = (n / 2) % 4,
    /// bit 6 = s at every pixel of an R frame, bit 7 = s at every pixel of a G frame too, bit 8 =
    /// E at the phase pixel of a G frame too, bit 9 = E at every pixel of every frame.
    pub fn word(&self) -> u32 {
        if !self.enabled {
            return 0;
        }
        let [px, py] = self.phase();
        1 | (u32::from(self.real()) << 1)
            | (px << 2)
            | (py << 3)
            | ((((self.frame / 2) % 4) as u32) << 4)
            | (u32::from(self.signals.shadow != ShadowRate::Phase) << 6)
            | (u32::from(self.signals.shadow == ShadowRate::Every) << 7)
            | (u32::from(self.signals.irradiance == IrradianceRate::Phase60) << 8)
            | (u32::from(self.signals.irradiance == IrradianceRate::Full) << 9)
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

    #[test]
    fn signal_rates_set_word_bits_and_g_phase() {
        let rank1 = InterlaceSignals {
            shadow: ShadowRate::Every,
            irradiance: IrradianceRate::Phase60,
        };
        let phases: Vec<_> = (0..8)
            .map(|n| LightingInterlace::with_signals(true, n, rank1).phase())
            .collect();
        assert_eq!(
            phases,
            [[0, 0], [1, 0], [1, 1], [0, 1], [1, 0], [0, 0], [0, 1], [1, 1]]
        );
        // Four consecutive frames from an R frame visit each pixel of a block once.
        for start in [0usize, 2, 4, 6] {
            let mut seen: Vec<_> = (start..start + 4).map(|n| phases[n % 8]).collect();
            seen.sort();
            assert_eq!(seen, [[0, 0], [0, 1], [1, 0], [1, 1]]);
        }
        // n = 5 (G, i = 2): E phase (i + 2) % 4 = 0 -> (0, 0); bits 6, 7, 8.
        assert_eq!(
            LightingInterlace::with_signals(true, 5, rank1).word(),
            1 | (2 << 4) | (1 << 6) | (1 << 7) | (1 << 8)
        );
        let g = LightingInterlace::with_signals(true, 5, rank1);
        assert!(g.shadow_everywhere() && g.irradiance_sampled());
        let rank2 = InterlaceSignals {
            irradiance: IrradianceRate::Full,
            ..rank1
        };
        let g2 = LightingInterlace::with_signals(true, 5, rank2);
        assert_eq!(g2.phase(), [1, 0]);
        assert_eq!(g2.word() >> 6, 0b1011);
        let real = InterlaceSignals {
            shadow: ShadowRate::Real,
            irradiance: IrradianceRate::Phase30,
        };
        assert!(!LightingInterlace::with_signals(true, 5, real).shadow_everywhere());
        assert!(LightingInterlace::with_signals(true, 4, real).shadow_everywhere());
        assert!(!LightingInterlace::with_signals(true, 5, real).irradiance_sampled());
        // Mode off: no signal bits.
        assert_eq!(LightingInterlace::with_signals(false, 5, rank1).word(), 0);
    }
}
