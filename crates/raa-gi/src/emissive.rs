//! Emissive surfaces as a per-stage GI light source (contract 'gi-emissive-sources'): the
//! [`EmissiveSources`] flags a graph switches per GI pass, the names of the WGSL overrides that
//! gate the emission ([`VOLUME_OVERRIDE`], [`SCREEN_OVERRIDE`]) and the pure helpers the passes
//! and their tests share: the override value, the emission change detector, the voxel cells and
//! word index of a screen hit, and the radiance without its emission.
//!
//! Voxel word layout (written by voxelize_kernel.wgsl, read by voxel_light.wgsl): 3 u32 per
//! voxel; voxel (level, x, y, z) starts at word (level n^3 + z n^2 + y n + x) * 3; word 2 holds
//! the emissive RGBM (`unpack4x8unorm(w).rgb * unpack4x8unorm(w).a * 128.0`).

/// Whether emissive surfaces feed the voxel volume (`pass.gi.volume`), the screen probes and
/// ReSTIR GI screen hits (`pass.gi.screen`) and the SSGI screen hits (`pass.gi.ssgi`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct EmissiveSources {
    pub volume: bool,
    pub screen: bool,
    pub ssgi: bool,
}

impl EmissiveSources {
    /// Every source on: the behaviour before the flags existed.
    pub const ALL: EmissiveSources = EmissiveSources {
        volume: true,
        screen: true,
        ssgi: true,
    };
}

impl Default for EmissiveSources {
    fn default() -> EmissiveSources {
        EmissiveSources::ALL
    }
}

/// The override of voxel_light.wgsl that scales the injected emission.
pub const VOLUME_OVERRIDE: &str = "RAA_GI_EMISSIVE";

/// The override of screen/screen_emission.wgsl that gates the emission at screen hits.
pub const SCREEN_OVERRIDE: &str = "RAA_GI_SCREEN_EMISSIVE";

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// The value passed in `compilation_options.constants` for an emission override: 1.0 when on,
/// 0.0 when off.
pub fn override_value(on: bool) -> f64 {
    if on { 1.0 } else { 0.0 }
}

/// FNV-1a 64 over the bits of every channel of every emission in order (each `u32` byte by byte,
/// least significant first), then over one byte 1 (`volume` on) or 0. A change detector, not a
/// check.
pub fn emission_signature(emissive: impl IntoIterator<Item = [f32; 3]>, volume: bool) -> u64 {
    let fold = |hash: u64, byte: u8| (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
    let hash = emissive
        .into_iter()
        .flatten()
        .flat_map(|channel| channel.to_bits().to_le_bytes())
        .fold(FNV_OFFSET, fold);
    fold(hash, u8::from(volume))
}

/// A proxy of the power the record emission injects into the volume: the sum of the Rec. 709
/// luminance of every emission when `volume` is on, 0 when it is off. It only has to fall when
/// an emissive source is switched off, dimmed or removed: the volume then clears its stored
/// radiance, so no bounce of the old emission survives in the voxel gather.
pub fn emission_power(emissive: impl IntoIterator<Item = [f32; 3]>, volume: bool) -> f32 {
    if !volume {
        return 0.0;
    }
    emissive
        .into_iter()
        .map(|e| (0.2126 * e[0] + 0.7152 * e[1] + 0.0722 * e[2]).max(0.0))
        .sum()
}

/// The 8 voxel cells around `p` for a trilinear read of a level whose grid starts at `origin`
/// with cells of `voxel` metres: base = floor((p - origin) / voxel - 0.5) per axis, item
/// `dx + 2 dy + 4 dz` = base + (dx, dy, dz).
pub fn hit_cells(origin: [f32; 3], voxel: f32, p: [f32; 3]) -> [[i32; 3]; 8] {
    let base: [i32; 3] = std::array::from_fn(|a| ((p[a] - origin[a]) / voxel - 0.5).floor() as i32);
    std::array::from_fn(|i| {
        let d = [i & 1, (i >> 1) & 1, (i >> 2) & 1];
        std::array::from_fn(|a| base[a] + d[a] as i32)
    })
}

/// The index of the emissive word (word 2) of `cell` at `level` in a volume of `resolution`
/// cells per axis, or `None` when a component lies outside [0, resolution) or the index
/// overflows `u32`.
pub fn emissive_word_index(resolution: u32, level: u32, cell: [i32; 3]) -> Option<u32> {
    let mut local = [0u32; 3];
    for (out, &c) in local.iter_mut().zip(&cell) {
        *out = u32::try_from(c).ok().filter(|&c| c < resolution)?;
    }
    let n = resolution;
    let n2 = n.checked_mul(n)?;
    let voxel = level
        .checked_mul(n2.checked_mul(n)?)?
        .checked_add(local[2].checked_mul(n2)?)?
        .checked_add(local[1].checked_mul(n)?)?
        .checked_add(local[0])?;
    voxel.checked_mul(3)?.checked_add(2)
}

/// `radiance` unchanged when `on`; otherwise `max(radiance - emission, 0)` per channel.
pub fn without_emission(radiance: [f32; 3], emission: [f32; 3], on: bool) -> [f32; 3] {
    if on {
        radiance
    } else {
        std::array::from_fn(|a| (radiance[a] - emission[a]).max(0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_all() {
        assert_eq!(EmissiveSources::default(), EmissiveSources::ALL);
    }

    /// On, then off: the volume's injected power falls to 0 (the volume clears its radiance),
    /// the signature changes (the voxel epoch restarts) and a screen hit loses the emission of
    /// the surface it hit. Expected numbers by hand: luminance of (1, 2, 4) = 0.2126 + 1.4304 +
    /// 0.2888 = 1.9318.
    #[test]
    fn turning_emission_off_after_on_injects_no_emission() {
        let emission = [[1.0, 2.0, 4.0], [0.0, 0.0, 0.0]];
        let on = emission_power(emission, true);
        assert!((on - 1.9318).abs() < 1.0e-5, "{on}");
        assert_eq!(emission_power(emission, false), 0.0);
        assert_ne!(
            emission_signature(emission, true),
            emission_signature(emission, false)
        );
        assert_eq!(override_value(false), 0.0);
        let hit = [1.5, 2.5, 4.25];
        assert_eq!(without_emission(hit, emission[0], true), hit);
        assert_eq!(without_emission(hit, emission[0], false), [0.5, 0.5, 0.25]);
        assert_eq!(
            without_emission([0.5, 0.5, 0.5], emission[0], false),
            [0.0; 3]
        );
    }
}
