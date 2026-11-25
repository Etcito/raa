//! Order angular columns for capture priority; residency never depends on the view.

use super::{CellKey, SphericalGrid, grid::dot};

pub(super) struct ViewOrder {
    forward: [f64; 3],
    half_fov_cos: f64,
    angular_rank: Vec<u32>,
    /// Capture class per column: 0 inside the view cone, 1 periphery, 2 behind.
    class: Vec<u8>,
}

/// Columns within this many radians outside the view cone are periphery: a turn of
/// the head brings them on screen before columns behind the camera.
const PERIPHERY_MARGIN: f64 = 0.6;

impl ViewOrder {
    pub(super) fn new(
        grid: &SphericalGrid,
        previous: Option<&Self>,
        forward: [f32; 3],
        half_fov_cos: f32,
    ) -> Option<Self> {
        let forward = forward.map(f64::from);
        let length = dot(forward, forward).sqrt();
        if !length.is_finite() || length == 0.0 || !(0.0..=1.0).contains(&half_fov_cos) {
            return None;
        }
        let forward = forward.map(|v| v / length);
        let half_fov_cos = f64::from(half_fov_cos);
        // Ignore small angular changes relative to a grid subdivision. This
        // retains scan progress during view jitter or slowly changing heading.
        let tolerance = 0.25 / f64::from(grid.angular_resolution());
        if previous.is_some_and(|old| {
            dot(old.forward, forward) >= tolerance.cos()
                && (old.half_fov_cos.acos() - half_fov_cos.acos()).abs() < tolerance
        }) {
            return None;
        }

        let fov_sin = (1.0 - half_fov_cos * half_fov_cos).sqrt();
        let periphery = (half_fov_cos.acos() + PERIPHERY_MARGIN).min(std::f64::consts::PI);
        let (periphery_cos, periphery_sin) = (periphery.cos(), periphery.sin());
        let columns = grid.cells_per_shell();
        let mut class = vec![2u8; columns as usize];
        let mut scores: Vec<_> = (0..columns)
            .map(|index| {
                let (center, tile_cos) = grid.column_cap(grid.key(index));
                let alignment = dot(forward, center);
                let tile_sin = (1.0 - tile_cos * tile_cos).max(0.0).sqrt();
                let visible = alignment >= half_fov_cos * tile_cos - fov_sin * tile_sin;
                let near = alignment >= periphery_cos * tile_cos - periphery_sin * tile_sin;
                class[index as usize] = if visible {
                    0
                } else if near {
                    1
                } else {
                    2
                };
                (index, visible, alignment)
            })
            .collect();
        scores.sort_unstable_by(|a, b| {
            b.1.cmp(&a.1)
                .then_with(|| b.2.total_cmp(&a.2))
                .then_with(|| a.0.cmp(&b.0))
        });
        let mut angular_rank = vec![0; columns as usize];
        for (rank, score) in scores.into_iter().enumerate() {
            angular_rank[score.0 as usize] = rank as u32;
        }
        Some(Self {
            forward,
            half_fov_cos,
            angular_rank,
            class,
        })
    }

    /// Capture priority of a cell: view class (inside, periphery, behind), then shell
    /// (nearer shells cover more of the screen), then closeness to the view centre.
    pub(super) fn capture_rank(&self, grid: &SphericalGrid, key: CellKey) -> (u8, u16, u32) {
        let index = (grid.ordinal(key) % grid.cells_per_shell()) as usize;
        (self.class[index], key.shell, self.angular_rank[index])
    }
}
