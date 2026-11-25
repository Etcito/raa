//! Which probe slots a frame updates, in the order a viewer notices: probes whose influence
//! reaches the view frustum first (invalid, then young, nearest / largest on screen first), then
//! the periphery, then the probes behind the camera, and only then refreshes of converged probes
//! (stalest first, visible first). A scroll resets the age of the slots that changed probe, so
//! newly scrolled-in visible probes preempt every refresh.
//!
//! A probe keeps one age per lighting group (geometry with the sky transfer, sun with the sky
//! residual, other lights). `invalidate` lowers the age of the named groups of the named probes;
//! a job refreshes only the groups that are still young, or all of them for a periodic refresh.

/// Geometry of the probe: depth moments, relocation, classification and the sky transfer.
pub const GROUP_GEOMETRY: u32 = 1;
/// The sun's response (shadows, bounces) and the sky residual.
pub const GROUP_SUN: u32 = 2;
/// Local lights and emission.
pub const GROUP_OTHER: u32 = 4;
pub const GROUP_ALL: u32 = 7;

/// The camera of the frame (eye and view-projection) and its sun.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DdgiView {
    pub eye: [f32; 3],
    /// Column-major view-projection (raa-math `Mat4::cols`, depth 0..1).
    pub view_proj: [[f32; 4]; 4],
    /// Travel direction of the sun light (unit) and its colour x intensity.
    pub sun_direction: [f32; 3],
    pub sun_radiance: [f32; 3],
}

impl DdgiView {
    /// The six frustum planes (normalized, inside >= 0), Gribb-Hartmann on a 0..1 depth range.
    fn planes(&self) -> [[f32; 4]; 6] {
        let m = &self.view_proj;
        let row = |r: usize| [m[0][r], m[1][r], m[2][r], m[3][r]];
        let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
        let add = |a: [f32; 4], b: [f32; 4]| std::array::from_fn::<f32, 4, _>(|i| a[i] + b[i]);
        let sub = |a: [f32; 4], b: [f32; 4]| std::array::from_fn::<f32, 4, _>(|i| a[i] - b[i]);
        [
            add(r3, r0),
            sub(r3, r0),
            add(r3, r1),
            sub(r3, r1),
            r2,
            sub(r3, r2),
        ]
        .map(|p| {
            let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt().max(1e-12);
            p.map(|x| x / len)
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Visibility {
    /// Probes whose influence (one cell) reaches the frustum.
    pub visible: u32,
    /// Of those, probes updated at least once since they entered the volume.
    pub valid: u32,
    /// Of those, probes past the progressive phase (`converge_age` updates).
    pub converged: u32,
}

#[derive(Default)]
pub(super) struct Schedule {
    pub(super) base: Option<[i32; 3]>,
    counts: [u32; 3],
    /// Updates each slot's current probe received (saturating): geometry group.
    age: Vec<u16>,
    /// Updates of the sun and of the other group since their last invalidation (saturating).
    age_sun: Vec<u8>,
    age_other: Vec<u8>,
    /// The next update of the other group replaces the probe's light mask.
    mask_reset: Vec<bool>,
    /// Frame of each slot's last update.
    last: Vec<u32>,
    frame: u32,
    keys: Vec<(u64, u32)>,
}

const CLASS_SHIFT: u32 = 40;

impl Schedule {
    pub(super) fn reset(&mut self, counts: [u32; 3]) {
        let total = counts.iter().product::<u32>() as usize;
        *self = Self {
            counts,
            age: vec![0; total],
            age_sun: vec![0; total],
            age_other: vec![0; total],
            mask_reset: vec![false; total],
            last: vec![0; total],
            ..Self::default()
        };
    }

    pub(super) fn slot(&self, g: [i32; 3]) -> u32 {
        let n = self.counts.map(|n| n as i32);
        let s: [u32; 3] = std::array::from_fn(|i| g[i].rem_euclid(n[i]) as u32);
        s[0] + self.counts[0] * (s[1] + self.counts[1] * s[2])
    }

    #[cfg(test)]
    fn age(&self, slot: u32) -> u16 {
        self.age[slot as usize]
    }

    /// Moves the volume to `base`; every slot whose world coordinate changed becomes fresh.
    /// Returns how many did.
    pub(super) fn scroll(&mut self, base: [i32; 3]) -> u32 {
        let Some(old) = self.base.replace(base) else {
            return 0;
        };
        if old == base {
            return 0;
        }
        let n = self.counts.map(|n| n as i32);
        let mut recycled = 0;
        for z in 0..n[2] {
            for y in 0..n[1] {
                for x in 0..n[0] {
                    let g = [base[0] + x, base[1] + y, base[2] + z];
                    if !(0..3).all(|i| g[i] >= old[i] && g[i] < old[i] + n[i]) {
                        let slot = self.slot(g) as usize;
                        self.age[slot] = 0;
                        self.age_sun[slot] = 0;
                        self.age_other[slot] = 0;
                        self.mask_reset[slot] = true;
                        recycled += 1;
                    }
                }
            }
        }
        recycled
    }

    /// Lowers the age of `groups` of one slot to at most `cap` updates (the geometry group keeps
    /// at least one: zero means a probe that was never traced). Returns whether an age dropped.
    pub(super) fn invalidate_slot(
        &mut self,
        slot: u32,
        groups: u32,
        cap: u8,
        reset_mask: bool,
    ) -> bool {
        let slot = slot as usize;
        let mut dropped = false;
        if groups & GROUP_GEOMETRY != 0 && self.age[slot] > u16::from(cap.max(1)) {
            self.age[slot] = u16::from(cap.max(1));
            dropped = true;
        }
        if groups & GROUP_SUN != 0 && self.age_sun[slot] > cap {
            self.age_sun[slot] = cap;
            dropped = true;
        }
        if groups & GROUP_OTHER != 0 {
            if self.age_other[slot] > cap {
                self.age_other[slot] = cap;
                dropped = true;
            }
            self.mask_reset[slot] |= reset_mask;
        }
        dropped
    }

    /// `invalidate_slot` on every slot; returns how many dropped.
    pub(super) fn invalidate_all(&mut self, groups: u32, cap: u8) -> u32 {
        (0..self.age.len() as u32)
            .filter(|&slot| self.invalidate_slot(slot, groups, cap, false))
            .count() as u32
    }

    /// Calls `visit(slot, probe grid position in metres)` for every probe of the volume.
    pub(super) fn for_each_probe(&self, spacing: f32, mut visit: impl FnMut(u32, [f32; 3])) {
        let Some(base) = self.base else {
            return;
        };
        let n = self.counts;
        for z in 0..n[2] {
            for y in 0..n[1] {
                for x in 0..n[0] {
                    let g = [base[0] + x as i32, base[1] + y as i32, base[2] + z as i32];
                    visit(self.slot(g), g.map(|c| c as f32 * spacing));
                }
            }
        }
    }

    fn effective_age(&self, slot: usize) -> u16 {
        self.age[slot]
            .min(u16::from(self.age_sun[slot]))
            .min(u16::from(self.age_other[slot]))
    }

    /// Picks `budget` slots (or `burst` while a visible probe is still invalid or young) and
    /// appends two words per job (`slot | geometry age << 16`, then
    /// `sun age | other age << 8 | groups << 16 | replace mask << 24`), ageing the refreshed
    /// groups of the picked slots: the young groups, or all of them when none is. Without a view
    /// every probe counts as visible and the order is by age then distance to the volume centre.
    /// The third value counts the refreshed groups (geometry, sun, other).
    pub(super) fn plan(
        &mut self,
        view: Option<&DdgiView>,
        spacing: f32,
        converge_age: u16,
        budget: u32,
        burst: u32,
        out: &mut Vec<u32>,
    ) -> (Visibility, bool, [u32; 3]) {
        out.clear();
        self.frame = self.frame.wrapping_add(1);
        let mut refreshed = [0u32; 3];
        let Some(base) = self.base else {
            return (Visibility::default(), false, refreshed);
        };
        let n = self.counts;
        let planes = view.map(DdgiView::planes);
        let centre: [f32; 3] =
            std::array::from_fn(|i| (base[i] as f32 + 0.5 * (n[i] - 1) as f32) * spacing);
        let eye = view.map_or(centre, |v| v.eye);
        let reach = 1.75 * spacing;
        let mut seen = Visibility::default();
        self.keys.clear();
        for z in 0..n[2] {
            for y in 0..n[1] {
                for x in 0..n[0] {
                    let g = [base[0] + x as i32, base[1] + y as i32, base[2] + z as i32];
                    let p = g.map(|c| c as f32 * spacing);
                    // 0 visible, 1 periphery, 2 elsewhere
                    let zone = planes.map_or(0u64, |planes| {
                        let worst = planes
                            .iter()
                            .map(|q| q[0] * p[0] + q[1] * p[1] + q[2] * p[2] + q[3])
                            .fold(f32::INFINITY, f32::min);
                        if worst >= -reach {
                            0
                        } else if worst >= -3.0 * reach {
                            1
                        } else {
                            2
                        }
                    });
                    let slot = self.slot(g);
                    let age = self.effective_age(slot as usize);
                    if zone == 0 {
                        seen.visible += 1;
                    }
                    let d = (0..3).map(|i| (p[i] - eye[i]).powi(2)).sum::<f32>().sqrt();
                    let near = ((d * 100.0) as u64).min((1 << 20) - 1);
                    let (class, metric) = if age >= converge_age {
                        let stale = u64::from(self.frame.wrapping_sub(self.last[slot as usize]));
                        (6 + zone.min(1), (1u64 << 32).saturating_sub(stale))
                    } else if age == 0 {
                        (zone * 2, near)
                    } else {
                        (zone * 2 + 1, (u64::from(age) << 20) | near)
                    };
                    self.keys.push(((class << CLASS_SHIFT) | metric, slot));
                }
            }
        }
        let urgent = self.keys.iter().any(|&(key, _)| key >> CLASS_SHIFT <= 1);
        let take = (if urgent { burst } else { budget }).min(self.keys.len() as u32) as usize;
        if take > 0 && take < self.keys.len() {
            self.keys.select_nth_unstable(take - 1);
        }
        self.keys[..take].sort_unstable();
        for &(_, slot) in &self.keys[..take] {
            let i = slot as usize;
            let (age, sun, other) = (self.age[i], self.age_sun[i], self.age_other[i]);
            let limit = converge_age.min(255);
            let mut groups = u32::from(age < limit) * GROUP_GEOMETRY
                | u32::from(u16::from(sun) < limit) * GROUP_SUN
                | u32::from(u16::from(other) < limit) * GROUP_OTHER;
            if groups == 0 || age == 0 {
                groups = GROUP_ALL;
            }
            let reset = groups & GROUP_OTHER != 0 && std::mem::take(&mut self.mask_reset[i]);
            out.push(slot | (u32::from(age) << 16));
            out.push(
                u32::from(sun)
                    | (u32::from(other) << 8)
                    | (groups << 16)
                    | (u32::from(reset) << 24),
            );
            if groups & GROUP_GEOMETRY != 0 {
                self.age[i] = age.saturating_add(1);
                refreshed[0] += 1;
            }
            if groups & GROUP_SUN != 0 {
                self.age_sun[i] = sun.saturating_add(1);
                refreshed[1] += 1;
            }
            if groups & GROUP_OTHER != 0 {
                self.age_other[i] = other.saturating_add(1);
                refreshed[2] += 1;
            }
            self.last[i] = self.frame;
        }
        // Visibility after this frame's updates.
        for &(key, slot) in &self.keys {
            let class = key >> CLASS_SHIFT;
            if class == 0 || class == 1 || class == 6 {
                seen.valid += u32::from(self.age[slot as usize] > 0);
                seen.converged += u32::from(self.effective_age(slot as usize) >= converge_age);
            }
        }
        (seen, urgent, refreshed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn looking_down_x() -> DdgiView {
        // Camera at the origin looking along +x, 90 degree fov, square, depth 0..1 (0.1..100).
        let (near, far) = (0.1f32, 100.0f32);
        let a = far / (far - near);
        // view: x_view = -z_world, y_view = y, z_view(forward, -) = -x_world  => right-handed look +x
        // clip = P * V, columns
        let view = [
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let proj = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -a, -1.0],
            [0.0, 0.0, -a * near, 0.0],
        ];
        let mut vp = [[0.0f32; 4]; 4];
        for c in 0..4 {
            for r in 0..4 {
                vp[c][r] = (0..4).map(|k| proj[k][r] * view[c][k]).sum();
            }
        }
        DdgiView {
            eye: [0.0; 3],
            view_proj: vp,
            sun_direction: [0.0, -1.0, 0.0],
            sun_radiance: [1.0; 3],
        }
    }

    #[test]
    fn scroll_resets_only_the_planes_that_left() {
        let mut s = Schedule::default();
        s.reset([4, 2, 4]);
        assert_eq!(s.scroll([0, 0, 0]), 0);
        let mut jobs = Vec::new();
        s.plan(None, 1.0, 4, 32, 32, &mut jobs);
        assert!((0..32).all(|slot| s.age(slot) == 1));
        assert_eq!(s.scroll([1, 0, 0]), 8);
        let fresh: Vec<u32> = (0..32).filter(|&slot| s.age(slot) == 0).collect();
        let mut want: Vec<u32> = (0..2)
            .flat_map(|y| (0..4).map(move |z| (y, z)))
            .map(|(y, z)| s.slot([4, y, z]))
            .collect();
        want.sort_unstable();
        assert_eq!(fresh, want);
        // The fresh slots go first and carry age 0 in their job.
        s.plan(None, 1.0, 4, 8, 8, &mut jobs);
        let mut picked: Vec<u32> = jobs.chunks(2).map(|j| j[0] & 0xffff).collect();
        picked.sort_unstable();
        assert_eq!(picked, want);
        assert!(
            jobs.chunks(2)
                .all(|j| j[0] >> 16 == 0 && (j[1] >> 16) & 7 == GROUP_ALL)
        );
        assert!(
            jobs.chunks(2).all(|j| j[1] >> 24 == 1),
            "a recycled slot replaces its light mask"
        );
    }

    #[test]
    fn an_invalidated_group_is_refreshed_alone_and_first() {
        let mut s = Schedule::default();
        s.reset([4, 1, 1]);
        s.scroll([0, 0, 0]);
        let mut jobs = Vec::new();
        // Converge every group of every probe (converge age 3).
        for _ in 0..3 {
            s.plan(None, 1.0, 3, 4, 4, &mut jobs);
        }
        let (seen, burst, _) = s.plan(None, 1.0, 3, 1, 4, &mut jobs);
        assert!(!burst);
        assert_eq!(seen.converged, 4);
        assert_eq!(
            (jobs[1] >> 16) & 7,
            GROUP_ALL,
            "a periodic refresh traces every group"
        );
        // One probe loses its other group, every probe its sun group.
        assert!(s.invalidate_slot(2, GROUP_OTHER, 1, true));
        assert!(
            !s.invalidate_slot(2, GROUP_OTHER, 1, false),
            "already that young"
        );
        assert_eq!(s.invalidate_all(GROUP_SUN, 2), 4);
        let (seen, burst, refreshed) = s.plan(None, 1.0, 3, 1, 4, &mut jobs);
        assert!(burst);
        assert_eq!(
            seen.converged, 3,
            "slot 2 still lacks one update of its other group"
        );
        assert_eq!(refreshed, [0, 4, 1]);
        // Slot 2 is the youngest: first, sun and other groups, geometry kept.
        assert_eq!(jobs[0] & 0xffff, 2);
        assert_eq!((jobs[1] >> 16) & 7, GROUP_SUN | GROUP_OTHER);
        assert_eq!(jobs[1] & 0xffff, 2 | (1 << 8));
        assert_eq!(jobs[1] >> 24, 1);
        assert!(
            jobs.chunks(2)
                .skip(1)
                .all(|j| (j[1] >> 16) & 7 == GROUP_SUN)
        );
        let (seen, burst, refreshed) = s.plan(None, 1.0, 3, 1, 4, &mut jobs);
        // The burst fills up with periodic refreshes of the three converged probes.
        assert_eq!(refreshed, [3, 3, 4]);
        assert_eq!(jobs[0] & 0xffff, 2);
        assert_eq!((jobs[1] >> 16) & 7, GROUP_OTHER);
        assert!(burst);
        let (seen2, burst2, _) = s.plan(None, 1.0, 3, 1, 4, &mut jobs);
        assert!(!burst2);
        assert_eq!((seen.converged, seen2.converged), (4, 4));
    }

    #[test]
    fn visible_probes_go_first_nearest_first_then_burst_ends() {
        let view = looking_down_x();
        let mut s = Schedule::default();
        s.reset([8, 1, 1]);
        // Probes at x = -4..3 on the camera axis: x > 0 are in front (x = 0 sits on the eye).
        s.scroll([-4, 0, 0]);
        let mut jobs = Vec::new();
        let (seen, burst, _) = s.plan(Some(&view), 1.0, 2, 2, 3, &mut jobs);
        assert!(burst);
        let xs: Vec<i32> = jobs
            .chunks(2)
            .map(|j| (j[0] & 0xffff) as i32)
            .map(|slot| (-4..4).find(|&x| s.slot([x, 0, 0]) == slot as u32).unwrap())
            .collect();
        // Influence reaches 1.75 cells behind the eye: x = 0 and -1 count as visible too,
        // nearest first (x = 1 and x = -1 tie).
        assert_eq!(xs[0], 0);
        let mut tied = xs[1..].to_vec();
        tied.sort_unstable();
        assert_eq!(tied, vec![-1, 1]);
        assert_eq!(seen.visible, 5);
        // Keep planning until every visible probe has converged: the budget drops to 2.
        let mut frames = 0;
        loop {
            let (seen, burst, _) = s.plan(Some(&view), 1.0, 2, 2, 3, &mut jobs);
            frames += 1;
            if !burst {
                assert_eq!(seen.converged, seen.visible);
                assert_eq!(jobs.len(), 4);
                break;
            }
            assert_eq!(jobs.len(), 6);
            assert!(frames < 10);
        }
    }
}
