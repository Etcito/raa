//! Which probe slots a frame updates, in the order a viewer notices: probes whose influence
//! reaches the view frustum first (invalid, then young, nearest / largest on screen first), then
//! the periphery, then the probes behind the camera, and only then refreshes of converged probes
//! (stalest first, visible first). A scroll resets the age of the slots that changed probe, so
//! newly scrolled-in visible probes preempt every refresh.

/// The camera of the frame: eye and view-projection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DdgiView {
    pub eye: [f32; 3],
    /// Column-major view-projection (raa-math `Mat4::cols`, depth 0..1).
    pub view_proj: [[f32; 4]; 4],
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
    /// Updates each slot's current probe received (saturating).
    age: Vec<u16>,
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
                        recycled += 1;
                    }
                }
            }
        }
        recycled
    }

    /// Picks `budget` slots (or `burst` while a visible probe is still invalid or young) and
    /// appends `slot | age << 16` jobs, ageing the picked slots. Without a view every probe
    /// counts as visible and the order is by age then distance to the volume centre.
    pub(super) fn plan(
        &mut self,
        view: Option<&DdgiView>,
        spacing: f32,
        converge_age: u16,
        budget: u32,
        burst: u32,
        out: &mut Vec<u32>,
    ) -> (Visibility, bool) {
        out.clear();
        self.frame = self.frame.wrapping_add(1);
        let Some(base) = self.base else {
            return (Visibility::default(), false);
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
                    let age = self.age[slot as usize];
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
            let age = &mut self.age[slot as usize];
            out.push(slot | (u32::from(*age) << 16));
            *age = age.saturating_add(1);
            self.last[slot as usize] = self.frame;
        }
        // Visibility after this frame's updates.
        for &(key, slot) in &self.keys {
            let class = key >> CLASS_SHIFT;
            if class == 0 || class == 1 || class == 6 {
                let age = self.age[slot as usize];
                seen.valid += u32::from(age > 0);
                seen.converged += u32::from(age >= converge_age);
            }
        }
        (seen, urgent)
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
        let mut picked: Vec<u32> = jobs.iter().map(|j| j & 0xffff).collect();
        picked.sort_unstable();
        assert_eq!(picked, want);
        assert!(jobs.iter().all(|j| j >> 16 == 0));
    }

    #[test]
    fn visible_probes_go_first_nearest_first_then_burst_ends() {
        let view = looking_down_x();
        let mut s = Schedule::default();
        s.reset([8, 1, 1]);
        // Probes at x = -4..3 on the camera axis: x > 0 are in front (x = 0 sits on the eye).
        s.scroll([-4, 0, 0]);
        let mut jobs = Vec::new();
        let (seen, burst) = s.plan(Some(&view), 1.0, 2, 2, 3, &mut jobs);
        assert!(burst);
        let xs: Vec<i32> = jobs
            .iter()
            .map(|j| (j & 0xffff) as i32)
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
            let (seen, burst) = s.plan(Some(&view), 1.0, 2, 2, 3, &mut jobs);
            frames += 1;
            if !burst {
                assert_eq!(seen.converged, seen.visible);
                assert_eq!(jobs.len(), 2);
                break;
            }
            assert_eq!(jobs.len(), 3);
            assert!(frames < 10);
        }
    }
}
