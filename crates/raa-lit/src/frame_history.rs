//! Per-viewport camera history and S10 Halton jitter state (RAA-FRAME); four slots are
//! reused per invocation and `finish(false)` invalidates all. Jitter is zero without a
//! consumer.
//!
//! Handoff: `crates/raa-lit/src/frame_history.rs` exports `pub(crate) fn
//! FrameHistory::prepare(&mut self, view: usize, view_proj: Mat4, unjittered_view_proj:
//! Mat4, size: (u32, u32), jitter_consumer: bool, discontinuity: bool) -> CameraHistory`.
//! The renderer share owns one `FrameHistory` in `LitRenderer`, calls it from
//! `begin_scene_frame`/`prepare_scene_view`/`submitted_scene_view`/`finish_scene_frame` with
//! viewport indices 0..3 and `M2-SUBMISSION-OBSERVATIONS` masks, builds `view_proj` with
//! `frame_jitter` from the same file and passes the record as `FrameParams`
//! `previous_view_proj`, `camera_jitter_ndc`, `previous_jitter_ndc` and `history_valid`.
#![allow(dead_code)] // Staged before the renderer share lands; see Handoff.
use raa_math::Mat4;

pub(crate) const MAX_VIEWS: usize = 4;
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct CameraHistory {
    pub previous_view_proj: Mat4,
    pub unjittered_view_proj: Mat4,
    pub current_jitter: [f32; 2],
    pub previous_jitter: [f32; 2],
    pub history_valid: bool,
    pub frame_index: u64,
}

#[derive(Copy, Clone, Debug)]
struct Pending(Mat4, [f32; 2], (u32, u32), bool);

#[derive(Copy, Clone, Debug, Default)]
struct ViewSlot {
    committed: Option<Pending>,
    pending: Option<Pending>,
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct FrameHistory {
    slots: [ViewSlot; MAX_VIEWS],
    frame_index: u64,
}

impl FrameHistory {
    #[allow(clippy::new_without_default)]
    pub(crate) fn new() -> Self {
        FrameHistory {
            slots: [ViewSlot::default(); MAX_VIEWS],
            frame_index: 0,
        }
    }

    /// Opens scene invocation `frame_index`; prior pending state is discarded.
    pub(crate) fn begin(&mut self, frame_index: u64) {
        self.frame_index = frame_index;
        for slot in &mut self.slots {
            slot.pending = None;
        }
    }

    /// Returns this view's history and records pending state; committed state is untouched.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare(
        &mut self,
        view: usize,
        view_proj: Mat4,
        unjittered_view_proj: Mat4,
        size: (u32, u32),
        jitter_consumer: bool,
        discontinuity: bool,
    ) -> CameraHistory {
        let jitter = frame_jitter(self.frame_index, size, jitter_consumer);
        let slot = &mut self.slots[view % MAX_VIEWS];
        let valid = !discontinuity
            && slot
                .committed
                .is_some_and(|c| c.2 == size && c.3 == jitter_consumer);
        let mut history = CameraHistory {
            previous_view_proj: view_proj,
            unjittered_view_proj,
            current_jitter: jitter,
            previous_jitter: jitter,
            history_valid: false,
            frame_index: self.frame_index,
        };
        if let Some(Pending(proj, previous_jitter, _, _)) = slot.committed.filter(|_| valid) {
            history.previous_view_proj = proj;
            history.previous_jitter = previous_jitter;
            history.history_valid = true;
        }
        slot.pending = Some(Pending(view_proj, jitter, size, jitter_consumer));
        history
    }

    /// Commits only `view`; other views keep their last submitted history.
    pub(crate) fn submitted(&mut self, view: usize) {
        let slot = &mut self.slots[view % MAX_VIEWS];
        slot.committed = slot.pending.take();
    }

    /// Drops pending state; an incomplete invocation invalidates history before next `begin`.
    pub(crate) fn finish(&mut self, complete: bool) {
        if complete {
            for slot in &mut self.slots {
                slot.pending = None;
            }
        } else {
            self.invalidate();
        }
    }

    pub(crate) fn invalidate(&mut self) {
        self.slots = [ViewSlot::default(); MAX_VIEWS];
    }
}

fn halton(mut index: u32, base: u32) -> f32 {
    let (mut fraction, mut result) = (1.0f32, 0.0f32);
    while index > 0 {
        fraction /= base as f32;
        result += fraction * (index % base) as f32;
        index /= base;
    }
    result
}

/// Pixel offsets are Y down and NDC is Y up, so the vertical component is negated.
fn pixel_offset_to_ndc(offset: [f32; 2], size: (u32, u32)) -> [f32; 2] {
    [
        2.0 * offset[0] / size.0.max(1) as f32,
        -2.0 * offset[1] / size.1.max(1) as f32,
    ]
}

pub(crate) fn frame_jitter(frame_index: u64, size: (u32, u32), jitter_consumer: bool) -> [f32; 2] {
    if !jitter_consumer {
        return [0.0; 2];
    }
    let sample = (frame_index % 8) as u32 + 1;
    let offset = [halton(sample, 2) - 0.5, halton(sample, 3) - 0.5];
    pixel_offset_to_ndc(offset, size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use raa_math::Vec3;

    const F: &str = include_str!("../tests/fixtures/frame-history.txt");
    const S: (u32, u32) = (1280, 720);

    fn close(a: [f32; 2], b: [f32; 2]) -> bool {
        (a[0] - b[0]).abs() <= 1e-7 && (a[1] - b[1]).abs() <= 1e-7
    }

    fn shift(x: f32) -> Mat4 {
        Mat4::from_translation(Vec3::new(x, 0.0, 0.0))
    }

    fn jit(m: Mat4, j: [f32; 2]) -> Mat4 {
        let mut s = Mat4::IDENTITY;
        s.cols[3] = [j[0], j[1], 0.0, 1.0];
        s * m
    }

    fn cycle(
        h: &mut FrameHistory,
        i: u64,
        p: (usize, Mat4, (u32, u32), bool, bool),
        ok: bool,
        done: bool,
    ) -> CameraHistory {
        let (v, m, sz, c, d) = p;
        h.begin(i);
        let r = h.prepare(v, m, m, sz, c, d);
        if ok {
            h.submitted(v);
        }
        h.finish(done);
        r
    }

    #[test]
    fn halton_samples_match_fixture() {
        let (mut hs, mut nd) = ([[0.0; 2]; 8], [[0.0; 2]; 8]);
        let mut seen = 0u16;
        for l in F.lines() {
            let f: Vec<_> = l.split_whitespace().collect();
            if f.len() < 4 || f[0] == "#" {
                continue;
            }
            let i = f[1].parse::<usize>().unwrap() - 1;
            let v = [f[2].parse::<f32>().unwrap(), f[3].parse::<f32>().unwrap()];
            if f[0] == "halton" {
                hs[i] = v;
                seen |= 1 << i;
            } else {
                nd[i] = v;
                seen |= 1 << (i + 8);
            }
        }
        assert_eq!(seen, u16::MAX);
        for i in 0..8 {
            let s = [halton(i as u32 + 1, 2), halton(i as u32 + 1, 3)];
            assert!(close(hs[i], s));
            assert!(close(nd[i], frame_jitter(i as u64, S, true)));
        }
        assert!(close(
            pixel_offset_to_ndc([1.0, 1.0], (100, 50)),
            [0.02, -0.04]
        ));
        assert!(close(
            pixel_offset_to_ndc([-1.0, -1.0], (100, 50)),
            [-0.02, 0.04]
        ));
    }

    #[test]
    fn history_invalidates_on_initial_resize_mode_and_partial_frames() {
        let mut h = FrameHistory::new();
        let m = Mat4::IDENTITY;
        let mut f =
            |i, sz, c, d, done| cycle(&mut h, i, (0, m, sz, c, d), true, done).history_valid;
        assert!(!f(0, S, false, false, true));
        assert!(f(1, S, false, false, true));
        assert!(!f(2, (1024, 768), false, false, true));
        assert!(f(3, (1024, 768), false, false, true));
        assert!(!f(4, (1024, 768), true, false, true));
        assert!(f(5, (1024, 768), true, false, true));
        assert!(!f(6, (1024, 768), true, true, true));
        assert!(f(7, (1024, 768), true, false, true));
        f(8, (1024, 768), true, false, false);
        assert!(!f(9, (1024, 768), true, false, true));
    }

    #[test]
    fn history_removes_jitter() {
        let mut h = FrameHistory::new();
        let (u0, u1) = (shift(1.0), shift(2.0));
        h.begin(0);
        let a = h.prepare(0, jit(u0, frame_jitter(0, S, true)), u0, S, true, false);
        assert!(!a.history_valid && a.unjittered_view_proj == u0);
        assert_eq!(a.previous_view_proj, jit(u0, a.current_jitter));
        assert_eq!(a.previous_jitter, a.current_jitter);
        h.submitted(0);
        h.finish(true);
        h.begin(1);
        let b = h.prepare(0, jit(u1, frame_jitter(1, S, true)), u1, S, true, false);
        assert!(b.history_valid && b.unjittered_view_proj == u1);
        assert_eq!(b.previous_view_proj, jit(u0, a.current_jitter));
        assert_eq!(b.previous_jitter, a.current_jitter);
        assert_ne!(b.current_jitter, b.previous_jitter);
    }

    #[test]
    fn pending_and_viewports_commit_only_submitted_views() {
        let mut h = FrameHistory::new();
        let (a0, b0, a1, b1) = (shift(1.0), shift(2.0), shift(3.0), shift(4.0));
        let (a2, b2) = (shift(5.0), shift(6.0));
        let t = (640, 360);
        h.begin(0);
        assert!(!h.prepare(0, a0, a0, S, false, false).history_valid);
        assert!(!h.prepare(1, b0, b0, t, false, false).history_valid);
        h.submitted(0);
        h.submitted(1);
        h.finish(true);
        h.begin(1);
        h.prepare(0, a1, a1, S, false, false);
        h.prepare(1, b1, b1, t, false, false);
        h.submitted(0);
        h.finish(true);
        h.begin(2);
        assert_eq!(h.prepare(0, a2, a2, S, false, false).previous_view_proj, a1);
        assert_eq!(h.prepare(1, b2, b2, t, false, false).previous_view_proj, b0);
    }

    #[test]
    fn jitter_is_zero_without_a_consumer() {
        let mut h = FrameHistory::new();
        let p = (0, Mat4::IDENTITY, S, false, false);
        let a = cycle(&mut h, 0, p, true, true);
        assert_eq!([a.current_jitter, a.previous_jitter], [[0.0; 2]; 2]);
    }
}
