//! CPU reference of the OIT resolve: the WGSL resolve uses the same algorithm.

use crate::node::{END, OitNode};

/// Default number of nearest fragments kept per pixel.
pub const DEFAULT_MAX_FRAGMENTS: usize = 16;
/// Upper bound accepted for `max`.
pub const MAX_FRAGMENTS_LIMIT: usize = 32;

/// A decoded fragment: straight linear colour, alpha, depth.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Fragment {
    pub color: [f32; 3],
    pub alpha: f32,
    pub depth: f32,
}

impl Fragment {
    pub fn from_node(n: &OitNode) -> Self {
        let (color, alpha) = n.color_alpha();
        Fragment {
            color,
            alpha,
            depth: n.depth(),
        }
    }
}

/// Walk the list from `head` and return the `max` nearest fragments, nearest
/// first (insertion sort on depth; farther ones are dropped). The walk stops
/// at END, at an out-of-range index, or after `nodes.len()` steps (cycle guard).
pub fn collect_nearest(nodes: &[OitNode], head: u32, max: usize) -> Vec<Fragment> {
    let max = max.clamp(1, MAX_FRAGMENTS_LIMIT);
    let mut kept: Vec<Fragment> = Vec::with_capacity(max);
    let mut cur = head;
    let mut steps = 0usize;
    while cur != END && steps <= nodes.len() {
        let Some(node) = nodes.get(cur as usize) else {
            break;
        };
        steps += 1;
        let f = Fragment::from_node(node);
        cur = node.next;
        if kept.len() == max {
            match kept.last() {
                Some(last) if f.depth < last.depth => {
                    kept.pop();
                }
                _ => continue,
            }
        }
        let at = kept.partition_point(|k| k.depth <= f.depth);
        kept.insert(at, f);
    }
    kept
}

/// Composite nearest-first `fragments` back to front over `background`:
/// dst = c * a + dst * (1 - a), with straight c.
pub fn composite_back_to_front(fragments: &[Fragment], background: [f32; 3]) -> [f32; 3] {
    let mut dst = background;
    for f in fragments.iter().rev() {
        let a = f.alpha.clamp(0.0, 1.0);
        for (d, c) in dst.iter_mut().zip(f.color) {
            *d = c * a + *d * (1.0 - a);
        }
    }
    dst
}

/// Full resolve of one pixel.
pub fn resolve_pixel(nodes: &[OitNode], head: u32, background: [f32; 3], max: usize) -> [f32; 3] {
    composite_back_to_front(&collect_nearest(nodes, head, max), background)
}
