//! Incremental binary BVH over instance world bounds, keyed by instance slot.
//!
//! Nodes live in flat vectors: an internal node holds its two child indices, a leaf
//! holds a range of at most [`LEAF_MAX`] entries in `items`. `slot_pos` maps a slot to
//! its position in `items` and `slot_leaf` to the leaf that owns it, so there is no
//! hash map and iteration order is deterministic. Slots with `None` bounds live in a
//! separate ascending list: they always pass a frustum and never hit a ray.
//!
//! Query paths never allocate once the caller's `out` has capacity: they walk a
//! fixed-size stack of 64 node indices and fall back to a flat scan on overflow.

use core::cmp::Ordering;

use raa_math::{Aabb, Mat4, Vec3};

/// A leaf holds at most this many slots; a fifth one splits it.
const LEAF_MAX: usize = 4;
/// Fixed query stack. Balanced trees need about `log2(slots)`, far below this.
const STACK: usize = 64;
/// Sentinel for "no node / no position".
const NONE: u32 = u32::MAX;

/// A node of the flat BVH. `count > 0` marks a leaf whose entries are
/// `items[left..left + count]`; otherwise `left` and `right` are child indices.
#[derive(Copy, Clone, Debug)]
struct Node {
    bounds: Aabb,
    left: u32,
    right: u32,
    count: u32,
}

/// The state of one instance slot.
#[derive(Copy, Clone, Debug, Default)]
enum Entry {
    #[default]
    Absent,
    /// No bounds: always visible to frustum queries, never to ray queries.
    Unbounded,
    Bounded(Aabb),
}

/// Incremental BVH over the world bounds of instances, keyed by instance slot
/// (`InstanceId::index`).
#[derive(Clone, Debug, Default)]
pub struct InstanceIndex {
    nodes: Vec<Node>,
    /// Flat sequence of leaf slots; each live leaf owns one contiguous range.
    items: Vec<u32>,
    /// Parent node of every node (`0` for the root).
    parent: Vec<u32>,
    /// Nodes freed by a leaf merge; never reachable from the root.
    dead: Vec<bool>,
    /// Entry per slot, indexed by slot.
    entries: Vec<Entry>,
    /// Unbounded slots, ascending.
    unbounded: Vec<u32>,
    /// Slot -> position in `items` (`NONE` when absent/unbounded).
    slot_pos: Vec<u32>,
    /// Slot -> owning leaf node (`NONE` when absent/unbounded).
    slot_leaf: Vec<u32>,
    count: usize,
    /// Mutations since the last full rebuild.
    pending: u64,
    /// Root surface area recorded at the last build.
    built_area: f32,
}

impl InstanceIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of live slots (bounded and unbounded).
    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Inserts `slot`, replacing any existing entry.
    pub fn insert(&mut self, slot: u32, bounds: Option<Aabb>) {
        self.ensure(slot);
        let existing = self.entries[slot as usize];
        if !matches!(existing, Entry::Absent) {
            match existing {
                Entry::Unbounded => self.remove_unbounded(slot),
                Entry::Bounded(_) => self.remove_bounded_tree(slot),
                Entry::Absent => {}
            }
            self.count -= 1;
        }
        match bounds {
            Some(b) => {
                self.entries[slot as usize] = Entry::Bounded(b);
                self.insert_bounded_tree(slot, b);
            }
            None => {
                self.entries[slot as usize] = Entry::Unbounded;
                self.add_unbounded(slot);
            }
        }
        self.count += 1;
        self.pending += 1;
        self.maybe_rebuild();
    }

    /// Removes `slot`; false when it was absent.
    pub fn remove(&mut self, slot: u32) -> bool {
        if slot as usize >= self.entries.len() {
            return false;
        }
        let existing = self.entries[slot as usize];
        match existing {
            Entry::Absent => return false,
            Entry::Unbounded => {
                self.remove_unbounded(slot);
                self.entries[slot as usize] = Entry::Absent;
            }
            Entry::Bounded(_) => {
                self.remove_bounded_tree(slot);
                self.entries[slot as usize] = Entry::Absent;
            }
        }
        self.count -= 1;
        self.pending += 1;
        self.maybe_rebuild();
        true
    }

    /// Replaces the bounds of an existing slot. Returns false, and changes nothing,
    /// when the slot is absent or the new bounds are bit-equal to the old ones.
    pub fn update(&mut self, slot: u32, bounds: Option<Aabb>) -> bool {
        if slot as usize >= self.entries.len() {
            return false;
        }
        let old = match self.entries[slot as usize] {
            Entry::Absent => return false,
            Entry::Unbounded => None,
            Entry::Bounded(b) => Some(b),
        };
        if option_bits_eq(old, bounds) {
            return false;
        }
        match (old, bounds) {
            (None, Some(nb)) => {
                self.remove_unbounded(slot);
                self.entries[slot as usize] = Entry::Bounded(nb);
                self.insert_bounded_tree(slot, nb);
            }
            (Some(_), None) => {
                self.remove_bounded_tree(slot);
                self.entries[slot as usize] = Entry::Unbounded;
                self.add_unbounded(slot);
            }
            (Some(_), Some(nb)) => {
                self.entries[slot as usize] = Entry::Bounded(nb);
                let leaf = self.slot_leaf[slot as usize];
                if leaf == NONE {
                    self.rebuild();
                } else {
                    self.refit_from(leaf);
                }
            }
            (None, None) => unreachable!("equal unbounded bounds were rejected above"),
        }
        self.pending += 1;
        self.maybe_rebuild();
        true
    }

    /// `None` when absent, `Some(None)` when unbounded, `Some(Some(b))` when bounded.
    pub fn bounds(&self, slot: u32) -> Option<Option<Aabb>> {
        match self.entries.get(slot as usize) {
            None | Some(Entry::Absent) => None,
            Some(Entry::Unbounded) => Some(None),
            Some(Entry::Bounded(b)) => Some(Some(*b)),
        }
    }

    /// Number of live BVH nodes.
    pub fn node_count(&self) -> usize {
        self.nodes
            .iter()
            .zip(self.dead.iter())
            .filter(|(_, dead)| !**dead)
            .count()
    }

    /// Clears `out`, then appends every slot not fully behind one of `planes`,
    /// ascending. A plane keeps points with `p[0]x + p[1]y + p[2]z + p[3] >= 0`.
    pub fn query_frustum(&self, planes: &[[f32; 4]], out: &mut Vec<u32>) {
        out.clear();
        out.extend_from_slice(&self.unbounded);
        if self.nodes.is_empty() {
            out.sort_unstable();
            return;
        }
        let mut stack = [0u32; STACK];
        let mut sp = 1usize;
        stack[0] = 0;
        let mut overflow = false;
        while sp > 0 {
            sp -= 1;
            let node = self.nodes[stack[sp] as usize];
            if fully_outside(&node.bounds, planes) {
                continue;
            }
            if node.count > 0 {
                let start = node.left as usize;
                let end = start + node.count as usize;
                for &slot in &self.items[start..end] {
                    if let Entry::Bounded(b) = self.entries[slot as usize]
                        && !fully_outside(&b, planes)
                    {
                        out.push(slot);
                    }
                }
            } else if sp + 2 > STACK {
                overflow = true;
                break;
            } else {
                stack[sp] = node.left;
                stack[sp + 1] = node.right;
                sp += 2;
            }
        }
        if overflow {
            out.clear();
            out.extend_from_slice(&self.unbounded);
            for (slot, entry) in self.entries.iter().enumerate() {
                if let Entry::Bounded(b) = entry
                    && !fully_outside(b, planes)
                {
                    out.push(slot as u32);
                }
            }
        }
        out.sort_unstable();
    }

    /// Clears `out`, then appends `(slot, entry t)` for every bounded box the ray
    /// enters with `t <= max_t`, sorted by `t` then slot. `dir` is normalised here.
    pub fn query_ray(&self, origin: Vec3, dir: Vec3, max_t: f32, out: &mut Vec<(u32, f32)>) {
        out.clear();
        let dir = dir.normalize_or_zero();
        if dir.length_squared() == 0.0 || max_t < 0.0 || self.nodes.is_empty() {
            return;
        }
        let mut stack = [0u32; STACK];
        let mut sp = 1usize;
        stack[0] = 0;
        let mut overflow = false;
        while sp > 0 {
            sp -= 1;
            let node = self.nodes[stack[sp] as usize];
            if ray_aabb(&node.bounds, origin, dir, max_t).is_none() {
                continue;
            }
            if node.count > 0 {
                let start = node.left as usize;
                let end = start + node.count as usize;
                for &slot in &self.items[start..end] {
                    if let Entry::Bounded(b) = self.entries[slot as usize]
                        && let Some(t) = ray_aabb(&b, origin, dir, max_t)
                    {
                        out.push((slot, t));
                    }
                }
            } else if sp + 2 > STACK {
                overflow = true;
                break;
            } else {
                stack[sp] = node.left;
                stack[sp + 1] = node.right;
                sp += 2;
            }
        }
        if overflow {
            out.clear();
            for (slot, entry) in self.entries.iter().enumerate() {
                if let Entry::Bounded(b) = entry
                    && let Some(t) = ray_aabb(b, origin, dir, max_t)
                {
                    out.push((slot as u32, t));
                }
            }
        }
        out.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    }
}

/// The six WebGPU clip planes of `view_proj`, in the order left, right, bottom,
/// top, near, far. Row `r` is `[cols[0][r], cols[1][r], cols[2][r], cols[3][r]]`;
/// near is row 2 and far is row 3 minus row 2.
pub fn frustum_planes(view_proj: Mat4) -> [[f32; 4]; 6] {
    let row = |r: usize| {
        [
            view_proj.cols[0][r],
            view_proj.cols[1][r],
            view_proj.cols[2][r],
            view_proj.cols[3][r],
        ]
    };
    let add = |a: [f32; 4], b: [f32; 4], sign: f32| {
        [
            a[0] + sign * b[0],
            a[1] + sign * b[1],
            a[2] + sign * b[2],
            a[3] + sign * b[3],
        ]
    };
    let [x, y, z, w] = [row(0), row(1), row(2), row(3)];
    [
        add(w, x, 1.0),
        add(w, x, -1.0),
        add(w, y, 1.0),
        add(w, y, -1.0),
        z,
        add(w, z, -1.0),
    ]
}

fn option_bits_eq(a: Option<Aabb>, b: Option<Aabb>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.min.x.to_bits() == b.min.x.to_bits()
                && a.min.y.to_bits() == b.min.y.to_bits()
                && a.min.z.to_bits() == b.min.z.to_bits()
                && a.max.x.to_bits() == b.max.x.to_bits()
                && a.max.y.to_bits() == b.max.y.to_bits()
                && a.max.z.to_bits() == b.max.z.to_bits()
        }
        _ => false,
    }
}

fn valid_aabb(b: &Aabb) -> bool {
    [b.min.x, b.min.y, b.min.z, b.max.x, b.max.y, b.max.z]
        .iter()
        .all(|v| v.is_finite())
        && b.min.x <= b.max.x
        && b.min.y <= b.max.y
        && b.min.z <= b.max.z
}

/// A box is outside when every corner is behind one plane (max signed distance < 0).
fn fully_outside(b: &Aabb, planes: &[[f32; 4]]) -> bool {
    if !valid_aabb(b) {
        return false;
    }
    for p in planes {
        let pick = |v: f32, lo: f32, hi: f32| v * if v >= 0.0 { hi } else { lo };
        let d = pick(p[0], b.min.x, b.max.x)
            + pick(p[1], b.min.y, b.max.y)
            + pick(p[2], b.min.z, b.max.z)
            + p[3];
        if d.is_finite() && d < 0.0 {
            return true;
        }
    }
    false
}

/// Entry distance (clamped to `>= 0`) when the ray hits the box within `max_t`.
fn ray_aabb(b: &Aabb, origin: Vec3, dir: Vec3, max_t: f32) -> Option<f32> {
    if !valid_aabb(b) {
        return None;
    }
    let mut tmin = 0.0f32;
    let mut tmax = max_t;
    let axes = [
        (origin.x, dir.x, b.min.x, b.max.x),
        (origin.y, dir.y, b.min.y, b.max.y),
        (origin.z, dir.z, b.min.z, b.max.z),
    ];
    for (o, d, lo, hi) in axes {
        if d.abs() < 1e-20 {
            if o < lo || o > hi {
                return None;
            }
        } else {
            let inv = 1.0 / d;
            let mut t1 = (lo - o) * inv;
            let mut t2 = (hi - o) * inv;
            if t1 > t2 {
                core::mem::swap(&mut t1, &mut t2);
            }
            if t1 > tmin {
                tmin = t1;
            }
            if t2 < tmax {
                tmax = t2;
            }
            if tmin > tmax {
                return None;
            }
        }
    }
    if tmax < 0.0 {
        return None;
    }
    Some(tmin)
}

fn surface_area(b: Aabb) -> f32 {
    if !valid_aabb(&b) {
        return 0.0;
    }
    let e = b.extent();
    2.0 * (e.x * e.y + e.x * e.z + e.y * e.z)
}

impl InstanceIndex {
    fn ensure(&mut self, slot: u32) {
        let need = slot as usize + 1;
        if self.entries.len() < need {
            self.entries.resize(need, Entry::Absent);
            self.slot_pos.resize(need, NONE);
            self.slot_leaf.resize(need, NONE);
        }
    }

    fn add_unbounded(&mut self, slot: u32) {
        if let Err(pos) = self.unbounded.binary_search(&slot) {
            self.unbounded.insert(pos, slot);
        }
    }

    fn remove_unbounded(&mut self, slot: u32) {
        if let Ok(pos) = self.unbounded.binary_search(&slot) {
            self.unbounded.remove(pos);
        }
    }

    /// Adds a slot that is already marked `Bounded`. The tree is first grown at the
    /// leaf whose bounds grow least (ties left), then split at five, then refitted.
    fn insert_bounded_tree(&mut self, slot: u32, bounds: Aabb) {
        if self.nodes.is_empty() {
            self.nodes.push(Node {
                bounds,
                left: 0,
                right: 0,
                count: 1,
            });
            self.parent.push(0);
            self.dead.push(false);
            self.items.push(slot);
            self.slot_pos[slot as usize] = 0;
            self.slot_leaf[slot as usize] = 0;
            self.built_area = surface_area(bounds);
            return;
        }
        let mut idx = 0u32;
        loop {
            let node = self.nodes[idx as usize];
            if node.count > 0 {
                break;
            }
            let l = node.left;
            let r = node.right;
            let lb = self.nodes[l as usize].bounds;
            let rb = self.nodes[r as usize].bounds;
            let lg = surface_area(lb.union(&bounds)) - surface_area(lb);
            let rg = surface_area(rb.union(&bounds)) - surface_area(rb);
            idx = if lg <= rg { l } else { r };
        }
        let leaf = idx;
        let node = self.nodes[leaf as usize];
        let at = (node.left + node.count) as usize;
        self.items.insert(at, slot);
        self.shift_starts(at as u32, 1);
        self.nodes[leaf as usize].count = node.count + 1;
        self.slot_leaf[slot as usize] = leaf;
        if self.nodes[leaf as usize].count as usize > LEAF_MAX {
            self.split_leaf(leaf);
        }
        self.refit_from(leaf);
        self.reindex();
    }

    /// Turns the overflowing leaf into an internal node with two leaves split at the
    /// median of the longest centroid axis.
    fn split_leaf(&mut self, leaf: u32) {
        let node = self.nodes[leaf as usize];
        let start = node.left as usize;
        let count = node.count as usize;
        let mut slots = self.items[start..start + count].to_vec();
        self.sort_slots(&mut slots);
        let mid = count / 2;
        let left_bounds = self.bounds_of_slots(&slots[..mid]);
        let right_bounds = self.bounds_of_slots(&slots[mid..]);
        self.items.drain(start..start + count);
        self.shift_starts((start + count) as u32, -(count as i32));
        let lstart = self.items.len() as u32;
        self.items.extend_from_slice(&slots[..mid]);
        let rstart = self.items.len() as u32;
        self.items.extend_from_slice(&slots[mid..]);
        let li = self.nodes.len() as u32;
        let ri = li + 1;
        self.nodes.push(Node {
            bounds: left_bounds,
            left: lstart,
            right: 0,
            count: mid as u32,
        });
        self.nodes.push(Node {
            bounds: right_bounds,
            left: rstart,
            right: 0,
            count: (count - mid) as u32,
        });
        self.parent.push(leaf);
        self.parent.push(leaf);
        self.dead.push(false);
        self.dead.push(false);
        self.nodes[leaf as usize] = Node {
            bounds: left_bounds.union(&right_bounds),
            left: li,
            right: ri,
            count: 0,
        };
    }

    /// Removes a bounded slot from the tree, merges an emptied leaf with its sibling
    /// and refits the ancestors. The entry is changed by the caller.
    fn remove_bounded_tree(&mut self, slot: u32) {
        let leaf = self.slot_leaf[slot as usize];
        if leaf == NONE {
            debug_assert!(false, "bounded slot has no leaf");
            return;
        }
        let node = self.nodes[leaf as usize];
        let start = node.left as usize;
        let count = node.count as usize;
        let Some(pos) = self.items[start..start + count]
            .iter()
            .position(|&s| s == slot)
        else {
            debug_assert!(false, "bounded slot missing from its leaf");
            return;
        };
        let last = start + count - 1;
        self.items.swap(start + pos, last);
        self.items.remove(last);
        self.nodes[leaf as usize].count = (count - 1) as u32;
        self.shift_starts(last as u32 + 1, -1);
        self.slot_pos[slot as usize] = NONE;
        self.slot_leaf[slot as usize] = NONE;
        if count == 1 {
            if leaf == 0 {
                self.nodes.clear();
                self.parent.clear();
                self.dead.clear();
                self.items.clear();
            } else {
                let parent = self.parent[leaf as usize];
                let pn = self.nodes[parent as usize];
                let sibling = if pn.left == leaf { pn.right } else { pn.left };
                self.absorb(parent, sibling);
                self.dead[leaf as usize] = true;
                self.refit_from(parent);
            }
        } else {
            self.refit_from(leaf);
        }
        self.reindex();
    }

    /// Moves the content of `src` into `dst` (a parent whose other child emptied),
    /// then frees `src`.
    fn absorb(&mut self, dst: u32, src: u32) {
        let node = self.nodes[src as usize];
        self.nodes[dst as usize] = node;
        self.dead[src as usize] = true;
        if node.count > 0 {
            let start = node.left as usize;
            let end = start + node.count as usize;
            for pos in start..end {
                let slot = self.items[pos] as usize;
                self.slot_leaf[slot] = dst;
                self.slot_pos[slot] = pos as u32;
            }
        } else {
            self.parent[node.left as usize] = dst;
            self.parent[node.right as usize] = dst;
        }
    }

    /// Recomputes bounds on the path from `start` to the root.
    fn refit_from(&mut self, start: u32) {
        let mut idx = start;
        loop {
            let node = self.nodes[idx as usize];
            let bounds = if node.count > 0 {
                let start = node.left as usize;
                let end = start + node.count as usize;
                let mut b = Aabb::EMPTY;
                for pos in start..end {
                    if let Entry::Bounded(slot) = self.entries[self.items[pos] as usize] {
                        b = b.union(&slot);
                    }
                }
                b
            } else {
                let l = self.nodes[node.left as usize].bounds;
                let r = self.nodes[node.right as usize].bounds;
                l.union(&r)
            };
            self.nodes[idx as usize].bounds = bounds;
            if idx == 0 {
                break;
            }
            idx = self.parent[idx as usize];
        }
    }

    /// Moves every live leaf start at or after `threshold` by `delta` positions.
    fn shift_starts(&mut self, threshold: u32, delta: i32) {
        for (idx, node) in self.nodes.iter_mut().enumerate() {
            if self.dead[idx] || node.count == 0 || node.left < threshold {
                continue;
            }
            node.left = (node.left as i64 + delta as i64) as u32;
        }
    }

    /// Rebuilds the slot maps from `items` and the live leaves.
    fn reindex(&mut self) {
        for (pos, &slot) in self.items.iter().enumerate() {
            self.slot_pos[slot as usize] = pos as u32;
        }
        for (idx, node) in self.nodes.iter().enumerate() {
            if self.dead[idx] || node.count == 0 {
                continue;
            }
            let start = node.left as usize;
            let end = start + node.count as usize;
            for pos in start..end {
                let slot = self.items[pos] as usize;
                self.slot_leaf[slot] = idx as u32;
            }
        }
    }

    /// A full median-split rebuild of every bounded slot, ascending.
    fn rebuild(&mut self) {
        let mut list: Vec<u32> = Vec::new();
        for (slot, entry) in self.entries.iter().enumerate() {
            if matches!(entry, Entry::Bounded(_)) {
                list.push(slot as u32);
            }
        }
        self.nodes.clear();
        self.parent.clear();
        self.dead.clear();
        self.items.clear();
        if list.is_empty() {
            self.built_area = 0.0;
            self.pending = 0;
            return;
        }
        let root = self.build_range(&mut list);
        debug_assert_eq!(root, 0);
        self.built_area = surface_area(self.nodes[0].bounds);
        self.pending = 0;
        self.reindex();
    }

    fn build_range(&mut self, list: &mut [u32]) -> u32 {
        let idx = self.nodes.len() as u32;
        self.nodes.push(Node {
            bounds: Aabb::EMPTY,
            left: 0,
            right: 0,
            count: 0,
        });
        self.parent.push(0);
        self.dead.push(false);
        if list.len() <= LEAF_MAX {
            let start = self.items.len() as u32;
            self.items.extend_from_slice(list);
            let bounds = self.bounds_of_slots(list);
            self.nodes[idx as usize] = Node {
                bounds,
                left: start,
                right: 0,
                count: list.len() as u32,
            };
        } else {
            self.sort_slots(list);
            let mid = list.len() / 2;
            let (left, right) = list.split_at_mut(mid);
            let li = self.build_range(left);
            let ri = self.build_range(right);
            self.parent[li as usize] = idx;
            self.parent[ri as usize] = idx;
            let bounds = self.nodes[li as usize]
                .bounds
                .union(&self.nodes[ri as usize].bounds);
            self.nodes[idx as usize] = Node {
                bounds,
                left: li,
                right: ri,
                count: 0,
            };
        }
        idx
    }

    /// Sorts by the longest axis of the centroid bounds; ties keep ascending slot order.
    fn sort_slots(&self, slots: &mut [u32]) {
        let axis = self.longest_centroid_axis(slots);
        let entries = &self.entries;
        let key = |slot: u32| match entries[slot as usize] {
            Entry::Bounded(b) => match axis {
                0 => b.center().x,
                1 => b.center().y,
                _ => b.center().z,
            },
            Entry::Absent | Entry::Unbounded => 0.0,
        };
        slots.sort_unstable_by(|&a, &b| match key(a).total_cmp(&key(b)) {
            Ordering::Equal => a.cmp(&b),
            order => order,
        });
    }

    fn longest_centroid_axis(&self, slots: &[u32]) -> usize {
        let mut bounds = Aabb::EMPTY;
        for &slot in slots {
            if let Entry::Bounded(b) = self.entries[slot as usize] {
                bounds.include(b.center());
            }
        }
        let e = bounds.extent();
        if e.x >= e.y && e.x >= e.z {
            0
        } else if e.y >= e.z {
            1
        } else {
            2
        }
    }

    fn bounds_of_slots(&self, slots: &[u32]) -> Aabb {
        let mut bounds = Aabb::EMPTY;
        for &slot in slots {
            if let Entry::Bounded(b) = self.entries[slot as usize] {
                bounds = bounds.union(&b);
            }
        }
        bounds
    }

    fn maybe_rebuild(&mut self) {
        if self.pending == 0 {
            return;
        }
        let threshold = 64u64.max(self.count as u64 / 2);
        let mut rebuild = self.pending > threshold;
        if !rebuild && self.built_area > 0.0 && !self.nodes.is_empty() {
            rebuild = surface_area(self.nodes[0].bounds) > 4.0 * self.built_area;
        }
        if rebuild {
            self.rebuild();
        }
    }
}
