//! Visual jiggle bones: secondary motion for cosmetic armatures (a jester hat's horns), never simulation state.
//!
//! Each jiggle bone carries a virtual tip point (its first child's rest offset, or its own rest offset when it has
//! no child). The tip is a point mass on a spring-damper pulled toward where the animated pose puts it; damping acts
//! on the velocity relative to that animated target, so a character running at constant speed shows no lag while a
//! turn, a bounce or a stop makes the bone swing. The tip stays at its rest distance from the bone head, and the
//! bone is rotated (world space, about its head) from the animated tip direction to the simulated one, clamped to
//! `maxAngle`. Children follow the jiggled parent.
//!
//! Frame-rate independence: a frame's real `dt` is cut into fixed substeps of 1/[`JIGGLE_SUBSTEP_HZ`] s (the
//! remainder carries to the next frame) with the animated target interpolated across the frame, and one frame never
//! integrates more than [`JIGGLE_MAX_FRAME_SECONDS`]. A chain that was not stepped in the previous call of its world
//! (culled, skipped under budget, quality Off) restarts at its animated pose instead of snapping from a stale state.
//!
//! Frames: a chain is posed in its *mount* frame. `world` is the instance placement (the character transform the
//! renderer applies) and `mount` maps the armature's rest frame into the character's posed frame (for a cosmetic
//! mounted on the penguin `head` bone: the penguin's `head` skin matrix). The returned palette is the skin matrices in
//! the instance frame: `inverse(world) * jiggledWorld * inverseBind`, so an unjiggled armature returns `mount` for
//! every bone.
use crate::rig::Jiggle;
use crate::{AnimError, Mat4, Rig};
use raa_det::fmath::trs_compose;

/// Fixed integration rate of the jiggle springs.
pub const JIGGLE_SUBSTEP_HZ: f32 = 240.0;
/// The most real time one step integrates; a longer frame (a stalled tab) is cut short.
pub const JIGGLE_MAX_FRAME_SECONDS: f32 = 1.0 / 15.0;
const GRAVITY: f32 = 9.81;
const IDENTITY: Mat4 = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

type V3 = [f32; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: V3, s: f32) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn length(a: V3) -> f32 {
    dot(a, a).sqrt()
}
fn lerp(a: V3, b: V3, u: f32) -> V3 {
    add(a, scale(sub(b, a), u))
}

/// Column-major product a * b.
pub fn mat_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [0.0; 16];
    for c in 0..4 {
        for r in 0..4 {
            out[c * 4 + r] = a[r] * b[c * 4]
                + a[4 + r] * b[c * 4 + 1]
                + a[8 + r] * b[c * 4 + 2]
                + a[12 + r] * b[c * 4 + 3];
        }
    }
    out
}

fn transform_point(m: &Mat4, p: V3) -> V3 {
    [
        m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
        m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
        m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
    ]
}

/// Inverse of an affine matrix (any invertible 3x3 basis plus translation); None when singular.
pub fn affine_inverse(m: &Mat4) -> Option<Mat4> {
    let c0 = [m[0], m[1], m[2]];
    let c1 = [m[4], m[5], m[6]];
    let c2 = [m[8], m[9], m[10]];
    let r0 = cross(c1, c2);
    let r1 = cross(c2, c0);
    let r2 = cross(c0, c1);
    let det = dot(c0, r0);
    if !det.is_finite() || det.abs() < 1e-12 {
        return None;
    }
    let rows = [
        scale(r0, 1.0 / det),
        scale(r1, 1.0 / det),
        scale(r2, 1.0 / det),
    ];
    let mut out = [0.0; 16];
    for (r, row) in rows.iter().enumerate() {
        for c in 0..3 {
            out[c * 4 + r] = row[c];
        }
        out[12 + r] = -(row[0] * m[12] + row[1] * m[13] + row[2] * m[14]);
    }
    out[15] = 1.0;
    Some(out)
}

/// Rotation matrix (3x3, column-major in a Mat4) of the shortest arc from unit `u` to unit `w`.
fn arc(u: V3, w: V3) -> [f32; 9] {
    let c = dot(u, w);
    let axis = cross(u, w);
    let (x, y, z, s) = if c < -0.999_999 {
        // Opposite directions: any axis perpendicular to u.
        let other = if u[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let a = cross(u, other);
        let l = length(a);
        (a[0] / l, a[1] / l, a[2] / l, 0.0)
    } else {
        let q = [axis[0], axis[1], axis[2], 1.0 + c];
        let l = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        (q[0] / l, q[1] / l, q[2] / l, q[3] / l)
    };
    [
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + z * s),
        2.0 * (x * z - y * s),
        2.0 * (x * y - z * s),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z + x * s),
        2.0 * (x * z + y * s),
        2.0 * (y * z - x * s),
        1.0 - 2.0 * (x * x + y * y),
    ]
}

/// Runtime spring constants of one jiggle bone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JiggleParams {
    /// Radians.
    pub max_angle: f32,
    /// Angular frequency, rad/s.
    pub omega: f32,
    /// Damping ratio.
    pub zeta: f32,
    /// Downward acceleration on the tip, m/s2.
    pub gravity: f32,
}

impl From<&Jiggle> for JiggleParams {
    fn from(j: &Jiggle) -> Self {
        let hz = 0.5 + 9.5 * j.stiffness as f32;
        Self {
            max_angle: (j.max_angle as f32).to_radians(),
            omega: std::f32::consts::TAU * hz,
            zeta: j.damping as f32,
            gravity: GRAVITY * j.gravity as f32,
        }
    }
}

#[derive(Debug, Clone)]
struct Spring {
    params: JiggleParams,
    /// Tip offset in the bone's local frame.
    tip: V3,
    p: V3,
    v: V3,
    prev_target: V3,
    prev_head: V3,
}

/// One armature instance (one worn cosmetic): its springs and its last palette.
#[derive(Debug, Clone)]
pub struct JiggleChain {
    parents: Vec<Option<usize>>,
    rest_local: Vec<Mat4>,
    inverse_bind: Vec<Mat4>,
    /// Animated locals; the root's is relative to the mount frame.
    local: Vec<Mat4>,
    springs: Vec<Option<Spring>>,
    world: Vec<Mat4>,
    palette: Vec<f32>,
    live: bool,
}

fn bad(message: impl Into<String>) -> AnimError {
    AnimError::new("jiggle.input", "", message)
}

impl JiggleChain {
    pub fn new(rig: &Rig) -> Result<Self, AnimError> {
        let rest = rig.local_rest_pose()?;
        let inverse_bind = rig.inverse_bind_matrices()?;
        let rest_local = rest
            .iter()
            .enumerate()
            .map(|(i, l)| {
                trs_compose(l.t, l.r, l.s).map_err(|e| {
                    AnimError::new("rig.schema", format!("/bones/{i}/rest"), e.to_string())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let parents: Vec<_> = rig.bones.iter().map(|b| b.parent).collect();
        let springs = rig
            .bones
            .iter()
            .enumerate()
            .map(|(i, bone)| {
                bone.jiggle.as_ref().map(|j| {
                    let child = parents.iter().position(|p| *p == Some(i));
                    let tip = match child {
                        Some(c) => rest[c].t,
                        None => {
                            // A leaf: extend along its own offset from the parent, in its own frame.
                            let [x, y, z, w] = rest[i].r;
                            let t = rest[i].t;
                            let qv = [-x, -y, -z];
                            let tv = scale(cross(qv, t), 2.0);
                            add(add(t, scale(tv, w)), cross(qv, tv))
                        }
                    };
                    let tip = if length(tip) < 1e-4 {
                        [0.0, 0.1, 0.0]
                    } else {
                        tip
                    };
                    Spring {
                        params: JiggleParams::from(j),
                        tip,
                        p: [0.0; 3],
                        v: [0.0; 3],
                        prev_target: [0.0; 3],
                        prev_head: [0.0; 3],
                    }
                })
            })
            .collect();
        let n = rig.bones.len();
        Ok(Self {
            parents,
            local: rest_local.clone(),
            rest_local,
            inverse_bind,
            springs,
            world: vec![IDENTITY; n],
            palette: vec![0.0; n * 16],
            live: false,
        })
    }

    pub fn bone_count(&self) -> usize {
        self.parents.len()
    }

    /// Bones that jiggle.
    pub fn jiggle_count(&self) -> usize {
        self.springs.iter().filter(|s| s.is_some()).count()
    }

    /// The animated pose as model matrices in the mount frame (16 floats per bone); an empty slice restores rest.
    pub fn set_pose(&mut self, model: &[f32]) -> Result<(), AnimError> {
        if model.is_empty() {
            self.local.clone_from(&self.rest_local);
            return Ok(());
        }
        let n = self.parents.len();
        if model.len() != n * 16 || !model.iter().all(|v| v.is_finite()) {
            return Err(bad(format!(
                "pose must be {} finite floats (16 per bone)",
                n * 16
            )));
        }
        let at = |i: usize| -> Mat4 { model[i * 16..i * 16 + 16].try_into().expect("16 floats") };
        for i in 0..n {
            self.local[i] = match self.parents[i] {
                None => at(i),
                Some(p) => mat_mul(
                    &affine_inverse(&at(p))
                        .ok_or_else(|| bad(format!("bone {p} model is singular")))?,
                    &at(i),
                ),
            };
        }
        Ok(())
    }

    /// The animated pose as skin matrices (model * inverse bind, 16 floats per bone, the palette a renderer draws
    /// without jiggle); an empty slice restores rest. For callers that only hold the skin palette (an editor preview).
    pub fn set_skin(&mut self, skin: &[f32]) -> Result<(), AnimError> {
        if skin.is_empty() {
            return self.set_pose(skin);
        }
        let n = self.parents.len();
        if skin.len() != n * 16 {
            return Err(bad(format!("skin must be {} floats (16 per bone)", n * 16)));
        }
        let mut model = vec![0.0; n * 16];
        for i in 0..n {
            let bind = affine_inverse(&self.inverse_bind[i])
                .ok_or_else(|| bad(format!("bone {i} bind is singular")))?;
            let s: Mat4 = skin[i * 16..i * 16 + 16].try_into().expect("16 floats");
            model[i * 16..i * 16 + 16].copy_from_slice(&mat_mul(&s, &bind));
        }
        self.set_pose(&model)
    }

    /// The next step starts from the animated pose (after a teleport or a cut).
    pub fn reset(&mut self) {
        self.live = false;
    }

    /// Integrate `substeps` steps of `h` seconds toward the animated pose under `world` * `mount`, then return the
    /// palette (16 floats per bone, instance frame). Zero substeps only re-poses the springs' current state.
    pub fn step(
        &mut self,
        substeps: u32,
        h: f32,
        world: &Mat4,
        mount: &Mat4,
    ) -> Result<&[f32], AnimError> {
        let inverse_world = affine_inverse(world).ok_or_else(|| bad("world matrix is singular"))?;
        let frame = mat_mul(world, mount);
        let restart = !self.live;
        for i in 0..self.parents.len() {
            let parent = match self.parents[i] {
                Some(p) => self.world[p],
                None => frame,
            };
            let anim = mat_mul(&parent, &self.local[i]);
            self.world[i] = match self.springs[i].as_mut() {
                Some(spring) => swing(spring, &anim, restart, substeps, h),
                None => anim,
            };
            let skin = mat_mul(
                &inverse_world,
                &mat_mul(&self.world[i], &self.inverse_bind[i]),
            );
            self.palette[i * 16..i * 16 + 16].copy_from_slice(&skin);
        }
        if !self.palette.iter().all(|v| v.is_finite()) {
            // A degenerate input never reaches the renderer: fall back to the rigid mount and restart.
            self.live = false;
            for i in 0..self.parents.len() {
                self.palette[i * 16..i * 16 + 16].copy_from_slice(mount);
            }
            return Ok(&self.palette);
        }
        self.live = true;
        Ok(&self.palette)
    }
}

/// One jiggle bone: advance its tip and return its jiggled world matrix.
fn swing(s: &mut Spring, anim: &Mat4, restart: bool, substeps: u32, h: f32) -> Mat4 {
    let head = [anim[12], anim[13], anim[14]];
    let target = transform_point(anim, s.tip);
    let len = length(sub(target, head));
    if len < 1e-6 || !len.is_finite() {
        return *anim;
    }
    let far = length(sub(s.p, target)) > 4.0 * len + 1.0;
    if restart || far || !s.p.iter().chain(&s.v).all(|v| v.is_finite()) {
        s.p = target;
        s.v = [0.0; 3];
        s.prev_target = target;
        s.prev_head = head;
    }
    let JiggleParams {
        max_angle,
        omega,
        zeta,
        gravity,
    } = s.params;
    if substeps > 0 {
        let span = substeps as f32 * h;
        let target_velocity = scale(sub(target, s.prev_target), 1.0 / span);
        for k in 1..=substeps {
            let u = k as f32 / substeps as f32;
            let tk = lerp(s.prev_target, target, u);
            let hk = lerp(s.prev_head, head, u);
            let spring = scale(sub(tk, s.p), omega * omega);
            let damper = scale(sub(s.v, target_velocity), 2.0 * zeta * omega);
            let acc = add(sub(spring, damper), [0.0, -gravity, 0.0]);
            s.v = add(s.v, scale(acc, h));
            s.p = add(s.p, scale(s.v, h));
            let d = sub(s.p, hk);
            let dl = length(d);
            if dl > 1e-9 {
                let dn = scale(d, 1.0 / dl);
                s.p = add(hk, scale(dn, len));
                s.v = sub(s.v, scale(dn, dot(s.v, dn)));
            }
        }
        s.prev_target = target;
        s.prev_head = head;
    }
    let u = scale(sub(target, head), 1.0 / len);
    let d = sub(s.p, head);
    let dl = length(d);
    let mut w = if dl > 1e-9 { scale(d, 1.0 / dl) } else { u };
    let c = dot(u, w).clamp(-1.0, 1.0);
    if c.acos() > max_angle {
        let perp = sub(w, scale(u, c));
        let pl = length(perp);
        if pl > 1e-9 {
            let perp = scale(perp, 1.0 / pl);
            w = add(scale(u, max_angle.cos()), scale(perp, max_angle.sin()));
            s.p = add(head, scale(w, len));
            let out = dot(s.v, perp);
            if out > 0.0 {
                s.v = sub(s.v, scale(perp, out));
            }
        } else {
            w = u;
            s.p = target;
        }
    }
    let r = arc(u, w);
    let mut out = *anim;
    for col in 0..3 {
        let v = [anim[col * 4], anim[col * 4 + 1], anim[col * 4 + 2]];
        for row in 0..3 {
            out[col * 4 + row] = r[row] * v[0] + r[3 + row] * v[1] + r[6 + row] * v[2];
        }
    }
    out
}

/// All jiggle chains of one view, stepped together once per rendered frame.
#[derive(Debug, Default)]
pub struct JiggleWorld {
    chains: Vec<Option<(JiggleChain, u64)>>,
    free: Vec<u32>,
    carry: f32,
    serial: u64,
}

fn handle_error(handle: u32) -> AnimError {
    AnimError::new(
        "jiggle.handle",
        "",
        format!("unknown jiggle handle {handle}"),
    )
}

impl JiggleWorld {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register one armature instance; the handle stays valid until `remove`.
    pub fn add(&mut self, rig: &Rig) -> Result<u32, AnimError> {
        let chain = JiggleChain::new(rig)?;
        Ok(match self.free.pop() {
            Some(h) => {
                self.chains[h as usize] = Some((chain, 0));
                h
            }
            None => {
                self.chains.push(Some((chain, 0)));
                (self.chains.len() - 1) as u32
            }
        })
    }

    pub fn remove(&mut self, handle: u32) {
        if let Some(slot) = self.chains.get_mut(handle as usize)
            && slot.take().is_some()
        {
            self.free.push(handle);
        }
    }

    pub fn len(&self) -> usize {
        self.chains.iter().filter(|c| c.is_some()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn chain(&self, handle: u32) -> Result<&JiggleChain, AnimError> {
        self.chains
            .get(handle as usize)
            .and_then(|c| c.as_ref())
            .map(|(c, _)| c)
            .ok_or_else(|| handle_error(handle))
    }

    pub fn chain_mut(&mut self, handle: u32) -> Result<&mut JiggleChain, AnimError> {
        self.chains
            .get_mut(handle as usize)
            .and_then(|c| c.as_mut())
            .map(|(c, _)| c)
            .ok_or_else(|| handle_error(handle))
    }

    /// Substeps one frame of `dt` real seconds integrates (the remainder carries over).
    fn substeps(&mut self, dt: f32) -> u32 {
        self.carry += dt.clamp(0.0, JIGGLE_MAX_FRAME_SECONDS);
        let n = (self.carry * JIGGLE_SUBSTEP_HZ).floor();
        self.carry = (self.carry - n / JIGGLE_SUBSTEP_HZ).max(0.0);
        n as u32
    }

    /// Step the listed chains by one frame of `dt` seconds. `frames` holds 32 floats per handle: its `world`
    /// matrix then its `mount` matrix (see the module docs). Returns the palettes, concatenated in handle order.
    /// A chain left out of the previous call restarts at its animated pose.
    pub fn step(
        &mut self,
        dt: f32,
        handles: &[u32],
        frames: &[f32],
    ) -> Result<Vec<f32>, AnimError> {
        if !dt.is_finite() || dt < 0.0 {
            return Err(bad("dt must be a finite number of seconds >= 0"));
        }
        if frames.len() != handles.len() * 32 {
            return Err(bad("frames must hold 32 floats (world, mount) per handle"));
        }
        for &h in handles {
            self.chain(h)?;
        }
        let substeps = self.substeps(dt);
        self.serial += 1;
        let serial = self.serial;
        let mut out = Vec::new();
        for (k, &h) in handles.iter().enumerate() {
            let (chain, last) = self.chains[h as usize].as_mut().expect("checked handle");
            if *last == serial {
                return Err(bad(format!("handle {h} listed twice in one step")));
            }
            if *last + 1 != serial {
                chain.reset();
            }
            *last = serial;
            let world: Mat4 = frames[k * 32..k * 32 + 16].try_into().expect("16 floats");
            let mount: Mat4 = frames[k * 32 + 16..k * 32 + 32]
                .try_into()
                .expect("16 floats");
            out.extend_from_slice(chain.step(substeps, 1.0 / JIGGLE_SUBSTEP_HZ, &world, &mount)?);
        }
        Ok(out)
    }
}
