// particles.sim: standalone compute module (no prelude). One thread per particle slot.
//
// A dead slot (age >= life) claims a spawn of its emitter's command for this step with an
// atomic counter and spawns serial `first + j` with the recipe of spawn.rs (same hash,
// same draw order, same f32 operation order). A live slot integrates one fixed step:
// gravity and the kind force, exponential drag, position, age, then the ground bounce.

struct Particle {
    pos_age: vec4<f32>,
    vel_life: vec4<f32>,
    size: f32,
    rotation: f32,
    color_index: u32,
    kind: u32,
}

struct EmitterGpu {
    position_spread: vec4<f32>,
    direction_emissive: vec4<f32>,
    lifetime_speed: vec4<f32>,
    size_gravity_drag: vec4<f32>,
    volume_ground: vec4<f32>,
    // kind, seed, has_ground, color count
    header: vec4<u32>,
    // slot base, max particles, 0, 0
    alloc: vec4<u32>,
    reserved: vec4<u32>,
    colors: array<vec4<f32>, 8>,
}

struct StepGpu {
    dt: f32,
    sim_time: f32,
    step_index: u32,
    emitter_count: u32,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> emitters: array<EmitterGpu>;
@group(0) @binding(2) var<storage, read> slot_emitter: array<u32>;
@group(0) @binding(3) var<storage, read> spawns: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> claims: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> sim_step: StepGpu;

const MAX_EMITTERS: u32 = 64u;
const SIM_PI: f32 = 3.14159265358979;

const KIND_SPLASH: u32 = 0u;
const KIND_SNOW: u32 = 1u;
const KIND_CONFETTI: u32 = 2u;
const KIND_SPARKLES: u32 = 3u;

const BOUNCE_RESTITUTION: f32 = 0.3;
const BOUNCE_FRICTION: f32 = 0.7;
const CONFETTI_LIFT: f32 = 0.7;
const CONFETTI_SWAY: f32 = 0.5;
const SNOW_DRIFT_SPEED: f32 = 0.35;
const SNOW_NOISE_SCALE: f32 = 0.4;
const SNOW_NOISE_SPEED: f32 = 0.05;
const SPARKLE_DRIFT_SPEED: f32 = 0.08;

// PCG-RXS-M-XS (Jarzynski and Olano 2020), the twin of spawn.rs `pcg_hash`.
fn pcg_hash(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// 24-bit uniform in [0, 1), then advances the hash state (spawn.rs `draw`).
fn sim_draw(h: ptr<function, u32>) -> f32 {
    let u = f32(*h >> 8u) / 16777216.0;
    *h = pcg_hash(*h);
    return u;
}

// a * (1 - t) + b * t, written out so it matches spawn.rs `mix` exactly.
fn sim_mix(a: f32, b: f32, t: f32) -> f32 {
    return a * (1.0 - t) + b * t;
}

fn spawn_particle(em: EmitterGpu, serial: u32) -> Particle {
    var h = pcg_hash(em.header.y ^ pcg_hash(serial));
    let u0 = sim_draw(&h);
    let u1 = sim_draw(&h);
    let u2 = sim_draw(&h);
    let u3 = sim_draw(&h);
    let u4 = sim_draw(&h);
    let u5 = sim_draw(&h);
    let u6 = sim_draw(&h);
    let u7 = sim_draw(&h);
    let u8 = sim_draw(&h);
    let color_hash = h;

    let half = em.volume_ground.xyz;
    let position = vec3<f32>(
        em.position_spread.x + (2.0 * u0 - 1.0) * half.x,
        em.position_spread.y + (2.0 * u1 - 1.0) * half.y,
        em.position_spread.z + (2.0 * u2 - 1.0) * half.z,
    );

    // The record's direction is already normalised by EmitterGpu::from_desc, exactly as
    // spawn.rs normalises desc.direction.
    let n = em.direction_emissive.xyz;
    let sgn = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (sgn + n.z);
    let b = n.x * n.y * a;
    let t = vec3<f32>(1.0 + sgn * n.x * n.x * a, sgn * b, -sgn * n.x);
    let bt = vec3<f32>(b, sgn + n.y * n.y * a, -n.y);
    let cos_spread = cos(em.position_spread.w);
    let cos_theta = sim_mix(1.0, cos_spread, u3);
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let phi = 2.0 * SIM_PI * u4;
    let l = vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
    let dir = vec3<f32>(
        t.x * l.x + bt.x * l.y + n.x * l.z,
        t.y * l.x + bt.y * l.y + n.y * l.z,
        t.z * l.x + bt.z * l.y + n.z * l.z,
    );

    let speed = sim_mix(em.lifetime_speed.z, em.lifetime_speed.w, u5);
    let life = sim_mix(em.lifetime_speed.x, em.lifetime_speed.y, u6);
    let size = sim_mix(em.size_gravity_drag.x, em.size_gravity_drag.y, u7);
    let rotation = 2.0 * SIM_PI * u8;

    let color_count = em.header.w;
    var color_index = 0u;
    if (color_count != 0u) {
        color_index = color_hash % color_count;
    }

    var p: Particle;
    p.pos_age = vec4<f32>(position, 0.0);
    p.vel_life = vec4<f32>(dir.x * speed, dir.y * speed, dir.z * speed, life);
    p.size = size;
    p.rotation = rotation;
    p.color_index = color_index;
    p.kind = em.header.x;
    return p;
}

// Uniform in [0, 1) from a 32-bit hash.
fn sim_unit(h: u32) -> f32 {
    return f32(h >> 8u) / 16777216.0;
}

fn sim_lattice(c: vec3<i32>) -> f32 {
    let h = pcg_hash(bitcast<u32>(c.x) ^ pcg_hash(bitcast<u32>(c.y) ^ pcg_hash(bitcast<u32>(c.z))));
    return sim_unit(h);
}

// Trilinear value noise with a smoothstep fade: (value, d/dx, d/dy, d/dz).
fn sim_value_noise_d(p: vec3<f32>) -> vec4<f32> {
    let i = floor(p);
    let f = p - i;
    let u = f * f * (3.0 - 2.0 * f);
    let du = 6.0 * f * (1.0 - f);
    let c = vec3<i32>(i);
    let n000 = sim_lattice(c);
    let n100 = sim_lattice(c + vec3<i32>(1, 0, 0));
    let n010 = sim_lattice(c + vec3<i32>(0, 1, 0));
    let n110 = sim_lattice(c + vec3<i32>(1, 1, 0));
    let n001 = sim_lattice(c + vec3<i32>(0, 0, 1));
    let n101 = sim_lattice(c + vec3<i32>(1, 0, 1));
    let n011 = sim_lattice(c + vec3<i32>(0, 1, 1));
    let n111 = sim_lattice(c + vec3<i32>(1, 1, 1));
    let k1 = n100 - n000;
    let k2 = n010 - n000;
    let k3 = n001 - n000;
    let k4 = n000 - n100 - n010 + n110;
    let k5 = n000 - n010 - n001 + n011;
    let k6 = n000 - n100 - n001 + n101;
    let k7 = -n000 + n100 + n010 - n110 + n001 - n101 - n011 + n111;
    let value = n000 + k1 * u.x + k2 * u.y + k3 * u.z + k4 * u.x * u.y + k5 * u.y * u.z
        + k6 * u.z * u.x + k7 * u.x * u.y * u.z;
    let grad = du * vec3<f32>(
        k1 + k4 * u.y + k6 * u.z + k7 * u.y * u.z,
        k2 + k5 * u.z + k4 * u.x + k7 * u.z * u.x,
        k3 + k6 * u.x + k5 * u.y + k7 * u.x * u.y,
    );
    return vec4<f32>(value, grad);
}

// Curl of the vector potential (Nx, Ny, Nz), three decorrelated value-noise channels.
fn sim_curl_noise(p: vec3<f32>) -> vec3<f32> {
    let gx = sim_value_noise_d(p).yzw;
    let gy = sim_value_noise_d(p + vec3<f32>(31.416, 47.853, 12.679)).yzw;
    let gz = sim_value_noise_d(p + vec3<f32>(-23.519, 61.327, -38.947)).yzw;
    return vec3<f32>(gz.y - gy.z, gx.z - gz.x, gy.x - gx.y);
}

// Kind acceleration (m/s^2) of a live particle; confetti also spins `rotation`.
fn kind_force(kind: u32, gravity: f32, ph: u32, p: ptr<function, Particle>) -> vec3<f32> {
    if (kind == KIND_CONFETTI) {
        // Flutter: spin 4..10 rad/s (random sense), lift against gravity from the
        // flake's orientation and a sideways sway along a per-particle heading.
        let spin = sim_mix(4.0, 10.0, sim_unit(ph)) * select(-1.0, 1.0, (ph & 1u) == 0u);
        (*p).rotation = (*p).rotation + spin * sim_step.dt;
        let r = (*p).rotation;
        let heading = 2.0 * SIM_PI * sim_unit(pcg_hash(ph));
        let side = vec3<f32>(cos(heading), 0.0, sin(heading));
        let lift = CONFETTI_LIFT * gravity * abs(cos(r));
        return vec3<f32>(0.0, lift, 0.0) + side * (CONFETTI_SWAY * gravity * sin(r));
    }
    return vec3<f32>(0.0);
}

// Kind drift velocity (m/s) added to the integrated velocity for the position update.
fn kind_drift(kind: u32, position: vec3<f32>, ph: u32) -> vec3<f32> {
    if (kind == KIND_SNOW) {
        let q = position * SNOW_NOISE_SCALE + vec3<f32>(sim_step.sim_time * SNOW_NOISE_SPEED);
        let c = sim_curl_noise(q);
        return c * (SNOW_DRIFT_SPEED / max(length(c), 1e-4));
    }
    if (kind == KIND_SPARKLES) {
        let phase = vec3<f32>(
            sim_unit(ph),
            sim_unit(pcg_hash(ph)),
            sim_unit(pcg_hash(ph ^ 0x9e3779b9u)),
        ) * (2.0 * SIM_PI);
        let t = sim_step.sim_time;
        let w = vec3<f32>(sin(t * 0.7 + phase.x), sin(t * 0.9 + phase.y), sin(t * 1.1 + phase.z));
        return w * SPARKLE_DRIFT_SPEED;
    }
    // Splash is plain ballistic (stretched at draw time).
    return vec3<f32>(0.0);
}

@compute @workgroup_size(64)
fn cs_sim(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= arrayLength(&slot_emitter) || slot >= arrayLength(&particles)) {
        return;
    }
    // A free slot holds u32::MAX, which is never below the emitter count.
    let e = slot_emitter[slot];
    if (e >= sim_step.emitter_count || e >= MAX_EMITTERS) {
        return;
    }
    let em = emitters[e];
    var p = particles[slot];

    if (p.pos_age.w >= p.vel_life.w) {
        let cmd_index = sim_step.step_index * MAX_EMITTERS + e;
        if (cmd_index >= arrayLength(&spawns)) {
            return;
        }
        let cmd = spawns[cmd_index];
        if (cmd.y == 0u) {
            return;
        }
        let j = atomicAdd(&claims[cmd_index], 1u);
        if (j < cmd.y) {
            particles[slot] = spawn_particle(em, cmd.x + j);
        }
        return;
    }

    let dt = sim_step.dt;
    let gravity = em.size_gravity_drag.z;
    let drag = em.size_gravity_drag.w;
    // Per-particle hash, stable over its life (its life is drawn at spawn).
    let ph = pcg_hash(bitcast<u32>(p.vel_life.w) ^ pcg_hash(slot));

    var vel = p.vel_life.xyz;
    vel = vel + vec3<f32>(0.0, -gravity, 0.0) * dt + kind_force(p.kind, gravity, ph, &p) * dt;
    vel = vel * exp(-drag * dt);
    var pos = p.pos_age.xyz + (vel + kind_drift(p.kind, p.pos_age.xyz, ph)) * dt;

    if (em.header.z != 0u && pos.y < em.volume_ground.w) {
        pos.y = em.volume_ground.w;
        if (vel.y < 0.0) {
            vel.y = -BOUNCE_RESTITUTION * vel.y;
            vel.x = vel.x * BOUNCE_FRICTION;
            vel.z = vel.z * BOUNCE_FRICTION;
        }
    }

    p.pos_age = vec4<f32>(pos, p.pos_age.w + dt);
    p.vel_life = vec4<f32>(vel, p.vel_life.w);
    particles[slot] = p;
}
