// particles.opaque: lit confetti and snow quads inside the forward MSAA pass. Appended to
// PRELUDE + LIB + the slot winners by `lit_module`, so groups 0 and 1 come from the
// prelude; group 2 is private to this contributor.
//
// Vertex pulling, no vertex buffers: one instance per particle slot (instance_index =
// slot) and 6 vertices per instance. The draw of emitter e uses vertices e*6 .. e*6+6, so
// vertex_index / 6 is the emitter id and vertex_index % 6 the quad corner. A dead slot
// (age >= life) collapses to a degenerate triangle.

struct ParticleState {
    pos_age: vec4<f32>,
    vel_life: vec4<f32>,
    size: f32,
    rotation: f32,
    color_index: u32,
    kind: u32,
}

struct ParticleEmitterGpu {
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

@group(2) @binding(0) var<storage, read> particles_state: array<ParticleState>;
@group(2) @binding(1) var<storage, read> particles_emitters: array<ParticleEmitterGpu>;

const PARTICLES_KIND_SNOW: u32 = 1u;
const PARTICLES_KIND_CONFETTI: u32 = 2u;
const PARTICLES_CONFETTI_ASPECT: f32 = 0.6;
const PARTICLES_SNOW_ROUGHNESS: f32 = 0.85;
const PARTICLES_CONFETTI_ROUGHNESS: f32 = 0.45;

struct ParticlesVaryings {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    // Quad normal (confetti) or camera-facing direction (snow), not yet flipped.
    @location(1) normal: vec3<f32>,
    // Quad corner in [-1, 1]^2.
    @location(2) corner: vec2<f32>,
    @location(3) @interpolate(flat, either) color: vec3<f32>,
    @location(4) @interpolate(flat, either) emissive: f32,
    @location(5) @interpolate(flat, either) kind: u32,
}

// PCG-RXS-M-XS, the hash of the simulation (sim.wgsl `pcg_hash`).
fn particles_hash(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn particles_unit(h: u32) -> f32 {
    return f32(h >> 8u) / 16777216.0;
}

// A unit vector from the slot hash (uniform on the sphere).
fn particles_axis(slot: u32) -> vec3<f32> {
    let h0 = particles_hash(slot ^ 0x9e3779b9u);
    let h1 = particles_hash(h0);
    let z = particles_unit(h0) * 2.0 - 1.0;
    let phi = particles_unit(h1) * 6.28318530718;
    let r = sqrt(max(1.0 - z * z, 0.0));
    return vec3<f32>(r * cos(phi), z, r * sin(phi));
}

// Rodrigues rotation of `x` by `angle` about the unit `axis`.
fn particles_rotate(x: vec3<f32>, axis: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return x * c + cross(axis, x) * s + axis * dot(axis, x) * (1.0 - c);
}

fn particles_corner(i: u32) -> vec2<f32> {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );
    return corners[min(i, 5u)];
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex: u32,
    @builtin(instance_index) slot: u32,
) -> ParticlesVaryings {
    let p = particles_state[slot];
    let eid = min(vertex / 6u, arrayLength(&particles_emitters) - 1u);
    let corner = particles_corner(vertex % 6u);
    let center = p.pos_age.xyz;
    let half_size = 0.5 * p.size;

    var right: vec3<f32>;
    var up: vec3<f32>;
    var normal: vec3<f32>;
    var extent: vec2<f32>;
    if (p.kind == PARTICLES_KIND_CONFETTI) {
        // Rectangle in the plane (normalize(axis + t), b), its normal at 45 degrees to the
        // spin axis, so the rotation about the axis tumbles the piece.
        let axis = particles_axis(slot);
        let helper = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(axis.x) > 0.9);
        let t = normalize(cross(axis, helper));
        let b = cross(axis, t);
        let u0 = normalize(axis + t);
        right = particles_rotate(u0, axis, p.rotation);
        up = particles_rotate(b, axis, p.rotation);
        normal = cross(right, up);
        extent = vec2<f32>(half_size, half_size * PARTICLES_CONFETTI_ASPECT);
    } else {
        // Camera-facing quad from the inverse view matrix.
        right = raa_frame.inv_view[0].xyz;
        up = raa_frame.inv_view[1].xyz;
        normal = raa_frame.inv_view[2].xyz;
        extent = vec2<f32>(half_size, half_size);
    }

    let world = center + right * (corner.x * extent.x) + up * (corner.y * extent.y);
    let alive = p.pos_age.w < p.vel_life.w && p.size > 0.0;
    var out: ParticlesVaryings;
    out.clip = select(vec4<f32>(0.0, 0.0, 0.0, 1.0), raa_frame.view_proj * vec4<f32>(world, 1.0), alive);
    out.world_position = world;
    out.normal = normal;
    // A snow flake's six-arm mask is turned per particle (rotation of the state record).
    let turn = select(0.0, p.rotation, p.kind == PARTICLES_KIND_SNOW);
    let cs = cos(turn);
    let sn = sin(turn);
    out.corner = vec2<f32>(cs * corner.x - sn * corner.y, sn * corner.x + cs * corner.y);
    let count = max(particles_emitters[eid].header.w, 1u);
    out.color = particles_emitters[eid].colors[min(p.color_index, count - 1u) & 7u].rgb;
    out.emissive = particles_emitters[eid].direction_emissive.w;
    out.kind = p.kind;
    return out;
}

// Snowflake mask over the quad corner coordinates: a soft round core with six short arms,
// so a flake reads as a crystal rather than a plain dot.
fn particles_snow_alpha(corner: vec2<f32>) -> f32 {
    let r = length(corner);
    let core = 1.0 - smoothstep(0.3, 0.5, r);
    let arms = pow(abs(cos(3.0 * atan2(corner.y, corner.x))), 8.0);
    let reach = arms * (1.0 - smoothstep(0.55, 1.0, r));
    return max(core, reach);
}

fn particles_material(v: ParticlesVaryings, view_dir: vec3<f32>) -> MaterialOutput {
    let snow = v.kind == PARTICLES_KIND_SNOW;
    var n = normalize(v.normal);
    // Facing the viewer (both sides of a confetti piece are lit as its front).
    n = select(n, -n, dot(n, view_dir) < 0.0);
    // Snow: half towards the sky, so a flake catches the sky light from above.
    let snow_n = normalize(n + vec3<f32>(0.0, 1.0, 0.0));
    var o: MaterialOutput;
    o.base_color = max(v.color, vec3<f32>(0.0));
    o.metallic = 0.0;
    o.roughness = select(PARTICLES_CONFETTI_ROUGHNESS, PARTICLES_SNOW_ROUGHNESS, snow);
    o.emissive = o.base_color * max(v.emissive, 0.0);
    o.normal = select(n, snow_n, snow);
    o.ao = 1.0;
    o.opacity = select(1.0, particles_snow_alpha(v.corner), snow);
    o.clearcoat = 0.0;
    o.clearcoat_roughness = 0.0;
    o.sheen_color = vec3<f32>(0.0);
    o.sheen_roughness = 1.0;
    o.transmission = 0.0;
    o.ior = 1.5;
    o.thickness = 0.0;
    o.attenuation_color = vec3<f32>(1.0);
    o.attenuation_distance = 1e30;
    return o;
}

// S11 opaque colour through the installed slots; every slot runs in uniform control flow
// before the msaa-1 mask discard. Alpha drives alpha-to-coverage at msaa > 1.
@fragment
fn fs_main(v: ParticlesVaryings) -> @location(0) vec4<f32> {
    let view_dir = normalize(raa_frame.camera_pos.xyz - v.world_position);
    let s = particles_material(v, view_dir);
    let shadow = raa_slot_shadow(v.world_position, s.normal, v.clip);
    let ao = raa_slot_ao(v.clip);
    var c: ShadeCtx;
    c.position = v.world_position;
    c.view_dir = view_dir;
    c.frag = v.clip;
    c.n_dot_v = max(dot(s.normal, view_dir), 1e-4);
    c.ao = ao;
    c.shadow = shadow;
    c.f0 = vec3<f32>(0.04);
    let color = s.emissive
        + raa_slot_direct(s, c, -raa_frame.sun_dir.xyz, raa_frame.sun_color.rgb * shadow)
        + raa_slot_ambient(s, c) * ao;
    // Without multisampling there is no alpha-to-coverage: the snow mask cuts at 0.5.
    if (raa_frame.flags.y <= 1u && s.opacity < 0.5) {
        discard;
    }
    return vec4<f32>(color, s.opacity);
}
