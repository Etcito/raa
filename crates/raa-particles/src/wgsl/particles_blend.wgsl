// particles.blend: splash droplets (premultiplied, env fresnel) and sparkles (additive
// emissive) over a copy of the HDR colour. Appended to PRELUDE + LIB + the slot winners by
// `lit_module`, so groups 0 and 1 come from the prelude; group 2 is private to this pass.
//
// Vertex pulling as in particles.opaque: one instance per particle slot, 6 vertices per
// instance, vertex_index / 6 = emitter id. A dead slot collapses to a degenerate triangle.
// Entry points: vs_splash + fs_splash (One / OneMinusSrcAlpha) and vs_sparkles +
// fs_sparkles (One / One). Both fade softly against the single-sample prepass depth.

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
    // opacity (splash droplet body), 0, 0, 0
    body: vec4<f32>,
    colors: array<vec4<f32>, 8>,
}

@group(2) @binding(0) var<storage, read> particles_state: array<ParticleState>;
@group(2) @binding(1) var<storage, read> particles_emitters: array<ParticleEmitterGpu>;
@group(2) @binding(2) var particles_depth: texture_depth_2d;

// Extra length of a splash droplet per m/s of speed.
const PARTICLES_STRETCH_PER_SPEED: f32 = 0.03;
// Distance over which a particle fades in front of the scene depth (m).
const PARTICLES_SOFT_DISTANCE: f32 = 0.1;
// Water F0 of the splash fresnel.
const PARTICLES_SPLASH_F0: f32 = 0.02;
const PARTICLES_SPLASH_ALPHA: f32 = 0.65;
// A sparkle quad is this many times larger than the particle size, so the star spikes
// reach out of the bright core (the core keeps the particle size).
const PARTICLES_SPARKLE_SCALE: f32 = 3.5;

struct ParticlesBlendVaryings {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    // Quad corner in [-1, 1]^2 (x across, y along the stretch axis).
    @location(1) corner: vec2<f32>,
    // Quad basis: across, along, towards the camera (constant over a quad).
    @location(2) side: vec3<f32>,
    @location(3) axis: vec3<f32>,
    @location(4) facing: vec3<f32>,
    @location(5) @interpolate(flat, either) color: vec3<f32>,
    @location(6) @interpolate(flat, either) emissive: f32,
    // Per-slot hash in [0, 1).
    @location(7) @interpolate(flat, either) hash: f32,
    // Splash droplet body opacity in [0, 1] (EmitterDesc.opacity).
    @location(8) @interpolate(flat, either) opacity: f32,
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

// The quad of `slot` for `vertex`: a droplet stretched along the velocity when `stretch`,
// else a camera-facing square of the particle size.
fn particles_blend_vertex(vertex: u32, slot: u32, stretch: bool) -> ParticlesBlendVaryings {
    let p = particles_state[slot];
    let eid = min(vertex / 6u, arrayLength(&particles_emitters) - 1u);
    let corner = particles_corner(vertex % 6u);
    let center = p.pos_age.xyz;

    var side = raa_frame.inv_view[0].xyz;
    var axis = raa_frame.inv_view[1].xyz;
    var facing = raa_frame.inv_view[2].xyz;
    var extent = vec2<f32>(0.5 * p.size, 0.5 * p.size);
    if (stretch) {
        // Velocity projected on the plane facing the camera; a droplet moving straight
        // at the camera keeps the camera-facing basis.
        let to_cam = normalize(raa_frame.camera_pos.xyz - center);
        let v = p.vel_life.xyz;
        let speed = length(v);
        let along = v - to_cam * dot(v, to_cam);
        if (length(along) > 1e-4) {
            axis = normalize(along);
            side = normalize(cross(axis, to_cam));
            facing = to_cam;
        }
        extent = vec2<f32>(0.5 * p.size, 0.5 * (p.size + speed * PARTICLES_STRETCH_PER_SPEED));
    } else {
        extent = extent * PARTICLES_SPARKLE_SCALE;
    }

    let world = center + side * (corner.x * extent.x) + axis * (corner.y * extent.y);
    let alive = p.pos_age.w < p.vel_life.w && p.size > 0.0;
    var out: ParticlesBlendVaryings;
    out.clip = select(vec4<f32>(0.0, 0.0, 0.0, 1.0), raa_frame.view_proj * vec4<f32>(world, 1.0), alive);
    out.world_position = world;
    out.corner = corner;
    out.side = side;
    out.axis = axis;
    out.facing = facing;
    let count = max(particles_emitters[eid].header.w, 1u);
    out.color = particles_emitters[eid].colors[min(p.color_index, count - 1u) & 7u].rgb;
    out.emissive = particles_emitters[eid].direction_emissive.w;
    out.hash = particles_unit(particles_hash(slot));
    out.opacity = clamp(particles_emitters[eid].body.x, 0.0, 1.0);
    return out;
}

// View-space distance of the point at pixel `frag_xy` and clip depth `depth`, linearised
// through the inverse projection.
fn particles_view_distance(frag_xy: vec2<f32>, depth: f32) -> f32 {
    let uv = frag_xy * raa_frame.viewport.zw;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let v = raa_frame.inv_proj * ndc;
    return length(v.xyz / v.w);
}

// Soft depth fade: saturate((scene distance - particle distance) / 0.1 m); 1 over the
// cleared depth (nothing behind the particle).
fn particles_soft_fade(frag: vec4<f32>) -> f32 {
    let dims = vec2<i32>(textureDimensions(particles_depth));
    let pixel = clamp(vec2<i32>(frag.xy), vec2<i32>(0), dims - vec2<i32>(1));
    let scene_depth = textureLoad(particles_depth, pixel, 0);
    if (scene_depth >= 1.0) {
        return 1.0;
    }
    let scene = particles_view_distance(frag.xy, scene_depth);
    let particle = particles_view_distance(frag.xy, frag.z);
    return saturate((scene - particle) / PARTICLES_SOFT_DISTANCE);
}

@vertex
fn vs_splash(
    @builtin(vertex_index) vertex: u32,
    @builtin(instance_index) slot: u32,
) -> ParticlesBlendVaryings {
    return particles_blend_vertex(vertex, slot, true);
}

@vertex
fn vs_sparkles(
    @builtin(vertex_index) vertex: u32,
    @builtin(instance_index) slot: u32,
) -> ParticlesBlendVaryings {
    return particles_blend_vertex(vertex, slot, false);
}

// The env specular radiance along `dir` (mip 0: a droplet is a mirror); the flat ambient
// while no env cube is installed.
fn particles_env(dir: vec3<f32>) -> vec3<f32> {
    let env = textureSampleLevel(raa_env_specular, raa_linear_clamp, dir, 0.0).rgb * raa_env_sh.info.x;
    return select(raa_frame.ambient.rgb, env, raa_env_sh.info.w > 0.5);
}

// Splash droplet, blended by the emitter's body opacity `o`: o = 0 is a clear water droplet
// (Schlick fresnel x env reflection + sun glint + a small tinted diffuse, alpha 0.65); o = 1
// an opaque wet blob of the palette colour (Lambert sun + ambient + a sky-tinted diffuse,
// under the same fresnel reflection and glint, alpha 1). Premultiplied by alpha x fade x
// disc mask, through the fog slot.
@fragment
fn fs_splash(v: ParticlesBlendVaryings) -> @location(0) vec4<f32> {
    let fade = particles_soft_fade(v.clip);
    let r2 = dot(v.corner, v.corner);
    let disc = 1.0 - smoothstep(0.6, 1.0, sqrt(r2));
    // Sphere-like normal over the quad.
    let n = normalize(
        normalize(v.side) * v.corner.x + normalize(v.axis) * v.corner.y
            + normalize(v.facing) * sqrt(max(1.0 - r2, 0.0)),
    );
    let view_dir = normalize(raa_frame.camera_pos.xyz - v.world_position);
    let n_dot_v = max(dot(n, view_dir), 1e-4);
    let fresnel = PARTICLES_SPLASH_F0 + (1.0 - PARTICLES_SPLASH_F0) * pow(1.0 - n_dot_v, 5.0);
    let r = reflect(-view_dir, n);
    let l = -raa_frame.sun_dir.xyz;
    let sun = raa_frame.sun_color.rgb;
    let glint = sun * pow(max(dot(r, l), 0.0), 256.0) * 4.0 * fresnel;
    let lit = raa_frame.ambient.rgb + sun * max(dot(n, l), 0.0) / 3.14159265;
    let water = fresnel * particles_env(r) + glint + 0.08 * v.color * lit;
    // The opaque body keeps a wet skin: the env reflection at fresnel strength and a
    // broader sun highlight on top of the lit palette colour.
    let sky = 0.25 * particles_env(n);
    let sheen = sun * pow(max(dot(r, l), 0.0), 48.0) * 0.6;
    let body = v.color * (lit + sky) + fresnel * particles_env(r) + glint + sheen;
    let shaded = mix(water, body, v.opacity);

    var c: ShadeCtx;
    c.position = v.world_position;
    c.view_dir = view_dir;
    c.frag = v.clip;
    c.n_dot_v = n_dot_v;
    c.ao = 1.0;
    c.shadow = 1.0;
    c.f0 = vec3<f32>(PARTICLES_SPLASH_F0);
    let fogged = raa_slot_fog(shaded, v.world_position, c);
    let alpha = mix(PARTICLES_SPLASH_ALPHA, 1.0, v.opacity) * fade * disc;
    if (fade <= 0.0) {
        discard;
    }
    return vec4<f32>(fogged * alpha, alpha);
}

// Additive emissive spark: a four-point star (hot core + two thin crossed spikes, turned
// per particle) x colour x emissive x twinkle, softly faded. The twinkle breathes the spike
// length as well as the brightness, so the stars read as glints, not round orbs.
@fragment
fn fs_sparkles(v: ParticlesBlendVaryings) -> @location(0) vec4<f32> {
    let fade = particles_soft_fade(v.clip);
    let time = raa_frame.camera_pos.w;
    let wave = 0.5 + 0.5 * sin(time * (7.0 + 7.0 * v.hash) + v.hash * 6.28);
    let twinkle = 0.12 + 0.88 * pow(wave, 5.0);
    let q = v.corner * PARTICLES_SPARKLE_SCALE;
    let turn = v.hash * 1.5708;
    let cs = cos(turn);
    let sn = sin(turn);
    let r = vec2<f32>(cs * q.x - sn * q.y, sn * q.x + cs * q.y);
    let core = exp(-6.0 * dot(q, q));
    let reach = 0.35 + 0.65 * twinkle;
    let spike_x = exp(-26.0 * r.y * r.y) * exp(-1.1 * abs(r.x) / reach);
    let spike_y = exp(-26.0 * r.x * r.x) * exp(-1.1 * abs(r.y) / reach);
    let edge = 1.0 - smoothstep(0.7, 1.0, max(abs(v.corner.x), abs(v.corner.y)));
    let shape = (1.6 * core + 0.9 * (spike_x + spike_y)) * edge;
    if (fade <= 0.0) {
        discard;
    }
    let glow = max(v.color, vec3<f32>(0.0)) * max(v.emissive, 0.0) * twinkle * shape * fade;
    return vec4<f32>(glow, 0.0);
}
