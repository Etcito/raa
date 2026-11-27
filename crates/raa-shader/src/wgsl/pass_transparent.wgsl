// RAA transparent pass (S11): shaded fragments appended to the per-pixel linked list of
// group 2 (raa-oit resolves it). The pass binds no colour target.

// One list node (16 B). The colour is straight (not premultiplied) linear HDR radiance.
struct OitNode {
    // pack2x16float(colour.rg).
    color_rg: u32,
    // pack2x16float(vec2(colour.b, alpha)).
    color_ba: u32,
    // Fragment depth in [0, 1] (frag_coord.z).
    depth: f32,
    // Index of the node appended before this one at the pixel, RAA_OIT_END for the last.
    next: u32,
}

// (width, height, capacity, 0): list size in pixels and node pool capacity.
struct OitParams {
    width: u32,
    height: u32,
    capacity: u32,
    reserved: u32,
}

// Empty head / end of list (heads are cleared to it every frame).
const RAA_OIT_END: u32 = 0xffffffffu;

@group(2) @binding(0) var<storage, read_write> raa_oit_heads: array<atomic<u32>>;
@group(2) @binding(1) var<storage, read_write> raa_oit_nodes: array<OitNode>;
@group(2) @binding(2) var<storage, read_write> raa_oit_counter: atomic<u32>;
@group(2) @binding(3) var<uniform> raa_oit_params: OitParams;

struct RaaVaryings {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) local_position: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec4<f32>,
    @location(4) uv: vec2<f32>,
    @location(5) color: vec4<f32>,
    @location(6) @interpolate(flat, either) material_index: u32,
    @location(7) @interpolate(flat, either) instance_index: u32,
}

// material_vertex, then the deform slot (local space), then the model and normal matrices.
// MaterialInput.local_position is the material_vertex output (before deform).
@vertex
fn vs_main(
    v0: RaaVertex0,
    v1: RaaVertex1,
    @location(2) raa_instance_id: u32,
) -> RaaVaryings {
    let inst = raa_instance(raa_instance_id);
    let x = raa_extras[raa_instance_id];
    let time = raa_frame.camera_pos.w;
    var vin: VertexMaterialInput;
    vin.local_position = v0.position;
    vin.normal = v0.normal;
    vin.uv = v1.uv;
    vin.color = v1.color;
    vin.time = time;
    vin.instance_params = x.params;
    vin.record = raa_materials[inst.material_index];
    let mv = material_vertex(vin);
    let d = raa_slot_deform(mv.local_position, mv.normal, x, time);
    let model = mat4x4<f32>(inst.model0, inst.model1, inst.model2, inst.model3);
    let normal_matrix = mat3x3<f32>(inst.normal0.xyz, inst.normal1.xyz, inst.normal2.xyz);
    let world = model * vec4<f32>(d.local_position, 1.0);
    var out: RaaVaryings;
    out.clip = raa_frame.view_proj * world;
    out.world_position = world.xyz;
    out.local_position = mv.local_position;
    out.normal = normal_matrix * d.normal;
    out.tangent = vec4<f32>((model * vec4<f32>(v1.tangent.xyz, 0.0)).xyz, v1.tangent.w);
    out.uv = v1.uv;
    out.color = v1.color;
    out.material_index = inst.material_index;
    out.instance_index = raa_instance_id;
    return out;
}

// MaterialInput of a fragment; the geometric normal is flipped on back faces of
// DOUBLE_SIDED materials.
fn raa_pass_material_input(v: RaaVaryings, front_facing: bool) -> MaterialInput {
    let record = raa_materials[v.material_index];
    var n = normalize(v.normal);
    if (!front_facing && (record.header.z & RAA_FLAG_DOUBLE_SIDED) != 0u) {
        n = -n;
    }
    var in: MaterialInput;
    in.position = v.world_position;
    in.local_position = v.local_position;
    in.normal = n;
    in.tangent = v.tangent;
    in.uv = v.uv;
    in.color = v.color;
    in.view_dir = normalize(raa_frame.camera_pos.xyz - v.world_position);
    in.time = raa_frame.camera_pos.w;
    in.front_facing = front_facing;
    in.frag_coord = v.clip;
    in.instance_params = raa_extras[v.instance_index].params;
    in.record = record;
    return in;
}

// material_main followed by the instance tint (S10). mat_instance_tint and
// mat_instance_tint_set are set first so material graph code can read them (S23).
fn raa_pass_material(in: MaterialInput, instance: u32) -> MaterialOutput {
    let x = raa_extras[instance];
    raa_set_instance_tint(x);
    return raa_apply_tint(material_main(in), x);
}

// Flat model or UNLIT flag: the fragment is base + emissive.
fn raa_pass_unlit(record: SurfaceMaterialGpu) -> bool {
    return record.header.x == RAA_MODEL_FLAT || (record.header.z & RAA_FLAG_UNLIT) != 0u;
}

// Normal-incidence reflectance: dielectric from the ior (0.04 at ior 1.5), base colour for
// metals.
fn raa_pass_f0(s: MaterialOutput) -> vec3<f32> {
    let ior = clamp(s.ior, 1.0, 3.0);
    let r = (ior - 1.0) / (ior + 1.0);
    return mix(vec3<f32>(r * r), s.base_color, clamp(s.metallic, 0.0, 1.0));
}

fn raa_pass_shade_ctx(s: MaterialOutput, in: MaterialInput, ao: f32, shadow: f32) -> ShadeCtx {
    var c: ShadeCtx;
    c.material_flags = in.record.header.z;
    c.position = in.position;
    c.view_dir = in.view_dir;
    c.frag = in.frag_coord;
    c.n_dot_v = max(dot(s.normal, in.view_dir), 1e-4);
    c.ao = min(s.ao, ao);
    c.shadow = shadow;
    c.f0 = raa_pass_f0(s);
    return c;
}

// Appends one fragment to the list of its pixel. Fragments behind the opaque depth copy
// are dropped here: storage writes are side effects a late depth test cannot undo.
fn raa_pass_oit_append(frag: vec4<f32>, color: vec3<f32>, alpha: f32) {
    if (alpha <= 0.0) {
        return;
    }
    let px = vec2<u32>(frag.xy);
    if (px.x >= raa_oit_params.width || px.y >= raa_oit_params.height) {
        return;
    }
    // Clamped so the 1x1 dummy (depth 1) keeps every fragment.
    let depth_px = min(px, textureDimensions(raa_opaque_depth) - vec2<u32>(1u));
    if (frag.z >= textureLoad(raa_opaque_depth, depth_px, 0).r) {
        return;
    }
    // The counter keeps counting past the capacity so the host sees the overflow.
    let index = atomicAdd(&raa_oit_counter, 1u);
    if (index >= raa_oit_params.capacity) {
        return;
    }
    let head = atomicExchange(&raa_oit_heads[px.y * raa_oit_params.width + px.x], index);
    let c = clamp(color, vec3<f32>(0.0), vec3<f32>(65504.0));
    raa_oit_nodes[index] = OitNode(
        pack2x16float(c.rg),
        pack2x16float(vec2<f32>(c.b, alpha)),
        frag.z,
        head,
    );
}

// Screen reflection of a smooth transmissive fragment (water, glass): the reflected ray is
// marched in world space against the opaque depth copy (group 1 binding 10) and the opaque
// colour copy (binding 9) is read where it passes behind the copy's surface. The result feeds
// ShadeCtx.screen_reflection: the ambient providers use it in place of the environment (sky
// cube occluded by the probes) by its weight, so the Fresnel split-sum weighting is unchanged.
// The weight fades with the roughness, at the screen borders and towards the march distance;
// a ray that leaves the screen, goes behind the camera or meets nothing has weight 0.
const RAA_TRANSPARENT_SSR_STEPS: u32 = 48u;
const RAA_TRANSPARENT_SSR_REFINE: u32 = 6u;
const RAA_TRANSPARENT_SSR_MAX_ROUGHNESS: f32 = 0.3;
const RAA_TRANSPARENT_SSR_DISTANCE: f32 = 40.0;

// View depth (metres in front of the camera) of the opaque copy at `uv`; 1e30 for the far plane.
fn raa_transparent_opaque_depth(uv: vec2<f32>, size: vec2<u32>) -> f32 {
    let px = min(vec2<u32>(uv * vec2<f32>(size)), size - vec2<u32>(1u));
    let depth = textureLoad(raa_opaque_depth, px, 0).r;
    if depth >= 1.0 { return 1e30; }
    let v = raa_frame.inv_proj * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    return -v.z / v.w;
}

// Screen uv and view depth of world point p: (u, v, depth, 0), or w = -1 when p is behind the camera.
fn raa_transparent_project(p: vec3<f32>) -> vec4<f32> {
    let clip = raa_frame.view_proj * vec4<f32>(p, 1.0);
    if clip.w <= 1e-4 { return vec4<f32>(0.0, 0.0, 0.0, -1.0); }
    let ndc = clip.xy / clip.w;
    let depth = -(raa_frame.view * vec4<f32>(p, 1.0)).z;
    return vec4<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, depth, 0.0);
}

fn raa_transparent_screen_reflection(c: ShadeCtx, s: MaterialOutput, geometric: vec3<f32>) -> vec4<f32> {
    let size = textureDimensions(raa_opaque_depth);
    if size.x <= 1u || s.roughness >= RAA_TRANSPARENT_SSR_MAX_ROUGHNESS { return vec4<f32>(0.0); }
    var r = reflect(-c.view_dir, s.normal);
    // A perturbed normal near grazing can send the ray under the surface: keep it above.
    let lift = dot(r, geometric);
    if lift < 0.02 { r = normalize(r + geometric * (0.02 - lift)); }
    let origin = c.position + geometric * 0.01;
    // Interleaved gradient noise: a per-pixel offset of the march samples (no banding).
    let dither = fract(52.9829189 * fract(dot(c.frag.xy, vec2<f32>(0.06711056, 0.00583715))));
    var prev = 0.0;
    var hit = -1.0;
    for (var i = 0u; i < RAA_TRANSPARENT_SSR_STEPS; i++) {
        let k = (f32(i) + dither) / f32(RAA_TRANSPARENT_SSR_STEPS);
        let t = RAA_TRANSPARENT_SSR_DISTANCE * k * k + 0.02;
        let q = raa_transparent_project(origin + r * t);
        if q.w < 0.0 || any(q.xy < vec2<f32>(0.0)) || any(q.xy > vec2<f32>(1.0)) { break; }
        let behind = q.z - raa_transparent_opaque_depth(q.xy, size);
        if behind > 0.0 && behind < max(0.12, t - prev) + 0.02 * t {
            // Bisect between the last sample in front and this one.
            var lo = prev;
            var hi = t;
            for (var j = 0u; j < RAA_TRANSPARENT_SSR_REFINE; j++) {
                let mid = 0.5 * (lo + hi);
                let m = raa_transparent_project(origin + r * mid);
                if m.z - raa_transparent_opaque_depth(m.xy, size) > 0.0 { hi = mid; } else { lo = mid; }
            }
            hit = hi;
            break;
        }
        prev = t;
    }
    if hit < 0.0 { return vec4<f32>(0.0); }
    let q = raa_transparent_project(origin + r * hit);
    let edge = smoothstep(0.0, 0.08, min(min(q.x, q.y), min(1.0 - q.x, 1.0 - q.y)));
    let far = 1.0 - smoothstep(0.7 * RAA_TRANSPARENT_SSR_DISTANCE, RAA_TRANSPARENT_SSR_DISTANCE, hit);
    let gloss = 1.0 - smoothstep(0.5 * RAA_TRANSPARENT_SSR_MAX_ROUGHNESS, RAA_TRANSPARENT_SSR_MAX_ROUGHNESS, s.roughness);
    let max_lod = f32(textureNumLevels(raa_opaque_color) - 1u);
    let color = textureSampleLevel(raa_opaque_color, raa_linear_clamp, q.xy, min(s.roughness * 16.0, max_lod)).rgb;
    return vec4<f32>(color, edge * far * gloss);
}

// S11 final colour plus transmission, then fog; alpha is the material opacity. A transmissive
// fragment takes no screen AO (the AO texture holds the opaque surface behind it) and reflects
// the opaque copy (raa_transparent_screen_reflection). With DDGI the fragment's own probe lookup
// (eight probes, raa_ddgi_sample: the screen resolve holds the opaque surface behind) blends the
// diffuse ambient and occludes the environment specular, as raa_opaque_lighting does.
@fragment
fn fs_main(v: RaaVaryings, @builtin(front_facing) front_facing: bool) {
    let in = raa_pass_material_input(v, front_facing);
    let s = raa_pass_material(in, v.instance_index);
    let unlit = raa_pass_unlit(in.record);
    let transmissive = s.transmission > 0.0;
    let shadow = raa_slot_shadow(in.position, in.normal, in.frag_coord);
    let ao = select(raa_slot_ao(in.frag_coord), 1.0, transmissive);
    var c = raa_pass_shade_ctx(s, in, ao, shadow);
    if transmissive {
        // raa_sky_sun_peak: the environment's sun follows this fragment's sun visibility.
        c.material_flags |= RAA_FLAG_TRANSMISSIVE;
    }
    if !unlit && raa_feature_enabled(RAA_DDGI_GI_FEATURE) && !raa_feature_enabled(RAA_DIRECT_ONLY_FEATURE)
        && raa_ddgi_active() {
        c.spherical_diffuse = raa_ddgi_sample(c.position, s.normal, c.view_dir);
    }
    if transmissive && !unlit {
        c.screen_reflection = raa_transparent_screen_reflection(c, s, in.normal);
    }
    let direct = raa_slot_direct(s, c, -raa_frame.sun_dir.xyz,
        raa_directional_radiance(raa_frame.sun_color.rgb) * shadow);
    let local = raa_local_lighting(s, c);
    var indirect = raa_slot_indirect(s, c);
    if c.spherical_diffuse.w > 0.0 {
        indirect *= 1.0 - c.spherical_diffuse.w;
    }
    indirect += raa_slot_ambient(s, c);
    let lit = s.emissive + direct + local + indirect + raa_slot_transmission(s, c);
    let shaded = raa_slot_fog(lit, in.position, c);
    var color = select(shaded, s.base_color + s.emissive, unlit);
    if raa_feature_enabled(RAA_DIRECT_ONLY_FEATURE) {
        color = select(s.emissive + direct + local, s.base_color + s.emissive, unlit);
    } else if raa_feature_enabled(RAA_INDIRECT_ONLY_FEATURE) {
        color = select(indirect, vec3<f32>(0.0), unlit);
    }
    raa_pass_oit_append(in.frag_coord, color, clamp(s.opacity, 0.0, 1.0));
}
