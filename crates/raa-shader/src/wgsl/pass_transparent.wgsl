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
    inst: RaaInstance,
    @builtin(instance_index) instance: u32,
) -> RaaVaryings {
    let x = raa_extras[instance];
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
    out.instance_index = instance;
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
    c.position = in.position;
    c.view_dir = in.view_dir;
    c.frag = in.frag_coord;
    c.n_dot_v = max(dot(s.normal, in.view_dir), 1e-4);
    c.ao = ao;
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

// S11 final colour plus transmission, then fog; alpha is the material opacity.
@fragment
fn fs_main(v: RaaVaryings, @builtin(front_facing) front_facing: bool) {
    let in = raa_pass_material_input(v, front_facing);
    let s = raa_pass_material(in, v.instance_index);
    let shadow = raa_slot_shadow(in.position, in.normal, in.frag_coord);
    let ao = raa_slot_ao(in.frag_coord);
    let c = raa_pass_shade_ctx(s, in, ao, shadow);
    let lit = s.emissive
        + raa_slot_direct(s, c, -raa_frame.sun_dir.xyz, raa_frame.sun_color.rgb * shadow)
        + raa_slot_ambient(s, c) * min(s.ao, ao)
        + raa_slot_transmission(s, c);
    let shaded = raa_slot_fog(lit, in.position, c);
    let color = select(shaded, s.base_color + s.emissive, raa_pass_unlit(in.record));
    raa_pass_oit_append(in.frag_coord, color, clamp(s.opacity, 0.0, 1.0));
}
