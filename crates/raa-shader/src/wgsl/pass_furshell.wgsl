// RAA fur shell pass (S11, contributor of forward): shell `index` of the group-2 uniform
// extruded along the normal; the fragment alpha is the strand coverage, turned into
// sample coverage by alpha-to-coverage (MSAA) or a discard at 1 sample.

// Shell being drawn: 0-based index below the material's shell count (fur.w).
struct RaaFurShellGpu {
    index: u32,
    reserved0: u32,
    reserved1: u32,
    reserved2: u32,
}

@group(2) @binding(0) var<uniform> raa_fur_shell: RaaFurShellGpu;

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
    // Shell height in (0, 1]: 1 at the strand tips.
    @location(8) shell_height: f32,
}

// material_vertex, then the deform slot (local space), then the model and normal matrices;
// the shell is pushed out along the world normal by fur length x height and droops by
// droop x length x height^2 along -Y.
@vertex
fn vs_main(
    v0: RaaVertex0,
    v1: RaaVertex1,
    inst: RaaInstance,
    @builtin(instance_index) instance: u32,
) -> RaaVaryings {
    let x = raa_extras[instance];
    let record = raa_materials[inst.material_index];
    let time = raa_frame.camera_pos.w;
    var vin: VertexMaterialInput;
    vin.local_position = v0.position;
    vin.normal = v0.normal;
    vin.uv = v1.uv;
    vin.color = v1.color;
    vin.time = time;
    vin.instance_params = x.params;
    vin.record = record;
    let mv = material_vertex(vin);
    let d = raa_slot_deform(mv.local_position, mv.normal, x, time);
    let model = mat4x4<f32>(inst.model0, inst.model1, inst.model2, inst.model3);
    let normal_matrix = mat3x3<f32>(inst.normal0.xyz, inst.normal1.xyz, inst.normal2.xyz);
    let n = normalize(normal_matrix * d.normal);
    let h = (f32(raa_fur_shell.index) + 1.0) / max(record.fur.w, 1.0);
    let fur_length = max(record.fur.x, 0.0);
    let world = (model * vec4<f32>(d.local_position, 1.0)).xyz
        + n * (fur_length * h)
        - vec3<f32>(0.0, record.fur.z * fur_length * h * h, 0.0);
    var out: RaaVaryings;
    out.clip = raa_frame.view_proj * vec4<f32>(world, 1.0);
    out.world_position = world;
    out.local_position = mv.local_position;
    out.normal = n;
    out.tangent = vec4<f32>((model * vec4<f32>(v1.tangent.xyz, 0.0)).xyz, v1.tangent.w);
    out.uv = v1.uv;
    out.color = v1.color;
    out.material_index = inst.material_index;
    out.instance_index = instance;
    out.shell_height = h;
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

// material_main followed by the instance tint (S10).
fn raa_pass_material(in: MaterialInput, instance: u32) -> MaterialOutput {
    return raa_apply_tint(material_main(in), raa_extras[instance]);
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

// Strand coverage in [0, 1]: one strand per Voronoi cell of the local position before
// deform (so every shell samples the same strands), cell size from the density in
// strands per cm2, the radius tapering to 0 at the tips; the edge is antialiased over one
// pixel.
fn raa_pass_fur_coverage(local_position: vec3<f32>, record: SurfaceMaterialGpu, h: f32) -> f32 {
    let strands_per_m2 = max(record.fur.y, 1e-3) * 1e4;
    let cell = inverseSqrt(strands_per_m2);
    let dist = raa_voronoi3(local_position / cell).x;
    let radius = 0.5 * (1.0 - h);
    return clamp((radius - dist) / max(fwidth(dist), 1e-4) + 0.5, 0.0, 1.0);
}

// S11 final colour with fur self-shadowing towards the roots, then fog; alpha = strand
// coverage x opacity.
@fragment
fn fs_main(v: RaaVaryings, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
    let in = raa_pass_material_input(v, front_facing);
    let coverage = raa_pass_fur_coverage(in.local_position, in.record, v.shell_height);
    let s = raa_pass_material(in, v.instance_index);
    let shadow = raa_slot_shadow(in.position, in.normal, in.frag_coord);
    let ao = raa_slot_ao(in.frag_coord);
    let c = raa_pass_shade_ctx(s, in, ao, shadow);
    let root = mix(0.35, 1.0, v.shell_height);
    let lit = s.emissive
        + raa_slot_direct(s, c, -raa_frame.sun_dir.xyz, raa_frame.sun_color.rgb * shadow) * root
        + raa_slot_ambient(s, c) * min(s.ao, ao) * root;
    let shaded = raa_slot_fog(lit, in.position, c);
    let color = select(shaded, s.base_color + s.emissive, raa_pass_unlit(in.record));
    let alpha = coverage * clamp(s.opacity, 0.0, 1.0);
    // Alpha-to-coverage needs MSAA; a single-sample target keeps the covered half.
    if (RAA_MSAA <= 1u && alpha < 0.5) {
        discard;
    }
    return vec4<f32>(color, alpha);
}
