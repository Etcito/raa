// RAA shadow caster pass (S11): vertex stage only, depth written by the fixed function.
// Group 1 holds the shadow map this pass renders, so the pass reads only groups 0 and 2.

// Light view-projection of the map being rendered (one cascade per draw pass; the owner
// of the pass selects it with a dynamic offset).
struct RaaShadowCasterGpu {
    view_proj: mat4x4<f32>,
}

@group(2) @binding(0) var<uniform> raa_shadow_caster: RaaShadowCasterGpu;

// material_vertex, then the deform slot (local space), then the model matrix, as the lit
// passes do, so casters match what is shaded.
@vertex
fn vs_main(
    v0: RaaVertex0,
    v1: RaaVertex1,
    inst: RaaInstance,
    @builtin(instance_index) instance: u32,
) -> @builtin(position) vec4<f32> {
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
    var clip = raa_shadow_caster.view_proj * (model * vec4<f32>(d.local_position, 1.0));
    // Pancaking: casters in front of the (orthographic) near plane are flattened onto it
    // instead of being clipped away.
    clip.z = max(clip.z, 0.0);
    return clip;
}
