// Draw the reflected HDR over the receiver's visible plane using the forward depth buffer.
struct Receiver {
    view_proj: mat4x4<f32>,
    world: mat4x4<f32>,
    plane: vec4<f32>,
    inverse_size: vec4<f32>,
}

@group(0) @binding(0) var<uniform> receiver: Receiver;
@group(0) @binding(1) var reflection: texture_2d<f32>;
@group(0) @binding(2) var reflection_sampler: sampler;

struct Varyings {
    @builtin(position) position: vec4<f32>,
    @location(0) world: vec3<f32>,
}

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> Varyings {
    let world = receiver.world * vec4<f32>(position, 1.0);
    return Varyings(receiver.view_proj * world, world.xyz);
}

@fragment
fn fs_main(in: Varyings) -> @location(0) vec4<f32> {
    if abs(dot(receiver.plane, vec4<f32>(in.world, 1.0))) > 0.001 {
        discard;
    }
    let uv = in.position.xy * receiver.inverse_size.xy;
    let reflected = textureSampleLevel(reflection, reflection_sampler, vec2<f32>(1.0 - uv.x, uv.y), 0.0);
    // The transparent clear and opaque capture draws form premultiplied coverage.
    return reflected * 0.92;
}
