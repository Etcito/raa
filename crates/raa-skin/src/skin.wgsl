// Four storage bindings, including both outputs, for compatibility WebGPU.
// Rest record: position[3], normal[3], tangent[4], packed joints[2], weights[4].
@group(0) @binding(0) var<storage, read> rest: array<u32>;
@group(0) @binding(1) var<storage, read> pose: array<mat4x4<f32>>;
// Scalar arrays preserve S1's tightly packed 24-byte stride.
@group(0) @binding(2) var<storage, read_write> stream0: array<f32>;
@group(0) @binding(3) var<storage, read_write> tangents: array<u32>;

fn direction(v: vec3<f32>) -> vec3<f32> {
    let length = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if length > 0.0 { return v / length; }
    return vec3<f32>(0.0);
}

@compute @workgroup_size(64)
fn skin_vertices(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= arrayLength(&rest) / 16u { return; }
    let base = i * 16u;
    var m = mat4x4<f32>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    for (var k = 0u; k < 4u; k++) {
        let packed = rest[base + 10u + k / 2u];
        let joint = (packed >> ((k % 2u) * 16u)) & 65535u;
        let w = bitcast<f32>(rest[base + 12u + k]);
        for (var c = 0u; c < 4u; c++) { m[c] += w * pose[joint][c]; }
    }
    let p = vec3<f32>(bitcast<f32>(rest[base]), bitcast<f32>(rest[base+1u]), bitcast<f32>(rest[base+2u]));
    let n = vec3<f32>(bitcast<f32>(rest[base+3u]), bitcast<f32>(rest[base+4u]), bitcast<f32>(rest[base+5u]));
    let t = vec3<f32>(bitcast<f32>(rest[base+6u]), bitcast<f32>(rest[base+7u]), bitcast<f32>(rest[base+8u]));
    let position = m[0].xyz*p.x + m[1].xyz*p.y + m[2].xyz*p.z + m[3].xyz;
    let normal = direction(m[0].xyz*n.x + m[1].xyz*n.y + m[2].xyz*n.z);
    let tangent = direction(m[0].xyz*t.x + m[1].xyz*t.y + m[2].xyz*t.z);
    for (var c = 0u; c < 3u; c++) {
        stream0[i*6u+c] = position[c];
        stream0[i*6u+3u+c] = normal[c];
        tangents[i*4u+c] = bitcast<u32>(tangent[c]);
    }
    tangents[i*4u+3u] = rest[base+9u];
}
