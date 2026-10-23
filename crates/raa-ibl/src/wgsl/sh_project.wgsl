// SH9 projection of the source sky cube by compute reduction (raa-ibl filter). Standalone
// compute module (no PRELUDE), own bindings in group 0. Same projection as the CPU
// reference src/sh.rs: texel directions (face order +X,-X,+Y,-Y,+Z,-Z, v grows downwards),
// exact texel solid angle, real SH basis order 2, cosine-lobe convolution A_l / pi.
//
// `sh_partial`: workgroup 8x8 over source mip 2 (64 px faces), global invocation z = face,
// dispatch (ceil(size / 8), ceil(size / 8), 6). A shared-memory tree reduction writes one
// record per workgroup at (wg.z * groups.y + wg.y) * groups.x + wg.x.
// `sh_final`: dispatch (1, 1, 1). One workgroup sums every record of `partials` (its
// arrayLength, so the buffer holds exactly one record per sh_partial workgroup) and writes
// the 9 coefficients (rgb, w = 0) in the layout of EnvSh.sh.

// Sums of one sh_partial workgroup per coefficient: rgb = radiance x basis x solid angle,
// w = solid angle.
struct ShPartial {
    c: array<vec4<f32>, 9>,
}

// Mip 2 of the source cube viewed as a 6-layer 2d array (one mip, read with textureLoad).
@group(0) @binding(0) var src_mip: texture_2d_array<f32>;
@group(0) @binding(1) var<storage, read_write> partials: array<ShPartial>;
@group(0) @binding(2) var<storage, read_write> sh_out: array<vec4<f32>, 9>;

// Invocations per workgroup of both entry points.
const WG_SIZE: u32 = 64u;

// Cosine-lobe factor (A_l / pi) per coefficient, src/sh.rs COSINE.
const SH_COSINE = array<f32, 9>(
    1.0,
    2.0 / 3.0,
    2.0 / 3.0,
    2.0 / 3.0,
    0.25,
    0.25,
    0.25,
    0.25,
    0.25,
);

var<workgroup> scratch: array<array<vec4<f32>, 9>, 64>;

// Unit direction through the centre of texel (x, y) of a `size` px face (src/sh.rs).
fn face_direction(face: u32, x: u32, y: u32, size: u32) -> vec3<f32> {
    let s = f32(max(size, 1u));
    let u = 2.0 * (f32(x) + 0.5) / s - 1.0;
    let v = 2.0 * (f32(y) + 0.5) / s - 1.0;
    var d: vec3<f32>;
    switch face {
        case 0u: { d = vec3<f32>(1.0, -v, -u); }
        case 1u: { d = vec3<f32>(-1.0, -v, u); }
        case 2u: { d = vec3<f32>(u, 1.0, v); }
        case 3u: { d = vec3<f32>(u, -1.0, -v); }
        case 4u: { d = vec3<f32>(u, -v, 1.0); }
        default: { d = vec3<f32>(-u, -v, -1.0); }
    }
    return normalize(d);
}

fn area_element(x: f32, y: f32) -> f32 {
    return atan2(x * y, sqrt(x * x + y * y + 1.0));
}

// Exact solid angle of texel (x, y) of a `size` px face (src/sh.rs).
fn texel_solid_angle(x: u32, y: u32, size: u32) -> f32 {
    let s = f32(max(size, 1u));
    let x0 = 2.0 * f32(x) / s - 1.0;
    let y0 = 2.0 * f32(y) / s - 1.0;
    let x1 = 2.0 * (f32(x) + 1.0) / s - 1.0;
    let y1 = 2.0 * (f32(y) + 1.0) / s - 1.0;
    return area_element(x0, y0) - area_element(x0, y1) - area_element(x1, y0)
        + area_element(x1, y1);
}

// The 9 real SH basis functions (order 2) at a unit direction (src/sh.rs).
fn sh_basis(d: vec3<f32>) -> array<f32, 9> {
    let x = d.x;
    let y = d.y;
    let z = d.z;
    return array<f32, 9>(
        0.282095,
        0.488603 * y,
        0.488603 * z,
        0.488603 * x,
        1.092548 * x * y,
        1.092548 * y * z,
        0.315392 * (3.0 * z * z - 1.0),
        1.092548 * x * z,
        0.546274 * (x * x - y * y),
    );
}

// Tree reduction of scratch[0..64] into scratch[0]; called by every invocation.
fn reduce_scratch(li: u32) {
    workgroupBarrier();
    for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (li < stride) {
            for (var k = 0u; k < 9u; k = k + 1u) {
                scratch[li][k] = scratch[li][k] + scratch[li + stride][k];
            }
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(8, 8, 1)
fn sh_partial(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let size = textureDimensions(src_mip).x;
    // Invocations outside the face contribute zero but still reach the barriers.
    var basis = array<f32, 9>();
    var rgb = vec3<f32>(0.0);
    var w = 0.0;
    if (id.x < size && id.y < size && id.z < 6u) {
        rgb = textureLoad(src_mip, id.xy, id.z, 0).rgb;
        w = texel_solid_angle(id.x, id.y, size);
        basis = sh_basis(face_direction(id.z, id.x, id.y, size));
    }
    for (var k = 0u; k < 9u; k = k + 1u) {
        scratch[li][k] = vec4<f32>(rgb * (basis[k] * w), w);
    }
    reduce_scratch(li);
    if (li == 0u) {
        let index = (wg.z * groups.y + wg.y) * groups.x + wg.x;
        for (var k = 0u; k < 9u; k = k + 1u) {
            partials[index].c[k] = scratch[0][k];
        }
    }
}

@compute @workgroup_size(64, 1, 1)
fn sh_final(@builtin(local_invocation_index) li: u32) {
    let count = arrayLength(&partials);
    var acc = array<vec4<f32>, 9>();
    for (var i = li; i < count; i = i + WG_SIZE) {
        for (var k = 0u; k < 9u; k = k + 1u) {
            acc[k] = acc[k] + partials[i].c[k];
        }
    }
    for (var k = 0u; k < 9u; k = k + 1u) {
        scratch[li][k] = acc[k];
    }
    reduce_scratch(li);
    if (li == 0u) {
        var cosine = SH_COSINE;
        for (var k = 0u; k < 9u; k = k + 1u) {
            sh_out[k] = vec4<f32>(scratch[0][k].rgb * cosine[k], 0.0);
        }
    }
}
