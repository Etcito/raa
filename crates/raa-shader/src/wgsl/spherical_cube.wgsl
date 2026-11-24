// Cell and capture faces share the CPU scheduler's +X -X +Y -Y +Z -Z convention.
fn raa_spherical_face_direction(face: u32, uv: vec2<f32>) -> vec3<f32> {
    switch face {
        case 0u: { return vec3<f32>( 1.0, uv.y, -uv.x); }
        case 1u: { return vec3<f32>(-1.0, uv.y,  uv.x); }
        case 2u: { return vec3<f32>(uv.x,  1.0, -uv.y); }
        case 3u: { return vec3<f32>(uv.x, -1.0,  uv.y); }
        case 4u: { return vec3<f32>( uv.x, uv.y,  1.0); }
        default: { return vec3<f32>(-uv.x, uv.y, -1.0); }
    }
}

// Returns face coordinates in xy and the face index in z. Resolve ties X, then Y, then Z.
fn raa_spherical_face_uv(v: vec3<f32>) -> vec3<f32> {
    let a = abs(v);
    if a.x >= a.y && a.x >= a.z {
        return vec3<f32>(select(v.z, -v.z, v.x >= 0.0) / a.x,
            v.y / a.x, select(1.0, 0.0, v.x >= 0.0));
    }
    if a.y >= a.z {
        return vec3<f32>(v.x / a.y, select(v.z, -v.z, v.y >= 0.0) / a.y,
            select(3.0, 2.0, v.y >= 0.0));
    }
    return vec3<f32>(select(-v.x, v.x, v.z >= 0.0) / a.z,
        v.y / a.z, select(5.0, 4.0, v.z >= 0.0));
}
