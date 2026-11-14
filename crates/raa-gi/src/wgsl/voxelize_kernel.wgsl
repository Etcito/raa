// View-independent voxelization kernel. Reads the globals raa_gi_voxels and raa_gi_voxelize that each
// driver declares.
//
// Adapted from solstice@7e465b5, crates/solstice-render/shaders/gi/voxelize.wgsl:41-156 and common/math.wgsl:17-28. MIT License
//
//
// Copyright (c) 2026 SamG-Coder
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

const RAA_GI_EMISSIVE_RANGE: f32 = 128.0;

struct GiVoxelizeGpu {
    origin: vec3<f32>,
    voxel_size: f32,
    resolution: u32,
    level_offset: u32,
    _pad0: u32,
    _pad1: u32,
}

struct RaaGiVoxelData {
    albedo: u32,
    normal: u32,
    emissive: u32,
}

fn raa_gi_encode_emissive(e: vec3<f32>) -> u32 {
    let peak = clamp(max(e.x, max(e.y, e.z)) / RAA_GI_EMISSIVE_RANGE, 1.0e-6, 1.0);
    let scale = ceil(peak * 255.0) / 255.0;
    return pack4x8unorm(vec4<f32>(e / (RAA_GI_EMISSIVE_RANGE * scale), scale));
}

// Signed octahedral coordinates in [-1, 1]^2.
fn raa_gi_oct_encode(n: vec3<f32>) -> vec2<f32> {
    let p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    if n.z < 0.0 {
        let s = vec2<f32>(select(-1.0, 1.0, p.x >= 0.0), select(-1.0, 1.0, p.y >= 0.0));
        return (vec2<f32>(1.0) - abs(p.yx)) * s;
    }
    return p;
}

fn raa_gi_write_voxel(coord: vec3<i32>, data: RaaGiVoxelData) {
    let n = i32(raa_gi_voxelize.resolution);
    if any(coord < vec3<i32>(0)) || any(coord >= vec3<i32>(n)) {
        return;
    }
    let index = u32(coord.z * n * n + coord.y * n + coord.x) + raa_gi_voxelize.level_offset;
    atomicStore(&raa_gi_voxels[index * 3u], data.albedo);
    atomicStore(&raa_gi_voxels[index * 3u + 1u], data.normal);
    atomicStore(&raa_gi_voxels[index * 3u + 2u], data.emissive);
}

// Narrows the span of X values that satisfy `slope * x + offset >= 0`.
fn raa_gi_clip_span(span: vec2<f32>, slope: f32, offset: f32) -> vec2<f32> {
    if slope > 0.0 {
        return vec2<f32>(max(span.x, -offset / slope), span.y);
    }
    if slope < 0.0 {
        return vec2<f32>(span.x, min(span.y, -offset / slope));
    }
    // A horizontal edge: the whole row is either inside or out.
    return select(vec2<f32>(1.0, 0.0), span, offset >= 0.0);
}

fn raa_gi_voxelize_triangle(v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>, data: RaaGiVoxelData) {
    let normal = cross(v1 - v0, v2 - v0);
    let abs_normal = abs(normal);

    // The projection axis `w` is the one the triangle faces most directly.
    var u_axis = 0;
    var v_axis = 1;
    var w_axis = 2;
    if abs_normal.x >= abs_normal.y && abs_normal.x >= abs_normal.z {
        u_axis = 1;
        v_axis = 2;
        w_axis = 0;
    } else if abs_normal.y >= abs_normal.z {
        u_axis = 0;
        v_axis = 2;
        w_axis = 1;
    }
    if abs_normal[w_axis] < 1.0e-8 {
        return;
    }

    let a = vec2<f32>(v0[u_axis], v0[v_axis]);
    let b = vec2<f32>(v1[u_axis], v1[v_axis]);
    let c = vec2<f32>(v2[u_axis], v2[v_axis]);
    let limit = f32(raa_gi_voxelize.resolution);

    // Triangles wholly outside the grid need no scan at all.
    let box_min = min(min(a, b), c);
    let box_max = max(max(a, b), c);
    let w_min = min(min(v0[w_axis], v1[w_axis]), v2[w_axis]);
    let w_max = max(max(v0[w_axis], v1[w_axis]), v2[w_axis]);
    if any(box_max < vec2<f32>(-1.0)) || any(box_min > vec2<f32>(limit + 1.0)) || w_max < -1.0 || w_min > limit + 1.0 {
        return;
    }
    let lo = clamp(floor(box_min), vec2<f32>(0.0), vec2<f32>(limit - 1.0));
    let hi = clamp(floor(box_max), vec2<f32>(0.0), vec2<f32>(limit - 1.0));

    // Edges are oriented inside-positive and widened to cover any touched column.
    let orientation = select(-1.0, 1.0, (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x) >= 0.0);
    let widen = 0.5;
    let e0 = vec3<f32>(a.y - b.y, b.x - a.x, 0.0) * orientation;
    let e1 = vec3<f32>(b.y - c.y, c.x - b.x, 0.0) * orientation;
    let e2 = vec3<f32>(c.y - a.y, a.x - c.x, 0.0) * orientation;
    let bias0 = -dot(e0.xy, a) + (abs(e0.x) + abs(e0.y)) * widen;
    let bias1 = -dot(e1.xy, b) + (abs(e1.x) + abs(e1.y)) * widen;
    let bias2 = -dot(e2.xy, c) + (abs(e2.x) + abs(e2.y)) * widen;

    // Depth across the plane, and how much it varies within one column.
    let dw_du = -normal[u_axis] / normal[w_axis];
    let dw_dv = -normal[v_axis] / normal[w_axis];
    let w0 = v0[w_axis] - dw_du * a.x - dw_dv * a.y;
    let spread = 0.5 * (abs(dw_du) + abs(dw_dv));

    for (var y = lo.y; y <= hi.y; y += 1.0) {
        let py = y + 0.5;
        var span = vec2<f32>(lo.x + 0.5, hi.x + 0.5);
        span = raa_gi_clip_span(span, e0.x, e0.y * py + bias0);
        span = raa_gi_clip_span(span, e1.x, e1.y * py + bias1);
        span = raa_gi_clip_span(span, e2.x, e2.y * py + bias2);
        if span.x > span.y {
            continue;
        }
        let first_column = ceil(span.x - 0.5);
        let last_column = floor(span.y - 0.5);
        for (var x = first_column; x <= last_column; x += 1.0) {
            let w = w0 + dw_du * (x + 0.5) + dw_dv * py;
            let first = i32(floor(w - spread));
            let last = i32(floor(w + spread));
            for (var z = first; z <= last; z++) {
                var coord = vec3<i32>(0);
                coord[u_axis] = i32(x);
                coord[v_axis] = i32(y);
                coord[w_axis] = z;
                raa_gi_write_voxel(coord, data);
            }
        }
    }
}
