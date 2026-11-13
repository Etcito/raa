/*
MIT License

Copyright (c) 2026 SamG-Coder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
// Adapted from solstice@7e465b599c13da165948ebaa44df824c36c3ae07
// crates/solstice-render/shaders/vg/raster_hw.wgsl:1-58.
// No include: the builder concatenates `raster_common.wgsl` before this file.
//
// WebGPU has no mesh shaders, so a cluster is one instance of a draw with enough
// vertices for the largest cluster. Vertices past the cluster's real triangle count
// collapse to a degenerate point, which the rasterizer discards cheaply.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) id: u32,
}

fn cluster_vertex(vertex_index: u32, instance_index: u32) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(0.0);
    out.id = 0u;

    let visible_index = arrayLength(&visible) - 1u - (state.hw_base + instance_index);
    let entry = visible[visible_index];
    // The visible entry holds the absolute cluster index in its low 24 bits (S32).
    let cluster = clusters[visible_cluster_index(entry)];
    let triangle = vertex_index / 3u;
    if triangle >= cluster_triangle_count(cluster) {
        return out;
    }

    let instance = instances[entry.instance];
    // Cluster offsets are relative to the instance's own section of each pool (S29).
    let packed_triangle = triangles[instance.ranges.w + cluster.triangle_offset + triangle];
    var clip: array<vec4<f32>, 3>;
    for (var corner = 0u; corner < 3u; corner++) {
        let packed = vertices[instance.ranges.z + cluster.vertex_offset + local_vertex(packed_triangle, corner)];
        let world = transform_point(instance, decode_position(instance, packed));
        clip[corner] = view.raster_view_proj * vec4<f32>(world, 1.0);
    }

    // The pipeline draws both faces, since one draw holds both kinds of material. Front
    // faces are counter-clockwise, which makes this determinant positive. Working in
    // homogeneous coordinates keeps it right for triangles that cross behind the camera
    // (Olano and Greer 1997).
    let facing = determinant(mat3x3<f32>(clip[0].xyw, clip[1].xyw, clip[2].xyw));
    if facing <= 0.0 && !instance_two_sided(instance) {
        return out;
    }

    out.position = clip[vertex_index % 3u];
    out.id = encode_id(visible_index, triangle);
    return out;
}

@vertex
fn vs_visibility(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> VertexOutput {
    return cluster_vertex(v, i);
}

// Depth-only entry point, used by shadow maps.
@vertex
fn vs_depth(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> @builtin(position) vec4<f32> {
    return cluster_vertex(v, i).position;
}
