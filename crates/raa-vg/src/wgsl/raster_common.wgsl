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
// crates/solstice-render/shaders/vg/raster_common.wgsl:1-22.
// Bindings move to group 2 (contract `vg-raster-wgsl`); the scene records are the
// `pass_view.wgsl` twins, concatenated before this file by the module builder.

@group(2) @binding(0) var<uniform> view: PassView;
@group(2) @binding(1) var<storage, read> instances: array<VgInstance>;
@group(2) @binding(2) var<storage, read> clusters: array<VgCluster>;
@group(2) @binding(3) var<storage, read> vertices: array<PackedVertex>;
@group(2) @binding(4) var<storage, read> triangles: array<u32>;
@group(2) @binding(5) var<storage, read> visible: array<VisibleCluster>;
@group(2) @binding(6) var<storage, read> state: CullStateRead;

// Visibility ID: visible-cluster index in the high bits, triangle in the low 7.
fn encode_id(visible_index: u32, triangle: u32) -> u32 {
    return (visible_index << 7u) | triangle;
}

fn local_vertex(packed_triangle: u32, corner: u32) -> u32 {
    return (packed_triangle >> (corner * 8u)) & 0xffu;
}
