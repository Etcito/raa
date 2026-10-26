// Frame uses the renderer's existing 512-byte uniform.
struct Frame { vp: mat4x4<f32>, rest: array<vec4<f32>,28> }
@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(0) var depth: texture_depth_2d;
@group(1) @binding(1) var mask: texture_2d<f32>;
struct In {
    @location(0) p: vec3<f32>, @location(1) q: vec3<f32>,
    @location(2) m0: vec4<f32>, @location(3) m1: vec4<f32>,
    @location(4) m2: vec4<f32>, @location(5) m3: vec4<f32>,
    @location(6) wire: vec4<f32>, @location(7) outline: vec4<f32>,
}
struct Out {
    @builtin(position) p: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(linear) side: f32,
}
fn display(c:vec3<f32>) -> vec3<f32> {
    // sRGB surfaces perform their own output encoding.
    if bitcast<vec4<u32>>(frame.rest[22]).x == 0u {
        return select(c/12.92,pow((c+0.055)/1.055,vec3<f32>(2.4)),c>vec3<f32>(0.04045));
    }
    return c;
}
@vertex fn vs_wire(v:In, @builtin(vertex_index) k:u32) -> Out {
    let world=mat4x4<f32>(v.m0,v.m1,v.m2,v.m3);
    var a=frame.vp*world*vec4<f32>(v.p,1);
    var b=frame.vp*world*vec4<f32>(v.q,1);
    // Clip segments to the near plane before the perspective divide.
    if a.z<0 && b.z>=0 {a=mix(a,b,-a.z/(b.z-a.z));}
    if b.z<0 && a.z>=0 {b=mix(b,a,-b.z/(a.z-b.z));}
    var o:Out;
    if a.z<0 || b.z<0 || a.w<=0 || b.w<=0 {
        o.p=vec4<f32>(2,2,2,1);o.color=vec4<f32>(0);o.side=0;return o;
    }
    let size=frame.rest[20].xy;
    let delta=(b.xy/b.w-a.xy/a.w)*size;
    let n=vec2<f32>(-delta.y,delta.x)/max(length(delta),0.00001);
    let corners=array<vec2<f32>,6>(vec2<f32>(0,-1),vec2<f32>(1,-1),vec2<f32>(0,1),vec2<f32>(0,1),vec2<f32>(1,-1),vec2<f32>(1,1));
    let c=corners[k];
    var p=mix(a,b,c.x);
    p=vec4<f32>(p.xy+n*c.y*2.5/size*p.w,p.zw);
    var color=vec3<f32>(0.05,0.06,0.08);
    if v.wire.w>1.5 {
        color=select(vec3<f32>(0.65,0.72,0.82),color,dot(v.wire.rgb,vec3<f32>(0.2126,0.7152,0.0722))>0.5);
    }
    o.p=p;o.color=vec4<f32>(display(color),1);o.side=c.y*1.25;return o;
}
fn visible(p:vec4<f32>) -> bool {
    let z=textureLoad(depth,vec2<i32>(p.xy),0);
    // A slope-scaled tolerance handles quantisation on steep surfaces.
    return p.z <= z + clamp(1.5*(abs(dpdx(p.z))+abs(dpdy(p.z))),0.000002,0.00005);
}
@fragment fn fs_wire(v:Out) -> @location(0) vec4<f32> {
    if !visible(v.p) {discard;}
    let alpha=1.0-smoothstep(0.55,1.25,abs(v.side));
    return vec4<f32>(v.color.rgb,alpha);
}
struct MaskIn {
    @location(0) p:vec3<f32>,
    @location(2) m0:vec4<f32>,@location(3) m1:vec4<f32>,
    @location(4) m2:vec4<f32>,@location(5) m3:vec4<f32>,
    @location(6) wire:vec4<f32>, @location(7) outline:vec4<f32>,
}
fn geometry(v:MaskIn, color:vec4<f32>) -> Out {
    var o:Out;
    o.p=frame.vp*mat4x4<f32>(v.m0,v.m1,v.m2,v.m3)*vec4<f32>(v.p,1);
    o.color=color;o.side=0;return o;
}
@vertex fn vs_mask(v:MaskIn) -> Out {return geometry(v,v.outline);}
@vertex fn vs_only(v:MaskIn) -> Out {return geometry(v,vec4<f32>(v.wire.rgb,1));}
fn surface_visible(p:vec4<f32>) -> bool {
    return p.z <= textureLoad(depth,vec2<i32>(p.xy),0)+0.000002;
}
@fragment fn fs_only(v:Out) -> @location(0) vec4<f32> {
    if !surface_visible(v.p) {discard;}
    // Match tonemap's display background exactly, without its surface transform.
    return v.color;
}
@fragment fn fs_mask(v:Out) -> @location(0) vec4<f32> {
    if !surface_visible(v.p) {discard;}
    return v.color;
}
@vertex fn vs_full(@builtin(vertex_index) i:u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(f32((i<<1u)&2u)*2-1,f32(i&2u)*2-1,0,1);
}
@fragment fn fs_outline(@builtin(position) p:vec4<f32>) -> @location(0) vec4<f32> {
    let coord=vec2<i32>(p.xy);
    if textureLoad(mask,coord,0).a>0 {discard;}
    let size=vec2<i32>(textureDimensions(mask));
    var coverage=0.0;
    var color=vec3<f32>(0);
    for(var y=-3;y<=3;y++) {for(var x=-3;x<=3;x++) {
        let q=coord+vec2<i32>(x,y);
        if any(q<vec2<i32>(0)) || any(q>=size) {continue;}
        let c=textureLoad(mask,q,0);
        let a=c.a*(1-smoothstep(2.0,3.0,length(vec2<f32>(f32(x),f32(y)))));
        if a>coverage {coverage=a;color=c.rgb;}
    }}
    return vec4<f32>(display(color),coverage);
}
