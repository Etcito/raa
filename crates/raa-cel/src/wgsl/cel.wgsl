// CelPost is exactly 112 bytes. Full-resolution phase: settings.w (factor) = 1.
struct CelPost {
 settings: vec4f,
 palette: vec4f,
 toon: vec4f,
 ink: vec4f,
 post: vec4f,
 image: vec4u,
 grid: vec4i,
}
struct CelCamera { inv_proj:mat4x4f, inv_view:mat4x4f }
@group(0) @binding(0) var<uniform> params:CelPost;
@group(0) @binding(1) var source:texture_2d<f32>;
@group(0) @binding(2) var depth:texture_depth_2d;
@group(0) @binding(3) var normals:texture_2d<f32>;
@group(0) @binding(4) var<uniform> camera:CelCamera;

@vertex
fn vs_main(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {
 let p=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));
 return vec4f(p[i],0,1);
}
@fragment
fn fs_cel(@builtin(position) position:vec4f)->@location(0) vec4f {
 let cell=vec2i(position.xy);
 let factor=max(1,i32(params.settings.w));
 var sum=vec3f(0);var inkSum=0.0;var inkMax=0.0;var count=0.0;
 var nearest=1.0;var world=vec3f(0);var hit=false;
 for(var y=0;y<factor;y++){for(var x=0;x<factor;x++){
  let p=cel_coord(cell*factor+vec2i(x,y));
  // HDR input is already linear. Average before any palette or display conversion.
  sum+=textureLoad(source,p,0).rgb;count+=1.0;
  let coverage=cel_coverage(p,factor);inkSum+=coverage;inkMax=max(inkMax,coverage);
  let d=cel_depth(p);
  if(d<nearest){nearest=d;world=(camera.inv_view*vec4f(cel_view(p,d),1)).xyz;hit=true;}
 }}
 let original=sum/count;
 var rgb=linear_to_srgb(original);
 let coverage=mix(inkSum/count,inkMax,params.post.x);
 let ink=mix(rgb*0.38,vec3f(0.025,0.028,0.04),0.5);
 if(params.ink.w>0.5){rgb=mix(rgb,ink,coverage);}
 if(hit){rgb=cel_quantize(rgb,cell+params.grid.xy,world,hit,true);}
 var linear=srgb_to_linear(rgb);
 if(!hit && coverage<=0.0){return vec4f(original,1);}
 // Protected ink is composited in linear space after palette and thresholding.
 if(params.ink.w<0.5){linear=mix(linear,srgb_to_linear(ink),coverage);}
 return vec4f(linear,1);
}
