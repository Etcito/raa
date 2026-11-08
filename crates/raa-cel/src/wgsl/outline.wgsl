fn cel_coord(p:vec2i)->vec2i {return clamp(p,vec2i(0),vec2i(textureDimensions(source))-1);}
fn cel_depth(p:vec2i)->f32 {return textureLoad(depth,cel_coord(p),0);}
fn cel_view(p:vec2i,d:f32)->vec3f {
 let uv=(vec2f(p)+0.5)/vec2f(textureDimensions(source));
 let v=camera.inv_proj*vec4f(uv*vec2f(2,-2)+vec2f(-1,1),d,1);
 return v.xyz/v.w;
}
fn cel_outline(p:vec2i)->f32 {
 let d=cel_depth(p);
 if(d>=1.0 || params.ink.x<=0.0){return 0.0;}
 let point=cel_view(p,d);
 let normal=textureLoad(normals,cel_coord(p),0).xyz;
 let pixelWorld=max(length(cel_view(p+vec2i(1,0),d)-point),0.00001);
 var ink=0.0;
 for(var band=0;band<2;band++) {
  let interior=band==1;
  if(interior && params.ink.z<0.5){continue;}
  let radius=params.ink.x*select(1.0,0.5,interior);
  for(var ring=0;ring<2;ring++) {
   let distance=floor(radius)+f32(ring);
   let weight=select(1.0,fract(radius),ring==1);
   if(distance<1.0 || weight<=0.0){continue;}
   let feather=mix(clamp((radius-distance+1.0)/max(1.0,radius),0.0,1.0),1.0,params.ink.y);
   for(var a=0;a<8;a++) {
    let angle=f32(a)*0.7853981634;
    let q=cel_coord(p+vec2i(round(vec2f(cos(angle),sin(angle))*distance)));
    let other=cel_depth(q);
    var edge=select(0.0,1.0,!interior && other>=1.0);
    if(interior && other<1.0){
     let neighbour=cel_view(q,other);
     let nn=textureLoad(normals,q,0).xyz;
     let planeBreak=abs(dot(neighbour-point,normal))/(pixelWorld*max(1.0,radius));
     edge=max(smoothstep(0.5,1.5,planeBreak),smoothstep(0.2,0.65,1.0-dot(normal,nn)));
    }
    ink=max(ink,pow(edge*feather,mix(0.5,3.0,params.ink.y))*weight);
   }
  }
 }
 return ink;
}
fn cel_coverage(p:vec2i,factor:i32)->f32 {
 let centre=cel_outline(p);
 if(params.ink.y>=1.0 || params.ink.x<=0.0){return centre;}
 var filtered=centre*4.0;
 for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
  if(x==0 && y==0){continue;}
  let weight=select(1.0,2.0,x==0)*select(1.0,2.0,y==0);
  filtered+=cel_outline(cel_coord(p+vec2i(x,y)*factor))*weight;
 }}
 return mix(centre,filtered/16.0,1.0-params.ink.y);
}
