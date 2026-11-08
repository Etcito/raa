fn raa_cel_linear_to_srgb(c:vec3f)->vec3f {
 return select(1.055*pow(c,vec3f(1.0/2.4))-0.055,c*12.92,c<=vec3f(0.0031308));
}
// Direct lighting is selected per material, with layered BRDF fallback.
fn raa_cel_palette(material:vec3f, light:vec3f, band:u32)->vec3f {
 var anchor=clamp(material,vec3f(0),vec3f(1));
 if(max(max(anchor.x,anchor.y),anchor.z)<0.06){anchor=vec3f(0.12,0.12,0.14);}
 let tint=vec3f(1.0,0.90,0.74)*0.7+clamp(light,vec3f(0),vec3f(1))*0.3;
 let ramp=array<vec3f,6>(
  anchor*0.18000000+vec3f(0.15380392,0.14792157,0.21372549)+tint*0.00000000,
  anchor*0.38000000+vec3f(0.16870588,0.12752941,0.12745098)+tint*0.00000000,
  anchor*0.65000000+vec3f(0.13470588,0.06509804,0.04019608)+tint*0.00000000,
  anchor*1.00000000+vec3f(0.00000000,0.00000000,0.00000000)+tint*0.00000000,
  anchor*0.72000000+vec3f(0.04109804,-0.03286275,-0.04641569)+tint*0.28000000,
  anchor*0.40000000+vec3f(-0.08784314,-0.08666667,-0.01262745)+tint*0.60000000 );
 return clamp(ramp[min(band,5u)],vec3f(0),vec3f(1));
}
fn raa_slot_direct(s:MaterialOutput,c:ShadeCtx,l:vec3f,radiance:vec3f)->vec3f {
 let control=raa_env.reserved[0];
 if((c.material_flags & RAA_FLAG_CEL)==0u || control.x<0.5){
  return raa_cel_fallback(s,c,l,radiance);
 }
 let nl=dot(s.normal,l);
 let soft=max(0.00001,control.y*0.12);
 let thresholds=array<f32,3>(0.05,0.32,0.68);
 // Authored ramp is display RGB; convert at the lighting boundary.
 let base=raa_cel_linear_to_srgb(clamp(s.base_color,vec3f(0),vec3f(1)));
 var color=raa_cel_palette(base,radiance,1u);
 for(var i=0u;i<3u;i++){
  color=mix(color,raa_cel_palette(base,radiance,i+2u),smoothstep(thresholds[i]-soft,thresholds[i]+soft,nl));
 }
 // Deep shadow and shaped highlight make the outer ramp rows available without curvature.
 color=mix(raa_cel_palette(base,radiance,0u),color,smoothstep(-0.35-soft,-0.35+soft,nl));
 let half_vector=l+c.view_dir;
 let h=half_vector/max(length(half_vector),0.00001);
 let spec_edge=mix(0.995,0.72,clamp(s.roughness,0.0,1.0));
 let highlight=smoothstep(spec_edge-soft*0.25,spec_edge+soft*0.25,dot(s.normal,h))*max(nl,0.0);
 color=mix(color,raa_cel_palette(base,radiance,5u),clamp(highlight*control.z,0.0,1.0));
 return raa_srgb_to_linear(color)*radiance;
}
