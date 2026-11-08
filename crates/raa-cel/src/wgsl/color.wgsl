
fn linear_to_srgb(c:vec3f)->vec3f {
 let x=clamp(c,vec3f(0),vec3f(1));return select(1.055*pow(x,vec3f(1.0/2.4))-0.055,x*12.92,x<=vec3f(0.0031308));
}
fn srgb_to_linear(c:vec3f)->vec3f {
 let x=clamp(c,vec3f(0),vec3f(1));return select(pow((x+0.055)/1.055,vec3f(2.4)),x/12.92,x<=vec3f(0.04045));
}
fn cel_oklab(rgb:vec3f)->vec3f {
 let lms=pow(max(vec3f(dot(rgb,vec3f(0.4122214708,0.5363325363,0.0514459929)),
  dot(rgb,vec3f(0.2119034982,0.6806995451,0.1073969566)),
  dot(rgb,vec3f(0.0883024619,0.2817188376,0.6299787005))),vec3f(0)),vec3f(1.0/3.0));
 return vec3f(dot(lms,vec3f(0.2104542553,0.7936177850,-0.0040720468)),
  dot(lms,vec3f(1.9779984951,-2.4285922050,0.4505937099)),
  dot(lms,vec3f(0.0259040371,0.7827717662,-0.8086757660)));
}
fn cel_from_oklab(lab:vec3f)->vec3f {
 let lms=vec3f(lab.x+0.3963377774*lab.y+0.2158037573*lab.z,
  lab.x-0.1055613458*lab.y-0.0638541728*lab.z,
  lab.x-0.0894841775*lab.y-1.2914855480*lab.z);
 let q=lms*lms*lms;
 return vec3f(dot(q,vec3f(4.0767416621,-3.3077115913,0.2309699292)),
  dot(q,vec3f(-1.2684380046,2.6097574011,-0.3413193965)),
  dot(q,vec3f(-0.0041960863,-0.7034186147,1.7076147010)));
}
fn cel_in_gamut(rgb:vec3f)->bool {return all(rgb>=vec3f(-0.00001)) && all(rgb<=vec3f(1.00001));}
// Reduce chroma toward neutral rather than clipping channels and changing hue.
fn cel_gamut(lab:vec3f)->vec3f {
 var rgb=cel_from_oklab(lab);
 if(cel_in_gamut(rgb)){return clamp(rgb,vec3f(0),vec3f(1));}
 var low=0.0; var high=1.0;
 for(var i=0u;i<6u;i++) {
  let mid=(low+high)*0.5;
  let trial=cel_from_oklab(vec3f(lab.x,lab.yz*mid));
  if(cel_in_gamut(trial)){low=mid;}else{high=mid;}
 }
 return clamp(cel_from_oklab(vec3f(lab.x,lab.yz*low)),vec3f(0),vec3f(1));
}
// An automatic perceptual palette: material texture and coloured illumination
// choose the occupied hue/chroma families. No readback, temporal clustering or
// fixed fantasy swatches. Near-neutral colours get finer saturation spacing.
fn cel_quantize(rgb:vec3f,ij:vec2i,world:vec3f,hit:bool,dither:bool)->vec3f {
 let levels=max(2.0,params.settings.y)-1.0;
 var threshold=0.5;
 if(dither){threshold=mix(0.5,cel_threshold(ij,world,hit),params.toon.w);}
 if(params.palette.x<0.5) {
  let l=dot(rgb,vec3f(0.2126,0.7152,0.0722));
  if(l<0.00001){return vec3f(0);}
  let q=clamp(floor(l*levels+1.0-threshold)/levels,0.0,1.0);
  return clamp(rgb*(q/l),vec3f(0),vec3f(1));
 }
 let lab=cel_oklab(srgb_to_linear(rgb));
 let lightness=clamp(floor(lab.x*levels+1.0-threshold)/levels,0.0,1.0);
 if(lightness<=0.0){return vec3f(0);}
 if(lightness>=1.0){return vec3f(1);}
 let chroma=length(lab.yz);
 let sectors=max(4.0,params.palette.y);
 let steps=max(2.0,params.palette.z);
 var hue=0.0;
 if(chroma>0.00001){hue=atan2(lab.z,lab.y);}
 let angle=floor(hue*sectors/(6.28318530718)+0.5)*(6.28318530718)/sectors;
 let ci=floor(sqrt(clamp(chroma/0.32,0.0,1.0))*steps+0.5)/steps;
 let c=0.32*ci*ci;
 return linear_to_srgb(cel_gamut(vec3f(lightness,c*cos(angle),c*sin(angle))));
}
