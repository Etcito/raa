// Toroidal 16x16 void-and-cluster rank mask (Gaussian sigma 1.5).
const raa_cel_blue = array<u32,256>(
27u,255u,139u,96u,180u,210u,11u,230u,136u,66u,117u,49u,208u,135u,103u,191u,
162u,120u,64u,216u,45u,145u,107u,83u,204u,18u,235u,190u,2u,75u,233u,55u,
225u,37u,186u,1u,165u,237u,59u,185u,38u,169u,98u,129u,176u,34u,154u,91u,
137u,108u,242u,87u,116u,23u,132u,250u,151u,72u,222u,43u,252u,111u,215u,9u,
52u,205u,70u,153u,224u,195u,78u,6u,118u,203u,13u,85u,146u,60u,196u,172u,
248u,16u,178u,30u,53u,102u,163u,219u,47u,101u,159u,183u,228u,19u,81u,114u,
157u,95u,124u,202u,143u,234u,26u,187u,140u,239u,65u,29u,104u,133u,217u,35u,
231u,193u,76u,244u,7u,69u,126u,88u,32u,211u,125u,194u,245u,166u,61u,141u,
3u,58u,39u,161u,113u,212u,177u,254u,57u,171u,0u,84u,51u,14u,189u,86u,
173u,131u,214u,184u,94u,48u,12u,150u,115u,97u,236u,156u,128u,213u,110u,253u,
106u,227u,82u,17u,144u,232u,192u,77u,218u,28u,198u,68u,226u,41u,149u,24u,
63u,31u,155u,247u,67u,127u,25u,167u,44u,142u,179u,10u,100u,182u,74u,207u,
168u,197u,119u,42u,206u,105u,223u,92u,241u,123u,56u,251u,138u,21u,238u,134u,
5u,240u,89u,188u,8u,174u,50u,147u,4u,209u,79u,112u,170u,200u,93u,46u,
148u,109u,54u,158u,130u,249u,71u,199u,99u,181u,22u,220u,36u,62u,122u,221u,
73u,201u,20u,229u,80u,33u,121u,164u,40u,246u,152u,90u,160u,243u,15u,175u
);
fn cel_threshold(ij:vec2i, world:vec3f, hit:bool)->f32 {
 var coord=ij;
 if(hit && params.post.y>0.5){
  let c=vec3i(floor(world*params.post.z));
  coord=vec2i(c.x+c.z*3,c.y+c.z*5);
 }
 let mode=u32(params.palette.w);
 if(mode==4u){return fract(52.9829189*fract(dot(vec2f(coord),vec2f(0.06711056,0.00583715))));}
 if(mode==5u){
  var h=(bitcast<u32>(coord.x)*0x9e3779b9u) ^ (bitcast<u32>(coord.y)*0x85ebca6bu);
  h=(h^(h>>16u))*0x7feb352du;h=(h^(h>>15u))*0x846ca68bu;
  return min(f32(h^(h>>16u))/4294967296.0,0.99999994);
 }
 let p=bitcast<vec2u>(coord);
 if(mode==6u){return (f32(raa_cel_blue[(p.y&15u)*16u+(p.x&15u)])+0.5)/256.0;}
 let bits=mode+1u;
 var rank=0u;
 for(var i=0u;i<bits;i++) {
  let a=(p.x>>i)&1u; let b=(p.y>>i)&1u;
  rank=rank*4u+((a^b)*2u+b);
 }
 return (f32(rank)+0.5)/f32(1u<<(bits*2u));
}
