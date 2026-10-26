import { loadRaa } from '@etcito/raa';
import { createSdfmLoader } from '@etcito/raa-sdfm';

const canvas = document.querySelector('canvas');
const status = document.querySelector('#status');
const errors = [];
window.addEventListener('error',e=>errors.push(e.message));
window.addEventListener('unhandledrejection',e=>errors.push(String(e.reason)));
// Validation errors are asynchronous; make them available to the capture proof.
const originalRequest = navigator.gpu.requestAdapter.bind(navigator.gpu);
navigator.gpu.requestAdapter = async options => {
  const adapter = await originalRequest(options);
  if (!adapter) return adapter;
  const originalDevice = adapter.requestDevice.bind(adapter);
  adapter.requestDevice = async desc => {
    const device = await originalDevice(desc);
    device.addEventListener('uncapturederror', e => errors.push(e.error.message));
    device.lost.then(info => errors.push('Device lost: '+info.reason+': '+info.message));
    return device;
  };
  return adapter;
};
window.editorReady = (async () => {
  const raa = await loadRaa();
  const engine = await raa.createEngine(canvas, { msaa: 4, maxInstances: 64, maxMaterials: 32 });
  for (const pass of engine.passes()) {
    if (!pass.required && pass.name !== 'editor') engine.setPassEnabled(pass.name, false);
  }
  // Capture the actual adapter upload descriptors without reaching inside RAA.
  const meshes = new Map();
  const loaderEngine = new Proxy(engine, { get(target, key) {
    if (key === 'addMesh') return desc => {
      const id = target.addMesh(desc);
      meshes.set(id, desc);
      return id;
    };
    const value = target[key];
    return typeof value === 'function' ? value.bind(target) : value;
  }});
  const loader = createSdfmLoader({
    engine: loaderEngine, mama: async () => new URL('./mama/', location.href), workers: 1,
  });
  const asset = await loader.load(await (await fetch('./asset.json')).json());
  const spawned = loader.spawn(asset);
  engine.setLight({ direction: [-0.5,-1,-0.6], color: [1,0.95,0.9], intensity: 3 });
  engine.setAmbient({ color: [0.6,0.7,0.9], intensity: 0.5 });
  const parts = Object.entries(spawned.parts);
  for (const [name, part] of parts) {
    document.querySelector('#selection').add(new Option(name, String(part.instance)));
  }
  let selected = parts[0]?.[1].instance;
  const select = value => {
    if (selected !== undefined) engine.setSelectionOutline(selected, null);
    selected = value;
    if (selected !== undefined) engine.setSelectionOutline(selected, {color:[1,0.7,0.1]});
    document.querySelector('#selection').value = selected === undefined ? '' : String(selected);
  };
  select(selected);
  document.querySelector('#selection').onchange = e => select(e.target.value === '' ? undefined : Number(e.target.value));
  document.querySelector('#wire').onchange = e => engine.setWireframe(e.target.value);
  engine.setWireframe('overlay');

  // The fixture camera is fitted using loader bounds, with the same span in both projections.
  const all = [...meshes.values()].flatMap(d => [...d.positions]);
  const min = [Infinity,Infinity,Infinity], max = [-Infinity,-Infinity,-Infinity];
  for(let i=0;i<all.length;i++) { const a=i%3; min[a]=Math.min(min[a],all[i]);max[a]=Math.max(max[a],all[i]); }
  const center = min.map((v,i)=>(v+max[i])/2);
  const span = Math.max(...max.map((v,i)=>v-min[i]), 1);
  function camera(projection = document.querySelector('#projection').value) {
    engine.setCamera({position:[center[0]+span*0.5,center[1]+span*0.4375,center[2]+span*0.9375],
      target:center,up:[0,1,0],projection,orthoHeight:span*0.9,fovYDeg:42,near:0.02,far:span*20});
  }
  camera();
  document.querySelector('#projection').onchange = () => camera();
  let narrow=false;
  document.querySelector('#resize').onclick=()=>{narrow=!narrow;canvas.style.width=narrow?'75%':'100%';};
  let replaced = 0, previous = -1;
  const [meshId, original] = meshes.entries().next().value;
  const replace = phase => {
    const positions = original.positions.slice();
    const factor=phase ? 1.12 : 0.92;
    const normals=original.normals.slice();
    for(let i=0;i<positions.length;i+=3){positions[i]*=factor;normals[i]/=factor;}
    // The same id is shared by a second copy below, so both must change together.
    engine.updateMesh(meshId,{...original,positions,normals});
    replaced++;
  };
  const copyNode = engine.addNode(null,[span*0.12,0,-span*0.18]);
  const second = engine.addMeshInstance(copyNode,meshId);
  engine.setSelectionOutline(second,{color:[0.15,0.85,1]});
  // A generated polygon quad is both an occluder and an edge-list reference:
  // its two triangles have four boundary edges and no diagonal.
  const x0=-span*0.22, x1=-span*0.05, y0=-0.05, y1=span*0.30, z=span*0.17;
  const quad=engine.addMesh({
    positions:new Float32Array([x0,y0,z,x1,y0,z,x1,y1,z,x0,y1,z]),
    normals:new Float32Array([0,0,1,0,0,1,0,0,1,0,0,1]),
    indices:new Uint32Array([0,1,2,0,2,3]),
    edges:new Uint32Array([0,1,1,2,2,3,3,0]),
  });
  const quadInstance=engine.addMeshInstance(engine.addNode(null),quad);
  let stopped=false, sizeKey='';
  function showStatus(w,h) {
    status.textContent=`${asset.name} · ${parts.length+2} instances · mesh ${meshId} replaced ${replaced} times · ${w}×${h}`;
  }
  function render(now) {
    if(stopped)return;
    const w=Math.max(1,Math.round(canvas.clientWidth)),h=Math.max(1,Math.round(canvas.clientHeight));
    const key=w+'x'+h;
    if(key!==sizeKey){engine.resize(w,h);sizeKey=key;}
    const tick=Math.floor(now/1000);
    if(document.querySelector('#replace').checked && tick!==previous){previous=tick;replace(tick%2);}
    engine.render(now);
    showStatus(w,h);
    requestAnimationFrame(render);
  }
  requestAnimationFrame(render);
  window.addEventListener('pagehide',()=>{stopped=true;loader.dispose();engine.dispose();},{once:true});
  return {
    engine, meshId, second, errors, replace, camera, select,
    state:()=>({replaced, meshId, center, span, parts:asset.parts.map(p=>({t:p.t,r:p.r,info:engine.meshInfo(p.mesh)})), instances:parts.length+2, errors:[...errors], stats:engine.stats()}),
    capture: async (mode,projection,phase=0) => {
      document.querySelector('#replace').checked=false;
      document.querySelector('#wire').value=mode==='disabled'?'off':mode;
      replace(phase);
      document.querySelector('#projection').value=projection;
      const disabled=mode==='disabled';
      engine.setWireframe(disabled?'off':mode);camera(projection);
      if(selected!==undefined)engine.setSelectionOutline(selected,disabled?null:{color:[1,0.7,0.1]});
      engine.setSelectionOutline(second,disabled?null:{color:[0.15,0.85,1]});
      document.querySelector('#selection').value=disabled?'':String(selected??'');
      stopped=true;
      engine.resize(Math.round(canvas.clientWidth),Math.round(canvas.clientHeight));
      engine.render(performance.now());
      showStatus(canvas.width,canvas.height);
      await new Promise(requestAnimationFrame);await new Promise(requestAnimationFrame);
      return {replaced,meshId,second,quadInstance,phase,errors:[...errors]};
    },
  };
})().catch(error=>{status.textContent=String(error);errors.push(String(error));throw error;});
