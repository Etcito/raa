// Headless Chrome proof, bounded to 55 seconds; both child processes stop by PID.
import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
const port=8187, cdpPort=9247;
const server=spawn(process.execPath,['web/serve.mjs','dist/editor','--port',String(port)],{stdio:'ignore'});
const chromeFlags=JSON.parse(process.env.EDITOR_CHROME_FLAGS || '["--use-webgpu-adapter=vulkan","--use-vulkan=swiftshader","--enable-features=Vulkan","--disable-vulkan-surface"]');
const chrome=spawn(process.env.CHROME_LAUNCHER || 'nmc-chrome',['--port',String(cdpPort),'--url','about:blank','--',...chromeFlags],{stdio:'ignore'});
const timer=setTimeout(()=>{server.kill('SIGTERM');chrome.kill('SIGTERM');process.exitCode=1;},55000);
let ws;
try {
  let tabs;
  for(let i=0;i<50;i++) {
    try {tabs=await(await fetch(`http://127.0.0.1:${cdpPort}/json`)).json();break;}catch{}
    await new Promise(r=>setTimeout(r,100));
  }
  if(!tabs)throw new Error('Chrome CDP did not start');
  ws=new WebSocket(tabs.find(p=>p.type==='page').webSocketDebuggerUrl);
  await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject;});
  let seq=0;const pending=new Map();
  ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){pending.get(m.id)?.(m);pending.delete(m.id);}};
  const call=(method,params={})=>new Promise((resolve,reject)=>{
    const id=++seq;const timeout=setTimeout(()=>reject(new Error(`CDP timeout: ${method}`)),20000);
    pending.set(id,m=>{clearTimeout(timeout);m.error?reject(new Error(JSON.stringify(m.error))):resolve(m.result);});
    ws.send(JSON.stringify({id,method,params}));
  });
  const evaluate=async expression=>{
    const r=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});
    if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));
    return r.result.value;
  };
  await call('Page.enable');await call('Runtime.enable');
  await call('Page.addScriptToEvaluateOnNewDocument',{source:`
    window.__editorGpu={passes:[]};
    const begin=GPUCommandEncoder.prototype.beginRenderPass;
    GPUCommandEncoder.prototype.beginRenderPass=function(d){window.__editorGpu.passes.push(d.label);return begin.call(this,d);};
    const texture=GPUDevice.prototype.createTexture;
    GPUDevice.prototype.createTexture=function(d){window.__editorDevice=this;return texture.call(this,d);};
  `});
  await call('Emulation.setDeviceMetricsOverride',{width:1280,height:720,deviceScaleFactor:1,mobile:false});
  await call('Page.navigate',{url:`http://127.0.0.1:${port}/`});
  await new Promise(r=>setTimeout(r,500));
  const ready=await evaluate('window.editorReady.then(()=>true)');
  if(!ready)throw new Error('Demo did not initialise');
  const adapter=await evaluate('window.editorReady.then(e=>e.engine.stats().adapter)');
  const hardware=!adapter.isFallbackAdapter&&!/swiftshader|llvmpipe|lavapipe|software|basic render/i.test(JSON.stringify(adapter));
  mkdirSync('proofs/captures',{recursive:true});
  mkdirSync('proofs/results',{recursive:true});
  const variants=process.argv.slice(2);
  const cases=variants.length?variants:['overlay-perspective','only-perspective','off-perspective','overlay-orthographic','disabled-perspective','overlay-perspective-replaced'];
  const captures=[];
  for(const variant of cases) {
    const [mode,projection,suffix]=variant.split('-');
    await evaluate('window.__editorGpu.passes=[]');
    const result=await evaluate(`window.editorReady.then(e=>e.capture(${JSON.stringify(mode)},${JSON.stringify(projection)},${suffix==='replaced'?1:0}))`);
    if(result.errors.length)throw new Error(result.errors.join('\n'));
    const nonEmpty=await evaluate('window.editorReady.then(e=>e.engine.probeNonEmpty(performance.now()))');
    if(!nonEmpty)throw new Error('Canvas probe is empty');
    await evaluate('window.__editorDevice.queue.onSubmittedWorkDone()');
    const passes=await evaluate('window.__editorGpu.passes');
    if(mode==='disabled' && passes.some(p=>p?.startsWith('raa.editor.')))throw new Error('Disabled editor recorded GPU passes');
    const shot=await call('Page.captureScreenshot',{format:'png'});
    const file=`proofs/captures/editor-${variant}.png`;
    writeFileSync(file,Buffer.from(shot.data,'base64'));
    captures.push({variant,file,nonEmpty,passes,...result});
  }
  const resizeCycle=await evaluate(`window.editorReady.then(async e=>{
    const sizes=[];
    for(const projection of ['perspective','orthographic']){
      e.camera(projection);
      for(const [w,h] of [[960,540],[1280,600]]){
        e.engine.resize(w,h);e.engine.resize(w,h);
        e.engine.render(performance.now());
        await window.__editorDevice.queue.onSubmittedWorkDone();
        sizes.push({projection,width:document.querySelector('canvas').width,height:document.querySelector('canvas').height});
      }
    }
    return sizes;
  })`);
  const state=await evaluate('window.editorReady.then(e=>e.state())');
  if(!state.stats.frames || state.errors.length)throw new Error('No valid rendered frames: '+JSON.stringify(state));
  writeFileSync('proofs/results/editor.json',JSON.stringify({adapter,hardware,captures,resizeCycle,state},null,2)+'\n');
  console.log(JSON.stringify({hardware,adapter,captures:captures.map(c=>c.file),state}));
} finally {
  clearTimeout(timer);ws?.close();
  // kill() sends a signal to these exact PIDs, never to unrelated browser/server processes.
  chrome.kill('SIGTERM');server.kill('SIGTERM');
}
