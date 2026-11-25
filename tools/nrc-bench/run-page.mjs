// Windows-only runner: serves a directory on 127.0.0.1, opens one page in a fresh Windows Chrome
// (temporary profile, RTX 3090 checked), waits for `window.__result`, writes result.json.
// Usage (under the GPU lock): node.exe run-page.mjs <outDir> <pageDir> <page>[?query] [timeoutSec] [chrome flags...]
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { createServer } from 'node:http'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { extname, join, normalize } from 'node:path'
import { setTimeout as delay } from 'node:timers/promises'

assert.equal(process.platform, 'win32')
const [out, pageDir, page, timeoutArg, ...flags] = process.argv.slice(2)
const timeoutMs = Number(timeoutArg || 300) * 1000
await mkdir(out, { recursive: true })
const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm', '.wgsl': 'text/plain', '.bin': 'application/octet-stream', '.png': 'image/png' }
const server = createServer(async (request, response) => {
  try {
    const path = normalize(decodeURIComponent(new URL(request.url, 'http://x').pathname)).replace(/^([\\/])+/, '')
    assert(!path.includes('..'))
    const body = await readFile(join(pageDir, path))
    response.writeHead(200, { 'content-type': types[extname(path)] ?? 'application/octet-stream', 'cache-control': 'no-store' })
    response.end(body)
  } catch { response.writeHead(404); response.end() }
})
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
const base = `http://127.0.0.1:${server.address().port}/`
await mkdir('D:/TEMPJOB', { recursive: true })
const profile = await mkdtemp('D:/TEMPJOB/raa-nrc-')
const browser = spawn('C:/Program Files/Google/Chrome/Application/chrome.exe', [
  '--new-window', '--force-device-scale-factor=1', '--disable-background-timer-throttling', '--disable-renderer-backgrounding',
  '--disable-backgrounding-occluded-windows', '--no-first-run', '--no-default-browser-check', '--enable-unsafe-webgpu',
  '--disable-software-rasterizer', '--remote-debugging-port=0', '--window-size=1280,800', ...flags,
  `--user-data-dir=${profile}`, 'about:blank',
], { stdio: 'ignore', windowsHide: false })
const log = []
let status = 0
let socket
const hardStop = setTimeout(() => { console.error('runner wall timeout'); try { browser.kill() } catch {} ; process.exit(3) }, timeoutMs + 60000)
try {
  let port
  for (let i = 0; i < 150 && !port; i++) {
    try { [port] = (await readFile(join(profile, 'DevToolsActivePort'), 'utf8')).split(/\r?\n/) } catch {}
    if (!port) await delay(100)
  }
  assert.match(port ?? '', /^\d+$/)
  const info = await (await fetch(`http://127.0.0.1:${port}/json/version`)).json()
  assert.doesNotMatch(info['User-Agent'] ?? '', /Headless/i)
  socket = new WebSocket(info.webSocketDebuggerUrl)
  await new Promise((resolve, reject) => { socket.addEventListener('open', resolve, { once: true }); socket.addEventListener('error', reject, { once: true }) })
  let serial = 0
  const pending = new Map()
  socket.addEventListener('message', ({ data }) => {
    const message = JSON.parse(data)
    const reply = pending.get(message.id)
    if (reply) { pending.delete(message.id); message.error ? reply.reject(new Error(message.error.message)) : reply.resolve(message.result) }
    if (message.method === 'Runtime.consoleAPICalled') {
      const line = message.params.args.map((a) => a.value ?? a.description).join(' ')
      log.push(`[${message.params.type}] ${line}`)
      if (line.startsWith('[nrc]')) console.log(line)
    }
    if (message.method === 'Log.entryAdded') log.push(`[log:${message.params.entry.level}] ${message.params.entry.text}`)
    if (message.method === 'Runtime.exceptionThrown') log.push('[exception] ' + (message.params.exceptionDetails.exception?.description ?? message.params.exceptionDetails.text))
  })
  const call = (method, params, sessionId) => new Promise((resolve, reject) => {
    const id = ++serial
    pending.set(id, { resolve, reject })
    socket.send(JSON.stringify({ id, method, params, sessionId }))
  })
  const gpu = (await call('SystemInfo.getInfo')).gpu
  const renderer = gpu.auxAttributes.glRenderer
  assert.match(renderer, /RTX 3090/i)
  assert.doesNotMatch(renderer, /software|swiftshader|warp/i)
  const { targetInfos } = await call('Target.getTargets')
  const { sessionId } = await call('Target.attachToTarget', { targetId: targetInfos.find((t) => t.type === 'page').targetId, flatten: true })
  const send = (method, params = {}) => call(method, params, sessionId)
  await send('Runtime.enable'); await send('Page.enable'); await send('Log.enable')
  await send('Page.navigate', { url: base + page })
  const end = performance.now() + timeoutMs
  let value
  for (;;) {
    const { result } = await send('Runtime.evaluate', { expression: 'JSON.stringify(window.__result ?? (window.__error ? {error: String(window.__error)} : null))', returnByValue: true })
    value = JSON.parse(result.value ?? 'null')
    if (value) break
    if (performance.now() > end) { value = { error: 'page timed out' }; break }
    await delay(500)
  }
  if (value.screenshot) {
    const shot = await send('Page.captureScreenshot', { format: 'png' })
    await writeFile(join(out, 'page.png'), Buffer.from(shot.data, 'base64'))
  }
  await writeFile(join(out, 'result.json'), JSON.stringify({ renderer, chrome: info.Browser, flags, page, result: value }, null, 1))
  if (value.error) { status = 1; console.error('page error:', value.error) }
} catch (error) {
  status = 2
  console.error(error)
} finally {
  await writeFile(join(out, 'console.log'), log.join('\n'))
  try { socket?.close() } catch {}
  browser.kill()
  server.close()
  await delay(1500)
  await rm(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 500 }).catch(() => {})
  clearTimeout(hardStop)
}
process.exit(status)
