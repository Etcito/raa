// Static file server for demo pages: node web/serve.mjs <dir> [--port N]
// Binds 127.0.0.1 only; port 0 (default) picks an ephemeral port.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.wasm': 'application/wasm',
  '.json': 'application/json; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.txt': 'text/plain; charset=utf-8',
};

function parseCli(argv) {
  let dir = null;
  let port = 0;
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--port') {
      port = Number(argv[++i]);
      if (!Number.isInteger(port) || port < 0 || port > 65535) {
        throw new Error(`invalid --port: ${argv[i]}`);
      }
    } else if (dir === null) {
      dir = argv[i];
    } else {
      throw new Error(`unexpected argument: ${argv[i]}`);
    }
  }
  if (dir === null) throw new Error('usage: node web/serve.mjs <dir> [--port N]');
  return { dir, port };
}

function send(res, status, body, headers = {}) {
  res.writeHead(status, {
    'Content-Type': 'text/plain; charset=utf-8',
    'Cache-Control': 'no-store',
    ...headers,
  });
  res.end(body);
}

let cli;
try {
  cli = parseCli(process.argv.slice(2));
} catch (e) {
  console.error(e.message);
  process.exit(2);
}
const root = path.resolve(cli.dir);
if (!fs.existsSync(root) || !fs.statSync(root).isDirectory()) {
  console.error(`not a directory: ${cli.dir}`);
  process.exit(2);
}

const server = http.createServer((req, res) => {
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    send(res, 405, 'method not allowed\n', { Allow: 'GET, HEAD' });
    return;
  }
  const rawPath = (req.url ?? '/').split('?')[0].split('#')[0];
  let decoded;
  try {
    decoded = decodeURIComponent(rawPath);
  } catch {
    send(res, 400, 'bad request\n');
    return;
  }
  if (decoded.includes('\0')) {
    send(res, 400, 'bad request\n');
    return;
  }
  let file = path.resolve(root, '.' + path.posix.sep + decoded);
  if (file !== root && !file.startsWith(root + path.sep)) {
    send(res, 403, 'forbidden\n');
    return;
  }
  let stat;
  try {
    stat = fs.statSync(file);
    if (stat.isDirectory()) {
      file = path.join(file, 'index.html');
      stat = fs.statSync(file);
    }
  } catch {
    send(res, 404, 'not found\n');
    return;
  }
  if (!stat.isFile()) {
    send(res, 404, 'not found\n');
    return;
  }
  const type = MIME[path.extname(file).toLowerCase()] ?? 'application/octet-stream';
  res.writeHead(200, {
    'Content-Type': type,
    'Content-Length': stat.size,
    'Cache-Control': 'no-store',
  });
  if (req.method === 'HEAD') {
    res.end();
    return;
  }
  fs.createReadStream(file)
    .on('error', () => res.destroy())
    .pipe(res);
});

server.listen(cli.port, '127.0.0.1', () => {
  const { port } = server.address();
  console.log(`serving ${cli.dir} on http://127.0.0.1:${port}/`);
});
