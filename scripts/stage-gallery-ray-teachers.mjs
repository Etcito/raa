#!/usr/bin/env node
import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { basename, join } from 'node:path';

const [page, ...sources] = process.argv.slice(2);
if (!page || !sources.length) {
  throw new Error('Usage: node scripts/stage-gallery-ray-teachers.mjs <built gallery> <independent capture dir> [...]');
}
const receiptPath = join(page, 'build.json');
const receipt = JSON.parse(await readFile(receiptPath, 'utf8'));
if (receipt.status !== 'ready' || !receipt.files.includes('lib/gallery-ray-budget.js')) {
  throw new Error('Build the gallery with the ray-budget runner first');
}

const names = ['reflectionHit', 'reflectionMeta', 'reflectionTarget'];
const staged = new Map();
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
for (const source of sources) {
  const manifestBytes = await readFile(join(source, 'manifest.json'));
  const manifest = JSON.parse(manifestBytes);
  const [clip] = manifest.sequences;
  if (manifest.sequences.length !== 1 || !/^joint-(?:development|final-[ab])$/.test(clip.id)
      || manifest.windows?.mode !== 'windowed') {
    throw new Error('Expected one independent windowed Windows trajectory');
  }
  const root = 'shared/ray-teachers/' + clip.id + '/';
  staged.set(root + 'manifest.json', manifestBytes);
  for (const frame of clip.frames) {
    for (const name of names) {
      const record = frame.planes[name];
      if (!record || basename(record.file) !== record.file) throw new Error('Invalid teacher plane ' + name);
      const bytes = await readFile(join(source, record.file));
      if (bytes.length !== record.byteLength || sha(bytes) !== record.sha256) {
        throw new Error('Teacher plane provenance mismatch ' + record.file);
      }
      staged.set(root + record.file, bytes);
    }
  }
  await mkdir(join(page, root), { recursive: true });
}
for (const [file, bytes] of staged) await writeFile(join(page, file), bytes, { flag: 'wx' });
receipt.inputs = { ...receipt.inputs, parentBuildId: receipt.buildId,
  rayTeachers: { ...receipt.inputs.rayTeachers,
    ...Object.fromEntries([...staged].filter(([file]) => file.endsWith('/manifest.json'))
      .map(([file, bytes]) => [file, sha(bytes)])) } };
receipt.buildId = randomUUID();
receipt.files = [...new Set([...receipt.files, ...staged.keys()])].sort();
await writeFile(receiptPath + '.ray-teachers', JSON.stringify(receipt, null, 2) + '\n', { flag: 'wx' });
await rename(receiptPath + '.ray-teachers', receiptPath);
