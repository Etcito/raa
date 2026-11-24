#!/usr/bin/env node
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { validateGalleryFit } from '../examples/global-illumination/lib/gallery-path-model.js';

const [fitDir, outputDir = 'examples/global-illumination/lib/gallery-path-fit'] = process.argv.slice(2);
if (!fitDir) throw new Error('Usage: node scripts/stage-gallery-path-fit.mjs <fit directory> [new staging directory]');
const manifest = await readFile(resolve(fitDir, 'manifest.json'));
const bytes = await readFile(resolve(fitDir, 'weights.fp32le'));
await validateGalleryFit(JSON.parse(manifest), bytes);
await mkdir(outputDir);
await writeFile(resolve(outputDir, 'weights.fp32le'), bytes, { flag: 'wx' });
await writeFile(resolve(outputDir, 'manifest.json'), manifest, { flag: 'wx' });
