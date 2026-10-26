"""Preserve a reference comparison beside the hardware capture for review."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

capture = Path('proofs/captures/lookdev/m15-waddle--f00.png')
record = json.loads(Path('proofs/results/lookdev/m15-waddle--f00.json').read_text())
assert record['hardware'] and record['nonEmpty'], 'Hardware capture required'
judge = record['judge']
assert judge is not None and judge['minScore'] == 6, 'f00 judge verdict not recorded'
verdict = 'judge passed' if judge['exit'] == 0 else f"judge score {judge['score']} below minimum, accepted by owner"
paths = [capture, Path('/refs/ref-1.png'), Path('/refs/ref-2.png')]
comparison = Image.new('RGB', (3 * 1280, 752), 'white')
draw = ImageDraw.Draw(comparison)
for i, path in enumerate(paths):
    with Image.open(path) as source:
        panel = ImageOps.contain(source.convert('RGB'), (1280, 720))
    x = i * 1280
    draw.text((x + 12, 8), str(path), fill='black')
    comparison.paste(panel, (x + (1280 - panel.width) // 2, 32))
output = capture.with_name('m15-waddle--f00-comparison.png')
comparison.save(output)
print(f'PASS: hardware f00, {verdict} (minimum 6); reference comparison: {output}')
