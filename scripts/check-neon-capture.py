"""Measure the neon capture and save the requested reference comparison (no GPU work)."""
import colorsys
from pathlib import Path
from PIL import Image, ImageDraw

capture = Image.open('proofs/captures/lookdev/m14-camera-effects--world-neon.png').convert('RGB')
reference = Image.open('/refs/ref-4.png').convert('RGB')
w, h = capture.size
sky = list(capture.crop((0, 0, w, round(h * 0.15))).getdata())
mean = [sum(p[c] for p in sky) / len(sky) for c in range(3)]
# Bright magenta/cyan pixels in the arch region, including their bright cores.
saturation = []
for r, g, b in capture.crop((w // 5, h // 5, 4 * w // 5, 4 * h // 5)).getdata():
    hue, sat, value = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    if value > 0.6 and (0.46 < hue < 0.56 or 0.78 < hue < 0.94):
        saturation.append(sat)
saturation.sort()
median = saturation[len(saturation) // 2] if saturation else 0
print(f'Top 15% mean RGB: {mean}; neon-region median saturation: {median:.3f}')
comparison = Image.new('RGB', (w * 2, h + 32))
comparison.paste(capture, (0, 32))
comparison.paste(reference.resize((w, h)), (w, 32))
draw = ImageDraw.Draw(comparison)
draw.text((8, 8), 'world-neon capture', fill='white')
draw.text((w + 8, 8), 'ref-4', fill='white')
output = Path('target/neon-comparison.png')
output.parent.mkdir(parents=True, exist_ok=True)
comparison.save(output)
assert all(v < limit for v, limit in zip(mean, [70, 50, 130])), mean
assert median > 0.7, median
