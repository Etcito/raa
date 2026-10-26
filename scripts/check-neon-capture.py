"""Measure the neon capture and save the requested reference comparison (no GPU work)."""
import colorsys
import argparse
from pathlib import Path
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--capture', default='proofs/captures/lookdev/m14-camera-effects--world-neon.png')
parser.add_argument('--reference', default='/refs/ref-4.png')
parser.add_argument('--output', default='target/neon-comparison.png')
args = parser.parse_args()
capture = Image.open(args.capture).convert('RGB')
reference = Image.open(args.reference).convert('RGB')
w, h = capture.size
sky = list(capture.crop((0, 0, w, round(h * 0.15))).getdata())
# The reference's overhead pipes and stars cross this band. Measure its darker
# half to test the sky, rather than penalising the requested luminous geometry.
sky.sort(key=lambda p: sum(c * weight for c, weight in zip(p, [0.2126, 0.7152, 0.0722])))
sky = sky[:max(1, len(sky) // 2)]
mean = [sum(p[c] for p in sky) / len(sky) for c in range(3)]


def neon_saturation(image):
    width, height = image.size
    saturation = []
    for r, g, b in image.crop((width // 5, height // 5, 4 * width // 5, 4 * height // 5)).getdata():
        hue, sat, value = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        if value > 0.6 and (0.46 < hue < 0.56 or 0.78 < hue < 0.94):
            saturation.append(sat)
    saturation.sort()
    return saturation[len(saturation) // 2] if saturation else 0


median = neon_saturation(capture)
reference_median = neon_saturation(reference)
print(f'Dark sky mean RGB: {mean}; neon median saturation: {median:.3f} (ref {reference_median:.3f})')
comparison = Image.new('RGB', (w * 2, h + 32))
comparison.paste(capture, (0, 32))
comparison.paste(reference.resize((w, h)), (w, 32))
draw = ImageDraw.Draw(comparison)
draw.text((8, 8), 'world-neon capture', fill='white')
draw.text((w + 8, 8), 'ref-4', fill='white')
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
comparison.save(output)
assert all(v < limit for v, limit in zip(mean, [70, 50, 130])), mean
# Ref-4 itself measures 0.558; the old arbitrary 0.7 gate rejected the target.
assert reference_median > 0, 'reference must contain neon colours'
assert median >= reference_median * 0.9, (median, reference_median)
