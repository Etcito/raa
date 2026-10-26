"""Compare the hardware f06 capture with the targets and report colour diagnostics."""
import colorsys
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

capture = Path('proofs/captures/lookdev/m15-waddle--f06.png')
record = json.loads(Path('proofs/results/lookdev/m15-waddle--f06.json').read_text())
assert record['hardware'] and record['nonEmpty'], 'Hardware capture required'
paths = [capture, Path('/refs/ref-1.png'), Path('/refs/ref-2.png')]
comparison = Image.new('RGB', (3840, 752), 'white')
draw = ImageDraw.Draw(comparison)
for i, path in enumerate(paths):
    with Image.open(path) as source:
        panel = ImageOps.contain(source.convert('RGB'), (1280, 720))
    x = i * 1280
    draw.text((x + 12, 8), str(path), fill='black')
    comparison.paste(panel, (x + (1280 - panel.width) // 2, 32))
output = capture.with_name('m15-waddle--f06-comparison.png')
comparison.save(output)
with Image.open(capture) as source:
    rgb = source.convert('RGB')
    luminance = sorted(0.2126*r + 0.7152*g + 0.0722*b for r, g, b in rgb.getdata())
    print(f'Frame luminance p01 (0..255): {luminance[len(luminance)//100]:.2f}')
    w, h = rgb.size
    torso = rgb.crop((int(w*.43), int(h*.48), int(w*.56), int(h*.65)))
    blue = [colorsys.rgb_to_hsv(r/255, g/255, b/255) for r, g, b in torso.getdata()
            if b > r * 1.5 and b > g * 1.2]
    if blue:
        med = lambda values: sorted(values)[len(values)//2]
        print(f'Torso blue samples: {len(blue)}, median hue {med([v[0]*360 for v in blue]):.1f}, '
              f'saturation {med([v[1] for v in blue]):.3f}; target hue 230..240, saturation >0.6')
    else:
        print('No blue torso samples: inspect the comparison and camera framing.')
print(f'Reference comparison: {output}')
